"""ハンガリアンマッチングと DETR の損失を自前で実装し、ライブラリ（transformers）の結果と一致するか確かめる。

CPU で数十秒で終わる。学習もデータのダウンロードも不要。
  python hungarian_check.py

確認すること:
  1. 小さな例（予測5個・正解3個）で、マッチングのコスト行列・割り当て・損失を手計算に近い形で実装し、
     ライブラリの HungarianMatcher / ImageLoss と一致するか。
     割り当ては「全通りを総当たり」でも求め、ハンガリアン法（scipy）が本当に最小コストを返しているかも確かめる。
  2. 同じ物体に重複した予測が2つあるとき、片方だけが正解に割り当てられ、もう片方は「物体なし」を
     教師にされる（= 重複が損失で罰せられる）ことを数値で見る。これが NMS 不要の理由の核心。
  3. ランダムな大きめの例（クエリ100個、バッチ2）でも自前実装とライブラリが一致するか。
  4. model.py の detr_loss（fp16 対応のために順伝播と損失を分けた実装）が、
     ライブラリの model(..., labels=...) の損失と一致するか。

結果は results/hungarian_check/check.json に保存する。
"""

import itertools
import os

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import DetrConfig, DetrForObjectDetection
from transformers.loss.loss_for_object_detection import HungarianMatcher, ImageLoss

from model import detr_loss
from utils import save_json, seed_everything

# DETR（とライブラリの既定値）のコストの重み・損失の重み
CLASS_COST, BBOX_COST, GIOU_COST = 1.0, 5.0, 2.0
LOSS_W = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
EOS_COEF = 0.1  # 「物体なし」クラスの分類損失の重み（物体なしが圧倒的に多いので軽くする）


# ---------------------------------------------------------------------------
# 自前実装
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def my_giou(a, b):
    """Generalized IoU。a: (N, 4), b: (M, 4) いずれも xyxy -> (N, M)

    GIoU = IoU - (C - U) / C
      U: 2つの箱の和集合の面積、C: 2つの箱を両方囲む最小の箱の面積。
    IoU は箱が離れていると常に 0 で勾配が出ないが、GIoU は「離れ具合」に応じて -1 まで下がるので、
    重なっていない予測にも「近づけ」という勾配が出る。
    """
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])                  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])                  # (M,)
    lt = torch.max(a[:, None, :2], b[None, :, :2])                      # (N, M, 2) 共通部分の左上
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])                      # (N, M, 2) 共通部分の右下
    inter = (rb - lt).clamp(min=0).prod(-1)                             # (N, M)
    union = area_a[:, None] + area_b[None, :] - inter                   # (N, M)
    iou = inter / union
    lt_c = torch.min(a[:, None, :2], b[None, :, :2])                    # 外接箱の左上
    rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])                    # 外接箱の右下
    area_c = (rb_c - lt_c).clamp(min=0).prod(-1)                        # (N, M)
    return iou - (area_c - union) / area_c


def my_cost_matrix(logits, boxes, tgt_labels, tgt_boxes):
    """マッチングのコスト行列 C: (Q, M)。C[i, j] = 予測 i を正解 j に割り当てたときの「まずさ」。

    C = 1 * (-p_i(c_j)) + 5 * ||b_i - b_j||_1 + 2 * (-GIoU(b_i, b_j))
    分類コストは損失（-log p）ではなく -p を使う（元論文の実装と同じ。スケールがボックスのコストとそろう）。
    """
    prob = logits.softmax(-1)                                    # (Q, C+1)
    cost_class = -prob[:, tgt_labels]                            # (Q, M)
    cost_bbox = (boxes[:, None, :] - tgt_boxes[None, :, :]).abs().sum(-1)   # (Q, M) L1 距離
    cost_giou = -my_giou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(tgt_boxes))  # (Q, M)
    return CLASS_COST * cost_class + BBOX_COST * cost_bbox + GIOU_COST * cost_giou


def brute_force_assignment(cost):
    """全ての「正解 M 個 -> 予測 Q 個 への単射」を総当たりして、コスト合計が最小の割り当てを返す。

    Q=5, M=3 なら 5*4*3 = 60 通り。ハンガリアン法はこれを O(n^3) で厳密に解く。
    戻り値: (pred_idx, tgt_idx, 最小コスト)  ※ pred_idx の昇順に並べる（ライブラリと同じ順）
    """
    q, m = cost.shape
    best, best_perm = float("inf"), None
    for perm in itertools.permutations(range(q), m):            # perm[j] = 正解 j に割り当てる予測
        c = sum(cost[perm[j], j].item() for j in range(m))
        if c < best:
            best, best_perm = c, perm
    pairs = sorted((best_perm[j], j) for j in range(m))
    return [p for p, _ in pairs], [t for _, t in pairs], best


def my_losses(logits, boxes, targets, indices, num_classes):
    """マッチング結果を使って DETR の損失を計算する（バッチ対応）。

    logits: (B, Q, C+1), boxes: (B, Q, 4), indices: 画像ごとの (pred_idx, tgt_idx)
    - loss_ce: 全クエリの分類のクロスエントロピー。マッチしたクエリの教師は正解クラス、
               それ以外のクエリの教師は「物体なし」（インデックス C）。物体なしの重みは 0.1。
               重み付き平均 = Σ w_i * (-log p_i(y_i)) / Σ w_i
    - loss_bbox: マッチしたペアだけの L1 距離の合計 / 正解ボックス数
    - loss_giou: マッチしたペアだけの (1 - GIoU) の合計 / 正解ボックス数
    """
    b, q, _ = logits.shape
    num_boxes = max(sum(len(t["class_labels"]) for t in targets), 1)
    target_classes = torch.full((b, q), num_classes, dtype=torch.long)      # 既定は全部「物体なし」
    l1_sum, giou_sum = 0.0, 0.0
    for i, (pi, ti) in enumerate(indices):
        pi, ti = torch.as_tensor(pi), torch.as_tensor(ti)
        target_classes[i, pi] = targets[i]["class_labels"][ti]
        src, tgt = boxes[i, pi], targets[i]["boxes"][ti]                       # (m, 4), (m, 4)
        l1_sum = l1_sum + (src - tgt).abs().sum()
        giou_sum = giou_sum + (1 - torch.diag(my_giou(cxcywh_to_xyxy(src), cxcywh_to_xyxy(tgt)))).sum()
    weight = torch.ones(num_classes + 1)
    weight[-1] = EOS_COEF
    nll = -logits.log_softmax(-1).gather(-1, target_classes[..., None])[..., 0]   # (B, Q)
    w = weight[target_classes]                                                   # (B, Q)
    loss_ce = (w * nll).sum() / w.sum()
    return {"loss_ce": loss_ce, "loss_bbox": l1_sum / num_boxes, "loss_giou": giou_sum / num_boxes}, \
        target_classes


def library_matcher_and_loss(logits, boxes, targets, num_classes):
    matcher = HungarianMatcher(class_cost=CLASS_COST, bbox_cost=BBOX_COST, giou_cost=GIOU_COST)
    criterion = ImageLoss(matcher=matcher, num_classes=num_classes, eos_coef=EOS_COEF, losses=["labels", "boxes"])
    outputs = {"logits": logits, "pred_boxes": boxes}
    indices = matcher(outputs, targets)
    losses = criterion(outputs, targets)
    return indices, losses


# ---------------------------------------------------------------------------
# 1 & 2. 小さな例
# ---------------------------------------------------------------------------

def small_example():
    # クラス: 0=cat, 1=dog, 2=person, 3=car, 4=物体なし（no-object）
    names = ["cat", "dog", "person", "car", "no-object"]
    num_classes = 4
    # 正解 3 個（正規化 cxcywh）
    tgt = {
        "class_labels": torch.tensor([1, 2, 3]),                       # dog, person, car
        "boxes": torch.tensor([[0.30, 0.40, 0.20, 0.30],                # g0: dog
                               [0.70, 0.50, 0.20, 0.60],                # g1: person
                               [0.50, 0.85, 0.40, 0.20]]),              # g2: car
    }
    # 予測 5 個（クエリ）。p1 と p2 は同じ犬に対する「重複した予測」。
    boxes = torch.tensor([[0.68, 0.52, 0.22, 0.58],     # p0: person の近く
                          [0.31, 0.41, 0.19, 0.28],     # p1: dog の近く
                          [0.29, 0.38, 0.23, 0.32],     # p2: dog の近く（重複）
                          [0.10, 0.10, 0.05, 0.05],     # p3: 何もない所
                          [0.52, 0.83, 0.38, 0.22]])    # p4: car の近く（クラスは cat と迷っている）
    logits = torch.tensor([[0.0, 0.2, 3.0, 0.1, 0.5],   # p0: person らしい
                           [0.1, 3.0, 0.0, 0.0, 0.3],   # p1: dog らしい
                           [0.2, 2.5, 0.0, 0.1, 0.4],   # p2: dog らしい（p1 よりやや弱い）
                           [0.0, 0.1, 0.1, 0.0, 3.0],   # p3: 物体なしらしい
                           [1.0, 0.0, 0.0, 1.2, 0.2]])  # p4: car と cat で迷っている

    cost = my_cost_matrix(logits, boxes, tgt["class_labels"], tgt["boxes"])     # (5, 3)
    bf_pred, bf_tgt, bf_cost = brute_force_assignment(cost)
    sp_pred, sp_tgt = linear_sum_assignment(cost.numpy())
    lib_indices, lib_losses = library_matcher_and_loss(logits[None], boxes[None], [tgt], num_classes)
    my_l, target_classes = my_losses(logits[None], boxes[None], [tgt], [(bf_pred, bf_tgt)], num_classes)

    print("=== 1. 小さな例（予測5個・正解3個）===")
    print("コスト行列 C[i, j]（行=予測 p0..p4, 列=正解 g0:dog g1:person g2:car）")
    for i, row in enumerate(cost):
        print(f"  p{i}: " + "  ".join(f"{v:7.3f}" for v in row.tolist()))
    print(f"総当たり    : pred={bf_pred} tgt={bf_tgt} 最小コスト={bf_cost:.4f}")
    print(f"scipy       : pred={sp_pred.tolist()} tgt={sp_tgt.tolist()} "
          f"コスト={cost[sp_pred, sp_tgt].sum().item():.4f}")
    print(f"ライブラリ  : pred={lib_indices[0][0].tolist()} tgt={lib_indices[0][1].tolist()}")
    print("各クエリの分類の教師:", [f"p{i}->{names[c]}" for i, c in enumerate(target_classes[0].tolist())])
    for k in LOSS_W:
        print(f"  {k}: 自前={my_l[k].item():.6f}  ライブラリ={lib_losses[k].item():.6f}")

    # 2. 重複予測の扱い: 重複していた p1 / p2 のどちらが「物体なし」を教師にされたか
    dup_loser = [p for p in (1, 2) if p not in bf_pred][0]
    p = logits.softmax(-1)
    print(f"\n=== 2. 重複予測 ===\n犬に対する重複予測 p1 / p2 のうち、マッチしたのは p{3 - dup_loser}、"
          f"p{dup_loser} は教師が「物体なし」になる。")
    print(f"  p{dup_loser} の現在の確率: dog={p[dup_loser, 1]:.3f}, no-object={p[dup_loser, 4]:.3f} "
          f"-> 学習が進むと no-object の確率を上げる方向に勾配が流れる（= 重複が自然に抑制される）")

    match_ok = (bf_pred == lib_indices[0][0].tolist() and bf_tgt == lib_indices[0][1].tolist()
                and bf_pred == sp_pred.tolist())
    loss_ok = all(torch.allclose(my_l[k], lib_losses[k], atol=1e-6) for k in LOSS_W)
    return {
        "cost_matrix": [[round(v, 6) for v in row] for row in cost.tolist()],
        "brute_force": {"pred": bf_pred, "tgt": bf_tgt, "min_cost": bf_cost},
        "scipy": {"pred": sp_pred.tolist(), "tgt": sp_tgt.tolist()},
        "library": {"pred": lib_indices[0][0].tolist(), "tgt": lib_indices[0][1].tolist()},
        "target_classes": [names[c] for c in target_classes[0].tolist()],
        "my_losses": {k: my_l[k].item() for k in LOSS_W},
        "library_losses": {k: lib_losses[k].item() for k in LOSS_W},
        "duplicate_query_assigned_no_object": f"p{dup_loser}",
        "matching_equal": match_ok, "losses_equal": loss_ok,
    }


# ---------------------------------------------------------------------------
# 3. ランダムな大きめの例
# ---------------------------------------------------------------------------

def random_example(num_trials=20):
    num_classes, q = 20, 100
    all_match, max_diff = True, 0.0
    for trial in range(num_trials):
        g = torch.Generator().manual_seed(trial)
        b = 2
        logits = torch.randn(b, q, num_classes + 1, generator=g) * 2
        boxes = torch.rand(b, q, 4, generator=g) * 0.5 + 0.25           # 正規化 cxcywh
        targets = []
        for _ in range(b):
            m = int(torch.randint(1, 8, (1,), generator=g))
            targets.append({"class_labels": torch.randint(0, num_classes, (m,), generator=g),
                            "boxes": torch.rand(m, 4, generator=g) * 0.4 + 0.3})
        my_indices = []
        for i in range(b):
            c = my_cost_matrix(logits[i], boxes[i], targets[i]["class_labels"], targets[i]["boxes"])
            pi, ti = linear_sum_assignment(c.numpy())
            my_indices.append((pi.tolist(), ti.tolist()))
        lib_indices, lib_losses = library_matcher_and_loss(logits, boxes, targets, num_classes)
        my_l, _ = my_losses(logits, boxes, targets, my_indices, num_classes)
        for (mp, mt), (lp, lt) in zip(my_indices, lib_indices):
            all_match &= (mp == lp.tolist() and mt == lt.tolist())
        max_diff = max(max_diff, max(abs(my_l[k].item() - lib_losses[k].item()) for k in LOSS_W))
    print(f"\n=== 3. ランダムな例（Q=100, バッチ2, {num_trials}回）===")
    print(f"  割り当てが全て一致: {all_match}, 損失の最大絶対誤差: {max_diff:.2e}")
    return {"num_trials": num_trials, "matching_equal": all_match, "max_abs_loss_diff": max_diff}


# ---------------------------------------------------------------------------
# 4. detr_loss（順伝播と損失を分けた実装）とライブラリの model(labels=...) の一致
# ---------------------------------------------------------------------------

def model_loss_example():
    config = DetrConfig(num_labels=20, auxiliary_loss=True)
    model = DetrForObjectDetection(config).eval()       # dropout を切って決定的にする
    g = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(2, 3, 160, 224, generator=g)
    pixel_mask = torch.ones(2, 160, 224, dtype=torch.long)
    pixel_mask[1, :, 192:] = 0                            # 2枚目は右側がパディング
    labels = [{"class_labels": torch.tensor([3, 14]), "boxes": torch.tensor([[0.3, 0.4, 0.2, 0.3],
                                                                              [0.6, 0.5, 0.3, 0.6]])},
              {"class_labels": torch.tensor([7]), "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]
    with torch.no_grad():
        lib = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        mine, mine_dict, _, _ = detr_loss(model, pixel_values, pixel_mask, labels, use_amp=False)
    diff = abs(lib.loss.item() - mine.item())
    print("\n=== 4. detr_loss とライブラリの model(labels=...) ===")
    print(f"  ライブラリ={lib.loss.item():.6f}  detr_loss={mine.item():.6f}  差={diff:.2e}"
          f"  （補助損失の項の数: {sum(1 for k in mine_dict if k.startswith('loss_ce'))} 層分）")
    return {"library_loss": lib.loss.item(), "detr_loss": mine.item(), "abs_diff": diff}


def main():
    seed_everything(0)
    result = {"small_example": small_example(), "random_example": random_example(),
              "model_loss": model_loss_example()}
    ok = (result["small_example"]["matching_equal"] and result["small_example"]["losses_equal"]
          and result["random_example"]["matching_equal"] and result["random_example"]["max_abs_loss_diff"] < 1e-5
          and result["model_loss"]["abs_diff"] < 1e-4)
    result["all_checks_passed"] = ok
    print(f"\nall checks passed: {ok}")
    save_json(result, os.path.join("results", "hungarian_check", "check.json"))


if __name__ == "__main__":
    main()
