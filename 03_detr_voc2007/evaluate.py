"""学習済み DETR を VOC 2007 test 全体で評価し、可視化する。

使い方:
  python evaluate.py --condition A          # 条件 A のチェックポイントを評価
  python evaluate.py --condition B
  python evaluate.py --compare              # A と B の結果を並べて比較図・表を作る
  python evaluate.py --condition A --quick  # 動作確認（train.py --quick の後に実行）

出力（results/<条件名>/）:
  eval_metrics.json        … mAP@0.5, mAP@[.5:.95], クラス別 AP@0.5, NMS を後から足したときの mAP, 重複検出の統計
  detections.png           … 検出結果（赤: 予測, 緑破線: 正解）
  cross_attention_*.png    … デコーダ最終層の cross-attention マップ（どのクエリが画像のどこを見ているか）
  query_specialization.png … 各物体クエリが test 全体でどの位置・形の箱を出しているか
出力（results/comparison/）:
  comparison_curves.png, summary.json, summary.md
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms, box_iou

from dataset import IMAGENET_MEAN, IMAGENET_STD, VOC_CLASSES, build_datasets, collate_fn
from model import CONDITIONS, build_model_from_checkpoint, postprocess, resnet_feature_size
from utils import compute_map, run_inference, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DETR on Pascal VOC 2007 test")
    p.add_argument("--condition", choices=["A", "B"])
    p.add_argument("--compare", action="store_true", help="A と B の結果を比較する")
    p.add_argument("--quick", action="store_true", help="動作確認（CPU・データの一部）")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--ckpt", default=None, help="省略時は results/<条件名>/checkpoints/last.pt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_min_size", type=int, default=512)
    p.add_argument("--max_size", type=int, default=800)
    p.add_argument("--max_test_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--score_thresh", type=float, default=0.7, help="可視化で表示する検出のスコア閾値")
    p.add_argument("--num_vis", type=int, default=8, help="検出結果を可視化する画像の枚数")
    p.add_argument("--num_attn_images", type=int, default=4, help="cross-attention を可視化する画像の枚数")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.compare and args.condition is None:
        p.error("--condition か --compare のどちらかを指定してください")
    # build_datasets が参照する学習用の設定（評価では使わないが値は必要）
    args.train_min_sizes = [args.eval_min_size]
    args.max_train_images = 1
    args.val_images = 1
    if args.quick:
        args.max_test_images = args.max_test_images or 8
        args.eval_min_size = 256
        args.train_min_sizes = [256]
        args.max_size = 384
        args.batch_size = 2
        args.num_workers = 0
        args.num_vis = min(args.num_vis, 4)
        args.num_attn_images = min(args.num_attn_images, 2)
    return args


def denormalize(pixel_values):
    """(3, H, W) の正規化済みテンソル -> (H, W, 3) の [0,1] 画像（表示用）"""
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (pixel_values.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# NMS なしで動くかの検証
# ---------------------------------------------------------------------------

def apply_nms(preds, iou_thresh):
    """DETR の出力に、従来手法と同じクラスごとの NMS を「後から」かけた検出結果を返す。"""
    out = []
    for p in preds:
        keep = batched_nms(p["boxes"], p["scores"], p["labels"], iou_thresh)
        out.append({k: v[keep] for k, v in p.items()})
    return out


def duplicate_stats(preds, score_thresh=0.5, iou_thresh=0.7):
    """重複検出の統計。スコア > score_thresh の予測のうち、同じクラスで IoU > iou_thresh のペアを数える。

    NMS を使う従来手法なら、NMS 前にはこのペアが大量にある。DETR が本当に重複を出さないならほぼ 0 になる。
    """
    n_pairs, n_images_with_dup, n_kept = 0, 0, 0
    for p in preds:
        keep = p["scores"] > score_thresh
        boxes, labels = p["boxes"][keep], p["labels"][keep]
        n_kept += int(keep.sum())
        if len(boxes) < 2:
            continue
        iou = box_iou(boxes, boxes)                                # (K, K)
        same = labels[:, None] == labels[None, :]                  # (K, K) 同じクラスか
        upper = torch.triu(torch.ones_like(same), diagonal=1)      # 自分自身と重複カウントを除く
        pairs = int(((iou > iou_thresh) & same & upper).sum())
        n_pairs += pairs
        n_images_with_dup += int(pairs > 0)
    return {"score_thresh": score_thresh, "iou_thresh": iou_thresh, "num_images": len(preds),
            "num_detections_above_thresh": n_kept, "duplicate_pairs": n_pairs,
            "images_with_duplicates": n_images_with_dup}


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------

def draw_boxes(ax, boxes, labels, scores=None, color="red", linestyle="-"):
    for i, (b, l) in enumerate(zip(boxes.tolist(), labels.tolist())):
        x1, y1, x2, y2 = b
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, lw=2, ls=linestyle))
        text = VOC_CLASSES[l] if scores is None else f"{VOC_CLASSES[l]} {scores[i]:.2f}"
        ax.text(x1, y1, text, color="white", fontsize=7, va="bottom",
                bbox=dict(facecolor=color, alpha=0.7, pad=1, lw=0))


def plot_detections(dataset, preds, num, score_thresh, path, title):
    """先頭 num 枚の test 画像に、予測（赤）と正解（緑破線）を描く。"""
    cols = 4
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    for ax in np.array(axes).reshape(-1):
        ax.axis("off")
    for i, ax in zip(range(num), np.array(axes).reshape(-1)):
        pixel_values, target = dataset[i]
        img = denormalize(pixel_values)
        h, w = target["orig_size"].tolist()
        # 予測・正解は元画像の座標なので、表示用のリサイズ後の座標に合わせる
        sx, sy = img.shape[1] / w, img.shape[0] / h
        scale = torch.tensor([sx, sy, sx, sy])
        ax.imshow(img)
        draw_boxes(ax, target["orig_boxes"] * scale, target["orig_labels"], color="limegreen", linestyle="--")
        p = preds[i]
        keep = p["scores"] > score_thresh
        draw_boxes(ax, p["boxes"][keep] * scale, p["labels"][keep], p["scores"][keep].tolist(), color="red")
        ax.set_title(f"{target['image_id']}: {int(keep.sum())} dets (score>{score_thresh})", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def plot_cross_attention(model, dataset, idx, device, path, title, num_queries_show=4):
    """デコーダ最終層の cross-attention マップを可視化する。

    cross-attention では、各物体クエリ（Q=100個）が Query、エンコーダが出力した画像特徴（h*w 個）が Key/Value。
    attention の重み (heads, Q, h*w) を8ヘッドで平均し、(Q, h, w) に並べ直すと、
    「そのクエリが画像のどこを見て予測を出したか」がわかる。スコアの高い上位のクエリを表示する。
    """
    model.eval()
    pixel_values, target = dataset[idx]
    _, H, W = pixel_values.shape
    out = model(pixel_values=pixel_values[None].to(device), output_attentions=True)
    # cross_attentions: デコーダ層の数（6）のタプル。各要素 (1, heads=8, Q=100, h*w)
    attn = out.cross_attentions[-1][0].float().mean(0)          # (8, 100, h*w) -> (100, h*w)
    fh, fw = resnet_feature_size(H, W)
    assert fh * fw == attn.shape[-1], (fh, fw, attn.shape)
    attn = attn.reshape(-1, fh, fw).cpu()                        # (100, h, w)

    pred = postprocess(out.logits, out.pred_boxes, torch.tensor([[H, W]], device=device))[0]
    scores = pred["scores"].cpu()
    top = scores.argsort(descending=True)[:num_queries_show]     # スコア上位のクエリ番号

    img = denormalize(pixel_values)
    fig, axes = plt.subplots(1, num_queries_show + 1, figsize=(4 * (num_queries_show + 1), 4))
    axes[0].imshow(img)
    colors = plt.cm.tab10(np.arange(num_queries_show))
    for k, q in enumerate(top.tolist()):
        b = pred["boxes"][q].cpu().tolist()
        axes[0].add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                             color=colors[k], lw=2))
        # 注意マップを入力画像の大きさに拡大して重ねる: (h, w) -> (H, W)
        a = torch.nn.functional.interpolate(attn[q][None, None], size=(H, W), mode="bilinear",
                                            align_corners=False)[0, 0].numpy()
        axes[k + 1].imshow(img)
        axes[k + 1].imshow(a, cmap="jet", alpha=0.5)
        axes[k + 1].add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                                 color=colors[k], lw=2))
        axes[k + 1].set_title(f"query {q}: {VOC_CLASSES[int(pred['labels'][q])]} {scores[q]:.2f}", fontsize=10)
    axes[0].set_title(f"{target['image_id']} (top-{num_queries_show} queries)", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_query_specialization(preds, orig_sizes, path, title, num_queries=20):
    """物体クエリごとに、test 全体で出した箱の中心位置を散布図にする（元論文の図7と同様の分析）。

    点の色: 緑=小さい箱, 赤=横長の大きい箱, 青=縦長の大きい箱。
    クエリが「画像のこのあたりの、この形の物体」を担当するように分業しているかがわかる。
    """
    cols = 5
    rows = int(np.ceil(num_queries / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for q, ax in zip(range(num_queries), np.array(axes).reshape(-1)):
        xs, ys, cs = [], [], []
        for p, (h, w) in zip(preds, orig_sizes):
            x1, y1, x2, y2 = p["boxes"][q].tolist()
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            xs.append((x1 + x2) / 2 / w)
            ys.append((y1 + y2) / 2 / h)
            if bw * bh < 0.1:
                cs.append("green")
            elif bw >= bh:
                cs.append("red")
            else:
                cs.append("blue")
        ax.scatter(xs, ys, c=cs, s=2, alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"query {q}", fontsize=9)
    fig.suptitle(title + "  (green: small, red: large wide, blue: large tall)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------------------

def evaluate_condition(args):
    seed_everything(args.seed)
    device = torch.device("cpu" if args.quick or not torch.cuda.is_available() else "cuda")
    use_amp = device.type == "cuda"
    name = CONDITIONS[args.condition] + ("_quick" if args.quick else "")
    out_dir = os.path.join(args.results_dir, name)
    ckpt_path = args.ckpt or os.path.join(out_dir, "checkpoints", "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # cross-attention の重みを取り出すため、Attention を明示的な実装（eager）にする
    # （SDPA などの高速実装は Attention の重み行列を返さない）
    model = build_model_from_checkpoint(ckpt, attn_implementation="eager").to(device).eval()
    print(f"[eval] {name} weights: {ckpt['model_info']['weights']}")

    _, _, test_ds, data_source = build_datasets(args)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=args.num_workers)
    print(f"[eval] data={data_source} test={len(test_ds)} images")

    # ---- mAP（NMS なし、全100クエリをそのまま評価） ----
    preds, targets, losses = run_inference(model, loader, device, use_amp)
    metrics = compute_map(preds, targets, per_class=True)
    metrics["ap50_per_class"] = {VOC_CLASSES[c]: v for c, v in metrics.pop("ap50_per_class_raw").items()}
    print(f"[eval] mAP@0.5={metrics['map_50']:.4f}  mAP@[.5:.95]={metrics['map']:.4f}")

    # ---- NMS を後から足すと mAP は変わるか？ 重複検出はどれくらいあるか？ ----
    nms_results = {}
    for th in (0.5, 0.7):
        m = compute_map(apply_nms(preds, th), targets)
        nms_results[f"nms_iou_{th}"] = {"map": m["map"], "map_50": m["map_50"]}
        print(f"[eval] +NMS(IoU {th}): mAP@0.5={m['map_50']:.4f} mAP@[.5:.95]={m['map']:.4f}")
    dup = duplicate_stats(preds)
    print(f"[eval] duplicates: {dup}")

    result = {"condition": args.condition, "name": name, "data_source": data_source,
              "num_test_images": len(test_ds), "weights": ckpt["model_info"]["weights"],
              "epochs_trained": ckpt["args"]["epochs"], "test_loss": losses,
              "metrics_no_nms": metrics, "metrics_with_nms": nms_results, "duplicate_stats": dup,
              "note": "mAP は torchmetrics（pycocotools、101点補間）。difficult は iscrowd として無視。"}
    save_json(result, os.path.join(out_dir, "eval_metrics.json"))

    # ---- 可視化 ----
    plot_detections(test_ds, preds, min(args.num_vis, len(test_ds)), args.score_thresh,
                    os.path.join(out_dir, "detections.png"), f"{name}: detections (no NMS)")
    for i in range(min(args.num_attn_images, len(test_ds))):
        plot_cross_attention(model, test_ds, i, device, os.path.join(out_dir, f"cross_attention_{i}.png"),
                             f"{name}: last decoder layer cross-attention (mean over 8 heads)")
    plot_query_specialization(preds, _orig_sizes_fast(test_ds), os.path.join(out_dir, "query_specialization.png"),
                              f"{name}: predicted box centers per query over test set")
    print(f"[eval] done -> {out_dir}")


def _orig_sizes_fast(test_ds):
    """画像を読まずに、アノテーション XML から元画像サイズ (H, W) を取る。"""
    import xml.etree.ElementTree as ET
    if not hasattr(test_ds, "ids"):
        return [tuple(test_ds[i][1]["orig_size"].tolist()) for i in range(len(test_ds))]
    sizes = []
    for image_id in test_ds.ids:
        size = ET.parse(os.path.join(test_ds.root, "Annotations", f"{image_id}.xml")).getroot().find("size")
        sizes.append((float(size.find("height").text), float(size.find("width").text)))
    return sizes


# ---------------------------------------------------------------------------
# A と B の比較
# ---------------------------------------------------------------------------

def compare(args):
    import json
    suffix = "_quick" if args.quick else ""
    runs = {}
    for cond, base in CONDITIONS.items():
        d = os.path.join(args.results_dir, base + suffix)
        hist_path, eval_path = os.path.join(d, "history.json"), os.path.join(d, "eval_metrics.json")
        if not os.path.exists(hist_path):
            print(f"[compare] {hist_path} がないのでスキップ")
            continue
        runs[cond] = {"name": base + suffix, "history": json.load(open(hist_path, encoding="utf-8")),
                      "eval": json.load(open(eval_path, encoding="utf-8")) if os.path.exists(eval_path) else None}
    if not runs:
        raise SystemExit("比較できる結果がありません")

    out_dir = os.path.join(args.results_dir, "comparison" + suffix)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    style = {"A": "o", "B": "s"}  # マーカーの形で条件を区別する
    for cond, r in runs.items():
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train"]["loss"] for h in r["history"]], marker=style[cond], label=f"{r['name']} train")
        axes[0].plot(ep, [h["val"]["loss"] for h in r["history"]], marker=style[cond], ls="--", alpha=0.6,
                     label=f"{r['name']} val")
        axes[1].plot(ep, [h["train"]["loss_ce"] for h in r["history"]], marker=style[cond], label=r["name"])
        axes[2].plot(ep, [h["train"]["loss_giou"] for h in r["history"]], marker=style[cond], label=r["name"])
        axes[3].plot(ep, [h["val"]["map_50"] for h in r["history"]], marker=style[cond], label=f"{r['name']} mAP@0.5")
        axes[3].plot(ep, [h["val"]["map"] for h in r["history"]], marker=style[cond], ls="--", alpha=0.6,
                     label=f"{r['name']} mAP@[.5:.95]")
    for ax, t in zip(axes, ["total loss", "train loss_ce (last layer)", "train loss_giou (last layer)",
                            "mAP on test subset"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_curves.png"), dpi=120)
    plt.close(fig)

    summary = {}
    lines = ["| 条件 | 初期値 | エポック | test mAP@0.5 | test mAP@[.5:.95] | +NMS(0.7) mAP@0.5 | 重複ペア数 |",
             "|---|---|---|---|---|---|---|"]
    for cond, r in runs.items():
        e = r["eval"]
        if e is None:
            continue
        s = {"weights": e["weights"], "epochs": e["epochs_trained"],
             "map_50": e["metrics_no_nms"]["map_50"], "map": e["metrics_no_nms"]["map"],
             "map_50_with_nms_0.7": e["metrics_with_nms"]["nms_iou_0.7"]["map_50"],
             "duplicate_pairs": e["duplicate_stats"]["duplicate_pairs"],
             "final_train_loss": r["history"][-1]["train"]["loss"], "data_source": e["data_source"]}
        summary[cond] = s
        lines.append(f"| {r['name']} | {s['weights']} | {s['epochs']} | {s['map_50']:.4f} | {s['map']:.4f} | "
                     f"{s['map_50_with_nms_0.7']:.4f} | {s['duplicate_pairs']} |")
    save_json(summary, os.path.join(out_dir, "summary.json"))
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[compare] -> {out_dir}")


def main():
    args = parse_args()
    if args.compare:
        compare(args)
    else:
        evaluate_condition(args)


if __name__ == "__main__":
    main()
