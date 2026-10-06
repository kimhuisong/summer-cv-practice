"""評価：通常精度 / 点順シャッフル / 入力点数 / 混同行列 / 3D可視化 / 条件間の比較。

使い方:
  python evaluate.py --condition pointnet            # results/pointnet/ のチェックポイントを評価
  python evaluate.py --compare                       # results/ 下の各条件を比較する図と表を作る
結果は results/<条件名>/metrics.json と画像に保存される。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_modelnet40, prepare_points, subsample_points
from model import build_model
from utils import get_device, order_points, set_seed

POINT_COUNTS = (1024, 512, 256)   # 入力点数を減らす実験（PointNetのみ）
N_TRIALS = 5                      # シャッフル・点の間引きは乱数に依存するので複数回の平均と標準偏差を出す


@torch.no_grad()
def predict(model, x, model_type, device, shuffle=False, seed=0, batch_size=128):
    """x: (M, N, 3) 全テスト点群 -> (preds: (M,), logits: (M, C))。

    shuffle=True なら各点群の点の順序をランダムに入れ替えてから推論する。
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    preds, logits = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size].to(device)                       # (B, N, 3)
        xb = order_points(xb, model_type, shuffle, gen)           # 並びを決める (B, N, 3)
        out, _ = model(xb)                                        # (B, C)
        logits.append(out.cpu())
        preds.append(out.argmax(dim=1).cpu())
    return torch.cat(preds), torch.cat(logits)


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def confusion_matrix(preds, labels, num_classes=40):
    """cm[真のクラス, 予測クラス] = 件数。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels.numpy(), preds.numpy()), 1)
    return cm


def plot_confusion_matrix(cm, names, path, title):
    # 行（真のクラス）ごとに正規化して「そのクラスのうち何割がどこに行ったか」を見る
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_confused(cm, names, k=10):
    """誤分類の多い（真, 予測）ペアの上位 k 個。"""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [{"true": names[t], "pred": names[p], "count": int(off[t, p])}
            for t, p in order if off[t, p] > 0]


def plot_predictions(x, labels, preds, names, path, n_show=8):
    """点群と予測ラベルの3D可視化。正解（緑）と誤分類（赤）を半分ずつ並べる。"""
    correct = torch.nonzero(preds == labels).flatten().tolist()
    wrong = torch.nonzero(preds != labels).flatten().tolist()
    # 正解は先頭から取ると同じクラスばかりになるので、間隔をあけて選ぶ
    pick = correct[::max(1, len(correct) // (n_show // 2))][:n_show // 2] + wrong[:n_show // 2]
    pick += [i for i in correct if i not in pick][:n_show - len(pick)]   # 誤分類が少ないときの補充
    cols = 4
    rows = (len(pick) + cols - 1) // cols
    fig = plt.figure(figsize=(3.4 * cols, 3.4 * rows))
    for k, i in enumerate(pick):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        p = x[i].numpy()                                   # (N, 3)
        ok = preds[i] == labels[i]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1.5, c="tab:green" if ok else "tab:red")
        ax.set_title(f"true: {names[labels[i]]}\npred: {names[preds[i]]}", fontsize=9,
                     color="tab:green" if ok else "tab:red")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_evaluation(model, cfg, data, device, cond_dir, quick, source):
    """1条件ぶんの評価を実行し、metrics.json と画像を cond_dir に保存する。"""
    model_type = cfg["model"]
    names = data["class_names"]
    labels = data["test_y"]
    x_test = prepare_points(data["test_x"], cfg["num_points"], train=False)  # (M, 1024, 3)
    metrics = {"condition": os.path.basename(cond_dir), "model": model_type,
               "use_tnet": cfg["use_tnet"], "data_source": source, "quick": quick,
               "num_test": len(labels),
               "num_params": sum(p.numel() for p in model.parameters())}

    # --- 1) 通常のテスト精度（MLPは固定順、PointNetは保存された順） ---
    preds, logits = predict(model, x_test, model_type, device)
    cm = confusion_matrix(preds, labels, len(names))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    seen = cm.sum(axis=1) > 0
    metrics["test"] = {"overall_acc": accuracy(preds, labels),
                       "mean_class_acc": float(per_class[seen].mean())}
    metrics["per_class_acc"] = {n: float(a) for n, a, s in zip(names, per_class, seen) if s}

    # --- 2) テスト時に点の順序をシャッフルしたときの精度（シードを変えて複数回） ---
    sh = [accuracy(predict(model, x_test, model_type, device, shuffle=True, seed=s)[0], labels)
          for s in range(N_TRIALS)]
    metrics["shuffled"] = {"mean_acc": float(np.mean(sh)), "std_acc": float(np.std(sh)), "trials": sh}

    # --- 3) 順序不変性の数値チェック：同じ点群を並べ替えたときの logits の最大差 ---
    _, logits_sh = predict(model, x_test[:256], model_type, device, shuffle=True, seed=123)
    metrics["max_abs_logit_diff_under_shuffle"] = float((logits[:256] - logits_sh).abs().max())

    # --- 4) 入力点数を減らしたときの精度（PointNetのみ。MLPは入力次元が固定で比較できない） ---
    if model_type == "pointnet":
        pc = {}
        for n in POINT_COUNTS:
            if n == cfg["num_points"]:
                pc[str(n)] = {"mean_acc": metrics["test"]["overall_acc"], "std_acc": 0.0}
                continue
            accs = []
            for s in range(N_TRIALS):
                g = torch.Generator().manual_seed(1000 + s)
                xs = subsample_points(x_test, n, g)                        # (M, n, 3)
                accs.append(accuracy(predict(model, xs, model_type, device)[0], labels))
            pc[str(n)] = {"mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs))}
        metrics["num_points"] = pc

    # --- 5) 混同行列・誤分類ペア・3D可視化 ---
    metrics["top_confused"] = top_confused(cm, names)
    plot_confusion_matrix(cm, names, os.path.join(cond_dir, "confusion_matrix.png"),
                          f"{metrics['condition']} confusion matrix (row-normalized)")
    plot_predictions(x_test, labels, preds, names, os.path.join(cond_dir, "predictions_3d.png"))
    np.save(os.path.join(cond_dir, "confusion_matrix.npy"), cm)

    with open(os.path.join(cond_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] {metrics['condition']}: test={metrics['test']['overall_acc']:.4f} "
          f"shuffled={metrics['shuffled']['mean_acc']:.4f}±{metrics['shuffled']['std_acc']:.4f} "
          f"(data={source}, quick={quick})")
    return metrics


def compare(results_dir, conditions):
    """各条件の metrics.json を集めて、通常 vs シャッフルの棒グラフと点数の折れ線を作る。"""
    ms = []
    for c in conditions:
        path = os.path.join(results_dir, c, "metrics.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                ms.append(json.load(f))
        else:
            print(f"[compare] {path} がないのでスキップ")
    if not ms:
        raise SystemExit("比較できる metrics.json がありません。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    xs = np.arange(len(ms))
    ax.bar(xs - 0.18, [m["test"]["overall_acc"] for m in ms], 0.36, label="test (normal order)")
    ax.bar(xs + 0.18, [m["shuffled"]["mean_acc"] for m in ms], 0.36,
           yerr=[m["shuffled"]["std_acc"] for m in ms], label="test (points shuffled)")
    ax.set_xticks(xs)
    ax.set_xticklabels([m["condition"] for m in ms])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("order invariance")
    ax.legend()
    ax = axes[1]
    for m in ms:
        if "num_points" in m:
            ns = sorted(map(int, m["num_points"]), reverse=True)
            ax.errorbar(ns, [m["num_points"][str(n)]["mean_acc"] for n in ns],
                        yerr=[m["num_points"][str(n)]["std_acc"] for n in ns], marker="o",
                        label=m["condition"])
    ax.set_xlabel("number of input points")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("fewer points (PointNet only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "comparison.png"), dpi=130)
    plt.close(fig)
    summary = [{k: m[k] for k in ("condition", "data_source", "quick", "test", "shuffled", "num_points",
                                  "max_abs_logit_diff_under_shuffle", "num_params") if k in m} for m in ms]
    with open(os.path.join(results_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[compare] {len(ms)} 条件を results/comparison.json と comparison.png に保存")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", help="results/ 下の条件名（pointnet / pointnet_tnet / mlp_baseline）")
    p.add_argument("--compare", action="store_true", help="条件間の比較図・表を作る")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--data_root", default="data")
    p.add_argument("--quick", action="store_true", help="CPUのみ・データの一部で動作確認")
    p.add_argument("--dummy", action="store_true", help="ダミーデータを使う（動作確認専用）")
    args = p.parse_args()

    if args.compare:
        compare(args.results_dir, ["pointnet", "pointnet_tnet", "mlp_baseline"])
        return
    if not args.condition:
        p.error("--condition か --compare を指定してください")

    cond_dir = os.path.join(args.results_dir, args.condition)
    ckpt = torch.load(os.path.join(cond_dir, "checkpoints", "last.pt"), map_location="cpu")
    cfg = ckpt["cfg"]
    set_seed(cfg["seed"])
    device = get_device(force_cpu=args.quick)
    data = load_modelnet40(args.data_root, quick=args.quick, dummy=args.dummy, seed=cfg["seed"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    run_evaluation(model, cfg, data, device, cond_dir, args.quick, data["source"])


if __name__ == "__main__":
    main()
