"""学習済み A（unet）と B（noskip）を test split で評価し、比較図を作る。

使い方:
  python evaluate.py --quick                  # 動作確認（train.py --quick の後に実行）
  python evaluate.py --seeds 0 1 2            # 本番（train.py を各 seed で実行した後）
出力: results/comparison/{metrics.json, per_class_iou.png, prediction_comparison.png}
  prediction_comparison.png は先頭 seed のモデルが、同一のテスト画像に出した予測を A・B 並べたもの。
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CLASS_NAMES, get_datasets, normalize
from model import build_model, count_params
from utils import evaluate_loader, iou_from_confusion, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"unet": "A: U-Net (with skip)", "noskip": "B: no skip"}
# 0=前景:赤, 1=背景:薄灰, 2=境界:黄
CMAP = ListedColormap(["#d62728", "#e8e8e8", "#f2c200"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n_vis", type=int, default=6, help="比較図に並べるテスト画像の枚数")
    p.add_argument("--data_root", default=os.path.join(HERE, "data"))
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dummy", action="store_true")
    return p.parse_args()


def load_model(results_dir, variant, seed, device):
    ckpt = torch.load(os.path.join(results_dir, f"{variant}_seed{seed}", "checkpoints", "best.pth"),
                      map_location=device)
    model = build_model(variant, base=ckpt["args"]["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["epoch"]


def main():
    args = parse_args()
    device = torch.device("cpu" if args.quick else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.quick:
        args.results_dir = os.path.join(args.results_dir, "quick")
    out_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seeds[0])
    _, _, test_ds, is_real = get_datasets(args.data_root, args.seeds[0], args.quick, args.dummy)
    test_loader = DataLoader(test_ds, batch_size=64)
    tag = " [quick/dummy: sanity check only]" if (args.quick or not is_real) else ""

    # --- 指標: seed × 条件ごとに test 全体の混同行列から IoU を出す ---
    per_seed = {v: [] for v in VARIANTS}
    models0 = {}
    for seed in args.seeds:
        for v in VARIANTS:
            model, best_epoch = load_model(args.results_dir, v, seed, device)
            iou, miou = iou_from_confusion(evaluate_loader(model, test_loader, device))
            per_seed[v].append({"seed": seed, "best_epoch": best_epoch, "miou": miou,
                                "iou": dict(zip(CLASS_NAMES, iou)), "n_params": count_params(model)})
            if seed == args.seeds[0]:
                models0[v] = model

    def stat(v, key):  # seed 間の平均と標準偏差（母標準偏差。seed が1つなら 0）
        vals = [r["miou"] if key == "miou" else r["iou"][key] for r in per_seed[v]]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

    keys = ["miou"] + CLASS_NAMES
    summary = {v: {k: stat(v, k) for k in keys} for v in VARIANTS}
    # 仮説の検証用: A に対する B の相対低下率 = (A - B) / A（大きいほど B で落ちている）
    rel_drop = {k: (summary["unet"][k]["mean"] - summary["noskip"][k]["mean"]) / max(summary["unet"][k]["mean"], 1e-12)
                for k in keys}
    out = {"seeds": args.seeds, "n_test": len(test_ds), "quick": args.quick, "real_data": is_real,
           "note": "quick/ダミーの値は動作確認用であり、本番の結果ではない" if tag else "",
           "n_params": {v: per_seed[v][0]["n_params"] for v in VARIANTS},
           "summary": summary, "relative_drop_B_vs_A": rel_drop, "per_seed": per_seed}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for v, name in VARIANTS.items():
        print(name, " ".join(f"{k}={summary[v][k]['mean']:.4f}±{summary[v][k]['std']:.4f}" for k in keys))
    print("relative drop (A-B)/A:", {k: round(x, 4) for k, x in rel_drop.items()})

    # --- 図1: クラスごとの IoU（seed 平均±標準偏差）---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels, w = ["mIoU"] + CLASS_NAMES, 0.38
    for j, v in enumerate(VARIANTS):
        ax.bar(np.arange(4) + (j - 0.5) * w, [summary[v][k]["mean"] for k in keys], w,
               yerr=[summary[v][k]["std"] for k in keys], capsize=3, label=VARIANTS[v])
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set(ylabel="IoU (test)", ylim=(0, 1), title=f"per-class IoU, seeds={args.seeds}{tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_iou.png"), dpi=130)
    plt.close(fig)

    # --- 図2: 同じテスト画像に対する A・B の予測（先頭 seed のモデル）---
    # 画像は seed 固定の乱数で選ぶ（A・B・どの回でも同じ画像になる）
    idx = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(0))[: args.n_vis]
    x = torch.stack([test_ds[int(i)][0] for i in idx])  # (n, 3, H, W) uint8
    y = torch.stack([test_ds[int(i)][1] for i in idx])  # (n, H, W)
    with torch.no_grad():
        preds = {v: m(normalize(x).to(device)).argmax(1).cpu() for v, m in models0.items()}
    cols = [("image", None), ("ground truth", y), (VARIANTS["unet"], preds["unet"]), (VARIANTS["noskip"], preds["noskip"])]
    fig, axes = plt.subplots(len(idx), 4, figsize=(8.5, 2.1 * len(idx)))
    axes = np.atleast_2d(axes)
    for r in range(len(idx)):
        for c, (title, m) in enumerate(cols):
            a = axes[r, c]
            if m is None:
                a.imshow(x[r].permute(1, 2, 0).numpy())  # (3,H,W) -> (H,W,3)
            else:
                a.imshow(m[r].numpy(), cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
            a.axis("off")
            if r == 0:
                a.set_title(title, fontsize=8)
    fig.suptitle("red=foreground, gray=background, yellow=boundary" + f" (seed {args.seeds[0]}){tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prediction_comparison.png"), dpi=130)
    plt.close(fig)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
