"""DETR を Pascal VOC 2007 trainval で学習する。

使い方:
  python train.py --condition A            # COCO 事前学習済み DETR をファインチューニング
  python train.py --condition B            # ImageNet バックボーンのみ事前学習、Transformer はランダム初期化
  python train.py --condition A --quick    # CPU・データの一部・1エポックで動作確認

A と B は「同じエポック数・同じハイパーパラメータ・同じデータ拡張」で学習する（初期値だけが違う）。
結果は results/<条件名>/ に保存する:
  history.json     … エポックごとの学習損失・監視用損失・監視用 mAP
  loss_curve.png   … 学習曲線
  train_config.json… 実行時の設定と、実際に使った初期値
  checkpoints/last.pt（git 管理しない）
"""

import argparse
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import build_datasets, collate_fn
from model import CONDITIONS, build_model, detr_loss
from utils import LOSS_KEYS, compute_map, run_inference, save_json, seed_everything, seed_worker, targets_to_device


def parse_args():
    p = argparse.ArgumentParser(description="DETR fine-tuning on Pascal VOC 2007")
    p.add_argument("--condition", choices=["A", "B"], required=True,
                   help="A: COCO 事前学習済み DETR 全体 / B: ImageNet バックボーンのみ")
    p.add_argument("--quick", action="store_true", help="CPU・データの一部・1エポックの動作確認")
    p.add_argument("--dummy", action="store_true", help="VOC の代わりに合成ダミーデータを使う（動作確認用）")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="Transformer・ヘッドの学習率（元論文と同じ）")
    p.add_argument("--lr_backbone", type=float, default=1e-5, help="バックボーンの学習率（元論文と同じ）")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_max_norm", type=float, default=0.1, help="勾配クリッピング（元論文と同じ）")
    p.add_argument("--lr_drop", type=int, default=None,
                   help="このエポック数を終えたら学習率を1/10にする。省略時は epochs の8割（5エポックなら4）")
    p.add_argument("--train_min_sizes", type=int, nargs="+", default=[384, 416, 448, 480, 512, 544, 576],
                   help="学習時の短辺サイズの候補（マルチスケール学習）")
    p.add_argument("--eval_min_size", type=int, default=512)
    p.add_argument("--max_size", type=int, default=800, help="長辺の上限")
    p.add_argument("--val_images", type=int, default=500,
                   help="毎エポックの監視に使う test 画像の枚数（モデル選択には使わない）")
    p.add_argument("--max_train_images", type=int, default=None)
    p.add_argument("--max_test_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true", help="混合精度（fp16）を使わない")
    args = p.parse_args()

    if args.quick:
        # 動作確認モード: CPU・少数の画像・小さい解像度・1エポック
        args.epochs = 1
        args.batch_size = 2
        args.max_train_images = args.max_train_images or 16
        args.max_test_images = args.max_test_images or 8
        args.val_images = min(args.val_images, args.max_test_images)
        args.train_min_sizes = [256]
        args.eval_min_size = 256
        args.max_size = 384
        args.num_workers = 0
    if args.lr_drop is None:
        args.lr_drop = max(1, math.floor(args.epochs * 0.8))
    return args


def plot_history(history, path, title):
    """エポックごとの損失（学習・監視用）と監視用 mAP をプロットする。"""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(epochs, [h["train"]["loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val"]["loss"] for h in history], "s--", label="val (test subset)")
    axes[0].set_title("total loss (with aux losses)")
    for ax, key in zip(axes[1:3], ["loss_ce", "loss_giou"]):
        ax.plot(epochs, [h["train"][key] for h in history], "o-", label="train")
        ax.plot(epochs, [h["val"][key] for h in history], "s--", label="val (test subset)")
        ax.set_title(f"{key} (last decoder layer)")
    axes[3].plot(epochs, [h["val"]["map_50"] for h in history], "o-", label="mAP@0.5")
    axes[3].plot(epochs, [h["val"]["map"] for h in history], "s-", label="mAP@[.5:.95]")
    axes[3].set_title("mAP on test subset")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cpu" if args.quick or not torch.cuda.is_available() else "cuda")
    use_amp = device.type == "cuda" and not args.no_amp

    name = CONDITIONS[args.condition] + ("_quick" if args.quick else "")
    out_dir = os.path.join(args.results_dir, name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[train] condition={args.condition} ({name}) device={device} amp={use_amp}")

    # ---- データ ----
    train_ds, val_ds, _, data_source = build_datasets(args)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g,
                              pin_memory=device.type == "cuda", drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=args.num_workers)
    print(f"[train] data={data_source} train={len(train_ds)} val(test subset)={len(val_ds)}")

    # ---- モデル ----
    model, info = build_model(args.condition, allow_random_fallback=args.quick)
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] weights: {info['weights']}")
    print(f"[train] params: total={n_total / 1e6:.1f}M trainable={n_train / 1e6:.1f}M")

    # ---- 最適化（元論文の設定: AdamW、バックボーンだけ学習率を1/10、勾配クリッピング 0.1） ----
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr}, {"params": backbone_params, "lr": args.lr_backbone}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_json({"args": vars(args), "model_info": info, "data_source": data_source,
               "num_train_images": len(train_ds), "num_val_images": len(val_ds),
               "params_total": n_total, "params_trainable": n_train,
               "transformers_version": __import__("transformers").__version__,
               "torch_version": torch.__version__},
              os.path.join(out_dir, "train_config.json"))

    history = []
    for epoch in range(1, args.epochs + 1):
        # ---------------- 学習 ----------------
        model.train()
        t0 = time.time()
        sums, n = {}, 0
        for it, (pixel_values, pixel_mask, targets) in enumerate(train_loader):
            # pixel_values: (B, 3, H, W), pixel_mask: (B, H, W)
            pixel_values, pixel_mask = pixel_values.to(device), pixel_mask.to(device)
            labels = targets_to_device(targets, device)

            loss, loss_dict, _, _ = detr_loss(model, pixel_values, pixel_mask, labels, use_amp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss が有限値ではありません: {loss_dict}")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # クリッピングの前に勾配を元のスケールに戻す
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            scaler.step(optimizer)
            scaler.update()

            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            for k in LOSS_KEYS:
                sums[k] = sums.get(k, 0.0) + float(loss_dict[k].detach())
            n += 1
            if it % 50 == 0 or it == len(train_loader) - 1:
                elapsed = time.time() - t0
                eta = elapsed / (it + 1) * (len(train_loader) - it - 1)
                print(f"  epoch {epoch} iter {it + 1}/{len(train_loader)} loss={loss.item():.3f} "
                      f"ce={float(loss_dict['loss_ce'].detach()):.3f} giou={float(loss_dict['loss_giou'].detach()):.3f} "
                      f"elapsed={elapsed / 60:.1f}min eta={eta / 60:.1f}min", flush=True)
        scheduler.step()
        train_time = time.time() - t0
        train_stats = {k: v / n for k, v in sums.items()}

        # ---------------- 監視（test の一部での損失と mAP） ----------------
        t1 = time.time()
        preds, metric_targets, val_losses = run_inference(model, val_loader, device, use_amp)
        val_stats = {**val_losses, **compute_map(preds, metric_targets)}
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats,
                        "train_time_sec": train_time, "val_time_sec": time.time() - t1,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"[epoch {epoch}] train_loss={train_stats['loss']:.3f} val_loss={val_stats['loss']:.3f} "
              f"val_mAP50={val_stats['map_50']:.4f} val_mAP={val_stats['map']:.4f} "
              f"time={train_time / 60:.1f}min", flush=True)

        # 途中で Colab が切れても結果が残るように、毎エポック保存する
        save_json(history, os.path.join(out_dir, "history.json"))
        plot_history(history, os.path.join(out_dir, "loss_curve.png"), name)

    torch.save({"model": model.state_dict(), "config": model.config.to_dict(), "condition": args.condition,
                "model_info": info, "args": vars(args)}, os.path.join(ckpt_dir, "last.pt"))
    print(f"[train] done. results -> {out_dir}")


if __name__ == "__main__":
    main()
