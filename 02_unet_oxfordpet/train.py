"""U-Net（スキップあり/なし）の学習。

使い方:
  python train.py --quick                         # 動作確認（CPU・一部データ・1エポック）
  python train.py --variant unet   --seed 0       # 条件A（本番、GPU推奨）
  python train.py --variant noskip --seed 0       # 条件B（本番、GPU推奨）
結果: results/<variant>_seed<seed>/{history.json, train_info.json, learning_curves.png, checkpoints/best.pth}
"""
import argparse
import json
import os
import time

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CLASS_NAMES, get_datasets, normalize
from model import build_model, count_params
from utils import evaluate_loader, iou_from_confusion, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["unet", "noskip"], default="unet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32, help="最初の層のチャネル数")
    p.add_argument("--data_root", default=os.path.join(HERE, "data"))
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--quick", action="store_true", help="CPU・一部データ・1エポックの動作確認")
    p.add_argument("--dummy", action="store_true", help="ダミーデータを使う（動作確認専用）")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.quick:
        args.epochs, args.batch_size = 1, 16
        args.results_dir = os.path.join(args.results_dir, "quick")  # 本番結果と混ぜない
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # T4 では混合精度で高速化

    train_ds, val_ds, _, is_real = get_datasets(args.data_root, args.seed, args.quick, args.dummy)
    # DataLoader の乱数も seed から作る（再現性のため）
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = build_model(args.variant, base=args.base).to(device)
    n_params = count_params(model)
    print(f"variant={args.variant} params={n_params:,} device={device} real_data={is_real}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()  # 全ピクセル・全クラスを同じ重みで扱う（A/Bで共通）

    out_dir = os.path.join(args.results_dir, f"{args.variant}_seed{args.seed}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    history, best_miou, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = normalize(x).to(device), y.to(device).long()
            # 左右反転の拡張（画像とマスクに同じ反転を適用）。サンプルごとにランダム
            flip = torch.rand(x.size(0), device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            y = torch.where(flip[:, None, None], y.flip(-1), y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)  # logits (B,3,H,W) と y (B,H,W)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        iou, miou = iou_from_confusion(evaluate_loader(model, val_loader, device))
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "val_miou": miou,
               "val_iou": dict(zip(CLASS_NAMES, iou)), "elapsed_sec": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch:3d} loss={rec['train_loss']:.4f} val_mIoU={miou:.4f} "
              f"boundary_IoU={iou[2]:.4f} ({rec['elapsed_sec']:.0f}s)")
        if miou > best_miou:  # val mIoU が最良のエポックを保存（test は選択に使わない）
            best_miou = miou
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                       os.path.join(ckpt_dir, "best.pth"))

    info = {"variant": args.variant, "seed": args.seed, "n_params": n_params,
            "epochs": args.epochs, "best_val_miou": best_miou, "train_sec": time.time() - t0,
            "quick": args.quick, "real_data": is_real,
            "note": "quick/ダミーの値は動作確認用であり、本番の結果ではない" if (args.quick or not is_real) else ""}
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 学習曲線: 左=学習損失、右=検証 IoU（mIoU と境界クラス）
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history])
    ax[0].set(xlabel="epoch", ylabel="train loss", title="train loss")
    ax[1].plot(ep, [h["val_miou"] for h in history], label="val mIoU")
    ax[1].plot(ep, [h["val_iou"]["boundary"] for h in history], label="val IoU (boundary)")
    ax[1].set(xlabel="epoch", ylabel="IoU", title="validation")
    ax[1].legend()
    fig.suptitle(f"{args.variant} seed{args.seed}" + (" [quick/dummy: sanity check only]" if (args.quick or not is_real) else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=120)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
