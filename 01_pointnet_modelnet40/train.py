"""PointNet（条件A）/ フラット化MLP（条件B）の学習。

使い方:
  python train.py --quick                                  # 動作確認（CPU・一部データ・1エポック）
  python train.py --model pointnet                         # 条件A
  python train.py --model pointnet --use_tnet              # 条件A + T-Net
  python train.py --model mlp                              # 条件B（ベースライン）
学習後に evaluate.run_evaluation を呼んで metrics.json と画像まで作る（--no_eval で省略）。

注意：検証用の分割を作っていないので、チェックポイントは「最終エポック」を使う
（テスト精度でベストエポックを選ぶとテストデータへのリークになるため）。
エポックごとのテスト精度は学習曲線を描くための監視用で、モデル選択には使わない。
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import augment, load_modelnet40, prepare_points
from evaluate import accuracy, predict, run_evaluation
from model import build_model
from utils import get_device, order_points, set_seed

REG_WEIGHT = 0.001   # T-Net の直交正則化項の重み（PointNet 論文と同じ値）


def condition_name(args):
    name = "mlp_baseline" if args.model == "mlp" else ("pointnet_tnet" if args.use_tnet else "pointnet")
    return ("quick_" + name) if args.quick else name


def plot_curves(history, path, title):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss (cross entropy + reg)")
    axes[0].set_title("loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["test_acc"] for h in history], label="test (monitor only)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("accuracy")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["pointnet", "mlp"], default="pointnet")
    p.add_argument("--use_tnet", action="store_true", help="PointNet に入力・特徴の T-Net を付ける")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="CPUのみ・データの一部・1エポックで動作確認")
    p.add_argument("--dummy", action="store_true", help="ダミーデータを使う（動作確認専用）")
    p.add_argument("--data_root", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--no_eval", action="store_true", help="学習後の評価を省略する")
    args = p.parse_args()
    if args.use_tnet and args.model != "pointnet":
        p.error("--use_tnet は --model pointnet のときだけ使えます")
    if args.quick:
        args.epochs = 1

    set_seed(args.seed)
    device = get_device(force_cpu=args.quick)
    name = condition_name(args)
    cond_dir = os.path.join(args.results_dir, name)
    os.makedirs(os.path.join(cond_dir, "checkpoints"), exist_ok=True)

    data = load_modelnet40(args.data_root, quick=args.quick, dummy=args.dummy, seed=args.seed)
    source = data["source"]
    print(f"[train] 条件={name} device={device} data={source} "
          f"train={len(data['train_y'])} test={len(data['test_y'])} epochs={args.epochs}")

    cfg = {"model": args.model, "use_tnet": args.use_tnet, "num_points": args.num_points,
           "num_classes": len(data["class_names"]), "dropout": args.dropout, "seed": args.seed}
    model = build_model(cfg).to(device)
    print(f"[train] パラメータ数: {sum(q.numel() for q in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_pool, train_y = data["train_x"], data["train_y"]
    # 評価用の固定入力（監視用のテスト精度）
    x_test = prepare_points(data["test_x"], args.num_points, train=False)       # (M, N, 3)
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_y))
        loss_sum, correct, seen = 0.0, 0, 0
        # BatchNorm は batch=1 だと学習できないので、端数バッチは捨てる
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = prepare_points(train_pool[idx], args.num_points, train=True)    # (B, N, 3) 正規化済み
            x = augment(x.to(device))                                           # スケール・移動・ノイズ
            x = order_points(x, args.model)                                     # MLPのみ x 昇順に並べる
            y = train_y[idx].to(device)                                         # (B,)
            logits, reg = model(x)                                              # (B, C), スカラー
            loss = F.cross_entropy(logits, y) + REG_WEIGHT * reg
            opt.zero_grad()
            loss.backward()      # 誤差逆伝播：損失を各パラメータで微分する
            opt.step()           # Adam でパラメータを更新
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()
        test_acc = accuracy(predict(model, x_test, args.model, device)[0], data["test_y"])
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                        "test_acc": test_acc, "elapsed_sec": time.time() - t0})
        print(f"[train] epoch {epoch:3d}/{args.epochs} loss={loss_sum / seen:.4f} "
              f"train_acc={correct / seen:.4f} test_acc={test_acc:.4f} ({time.time() - t0:.0f}s)")
        with open(os.path.join(cond_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "data_source": source, "history": history}, f,
                      ensure_ascii=False, indent=2)

    torch.save({"state_dict": model.state_dict(), "cfg": cfg},
               os.path.join(cond_dir, "checkpoints", "last.pt"))
    plot_curves(history, os.path.join(cond_dir, "training_curves.png"),
                f"{name} (data={source}{', QUICK CHECK' if args.quick else ''})")

    if not args.no_eval:
        run_evaluation(model, cfg, data, device, cond_dir, args.quick, source)


if __name__ == "__main__":
    main()
