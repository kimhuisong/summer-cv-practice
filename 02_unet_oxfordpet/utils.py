"""シード固定・混同行列・IoU など、train.py / evaluate.py の共通処理。"""
import random

import numpy as np
import torch

from data import N_CLASSES, normalize


def set_seed(seed):
    """random / numpy / torch / CUDA の乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_matrix(pred, target, n_classes=N_CLASSES):
    """pred, target: (B, H, W) の整数ラベル -> (n_classes, n_classes)。行=正解、列=予測。"""
    idx = target.reshape(-1).long() * n_classes + pred.reshape(-1).long()
    return torch.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_from_confusion(cm):
    """クラスごとの IoU = TP / (TP + FP + FN) と mIoU（クラス平均）。

    cm[i, j] = 正解が i で予測が j のピクセル数。
      TP = cm[i, i]、FN = 行和 - TP、FP = 列和 - TP
    """
    cm = cm.double()
    tp = cm.diag()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = tp / union.clamp(min=1)
    return iou.tolist(), iou.mean().item()


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """loader 全体の混同行列を累積して返す（画像ごとではなくデータセット全体で IoU を出す）。"""
    model.eval()
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
    for x, y in loader:
        x = normalize(x).to(device)
        pred = model(x).argmax(1).cpu()  # (B, 3, H, W) -> (B, H, W)
        cm += confusion_matrix(pred, y)
    return cm
