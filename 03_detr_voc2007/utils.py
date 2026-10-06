"""train.py と evaluate.py で共通に使う処理（シード固定・推論・mAP 計算・JSON 保存）。"""

import json
import os
import random

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from model import detr_loss, postprocess

# 損失の内訳のうち、ログに残すもの（最終層の値。"_0"〜"_4" が付いたものは中間層の補助損失）
LOSS_KEYS = ["loss_ce", "loss_bbox", "loss_giou", "cardinality_error"]


def seed_everything(seed):
    """random / numpy / torch / CUDA の乱数シードを固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN の自動アルゴリズム選択を切って再現性を優先する（完全な決定性までは保証されない）
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader の各ワーカーの乱数（データ拡張用）をシードから決める。"""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def targets_to_device(targets, device):
    """DETR の損失に渡す部分（class_labels, boxes）だけを device に送る。"""
    return [{"class_labels": t["class_labels"].to(device), "boxes": t["boxes"].to(device)} for t in targets]


def to_metric_target(t):
    """torchmetrics 用の正解。difficult の物体は iscrowd=1 にして「当てても外しても評価に影響しない」扱いにする
    （VOC の評価で difficult を無視する慣習に近づけるため）。"""
    return {"boxes": t["orig_boxes"], "labels": t["orig_labels"], "iscrowd": t["orig_difficult"].long()}


@torch.no_grad()
def run_inference(model, loader, device, use_amp=False, compute_loss=True):
    """データ全体に推論をかけ、検出結果（NMS なし・全100クエリ）と平均損失を返す。

    戻り値: (preds, targets, mean_losses)
      preds:   画像ごとの {"boxes": (Q,4) xyxy ピクセル, "scores": (Q,), "labels": (Q,)}（CPU）
      targets: 画像ごとの torchmetrics 用の正解（CPU）
    """
    model.eval()
    preds, metric_targets = [], []
    sums, n_batches = {}, 0
    for pixel_values, pixel_mask, targets in loader:
        pixel_values, pixel_mask = pixel_values.to(device), pixel_mask.to(device)
        labels = targets_to_device(targets, device)
        loss, loss_dict, logits, pred_boxes = detr_loss(model, pixel_values, pixel_mask, labels, use_amp)
        if compute_loss:
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            for k in LOSS_KEYS:
                sums[k] = sums.get(k, 0.0) + float(loss_dict[k].detach())
            n_batches += 1
        orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        for p in postprocess(logits, pred_boxes, orig_sizes):
            preds.append({k: v.cpu() for k, v in p.items()})
        metric_targets.extend(to_metric_target(t) for t in targets)
    mean_losses = {k: v / max(n_batches, 1) for k, v in sums.items()}
    return preds, metric_targets, mean_losses


def compute_map(preds, targets, per_class=False):
    """torchmetrics で mAP を計算する。

    - map:    mAP@[0.5:0.95]（IoU 閾値 0.50, 0.55, ..., 0.95 の平均。COCO の主指標）
    - map_50: mAP@0.5（VOC の主指標に相当。ただし補間方法は COCO 式の101点で、VOC2007 公式の11点補間とは異なる）
    per_class=True のときは、クラスごとの AP@0.5 も返す。
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    r = metric.compute()
    out = {"map": float(r["map"]), "map_50": float(r["map_50"]), "map_75": float(r["map_75"]),
           "map_small": float(r["map_small"]), "map_medium": float(r["map_medium"]),
           "map_large": float(r["map_large"])}
    if per_class:
        m50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
        m50.update(preds, targets)
        r50 = m50.compute()
        out["ap50_per_class_raw"] = {int(c): float(v) for c, v in zip(r50["classes"], r50["map_per_class"])}
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
