"""DETR モデルの構築・損失計算・後処理。

モデル本体は Hugging Face transformers の DetrForObjectDetection をそのまま使う。
比較する2条件は「アーキテクチャは完全に同じで、初期値だけが違う」ようにしている。

  A. COCO 事前学習済み DETR 全体（facebook/detr-resnet-50）から開始し、
     分類ヘッドだけを VOC の 20+1 クラス用に作り直してファインチューニング
  B. バックボーン（ResNet-50）だけ ImageNet 事前学習（torchvision の IMAGENET1K_V1、元論文の DETR と同じ重み）、
     入力射影・Transformer エンコーダ/デコーダ・物体クエリ・予測ヘッドはランダム初期化

DETR の全体の流れ（形状は batch=B, 画像 H×W, d_model=256, クエリ数 Q=100 の場合）:
  画像 (B, 3, H, W)
   -> ResNet-50                  -> 特徴マップ (B, 2048, H/32, W/32)
   -> 1x1 conv（input_projection）-> (B, 256, h, w) -> 平坦化 (B, h*w, 256)
   -> Transformer エンコーダ（6層の self-attention、位置埋め込みつき） -> (B, h*w, 256)
   -> Transformer デコーダ（6層。Q=100 個の物体クエリが self-attention で互いを見て、
                           cross-attention で画像特徴を見る）           -> (B, 100, 256)
   -> クラス分類ヘッド（Linear）  -> logits (B, 100, 21)   … 20クラス + 「物体なし」
   -> ボックス回帰ヘッド（3層MLP）-> boxes  (B, 100, 4)    … 正規化 cxcywh（sigmoid で [0,1]）
"""

import torch
from transformers import DetrConfig, DetrForObjectDetection
from transformers.loss.loss_for_object_detection import ForObjectDetectionLoss

from dataset import VOC_CLASSES

HUB_ID = "facebook/detr-resnet-50"

# 条件名（results/<条件名>/ のフォルダ名になる）
CONDITIONS = {
    "A": "A_coco_pretrained",
    "B": "B_imagenet_backbone",
}


def _label_maps():
    id2label = {i: c for i, c in enumerate(VOC_CLASSES)}
    label2id = {c: i for i, c in enumerate(VOC_CLASSES)}
    return id2label, label2id


def _common_config_kwargs():
    """A と B で共通にする設定。

    auxiliary_loss=True: デコーダの全6層の出力それぞれにハンガリアンマッチング＋損失をかける（元論文と同じ）。
    HF の facebook/detr-resnet-50 の config では False になっているので、両条件で明示的に True にそろえる。
    """
    id2label, label2id = _label_maps()
    # num_labels は id2label の長さ（20）から決まる
    return dict(id2label=id2label, label2id=label2id, auxiliary_loss=True)


def set_backbone_trainable(model):
    """バックボーンの学習対象を元論文と同じにそろえる。

    - BatchNorm は DetrFrozenBatchNorm2d（統計量もアフィン係数も固定）にライブラリ側で置き換え済み。
    - stem（conv1）と layer1 は凍結し、layer2〜layer4 だけを学習する。
    ライブラリのバージョンによって凍結処理の挙動が違ったため、ここで明示的に設定する。
    """
    for name, p in model.model.backbone.model.named_parameters():
        p.requires_grad_(any(k in name for k in ("layer2", "layer3", "layer4")))


def load_imagenet_resnet50_into_backbone(model):
    """torchvision の ImageNet 事前学習済み ResNet-50（IMAGENET1K_V1）の重みをバックボーンに読み込む。

    DETR のバックボーンは timm の resnet50（features_only）で、パラメータ名が torchvision の ResNet と同じ
    （conv1, bn1, layer1.0.conv1, ...）なので、分類用の fc 層以外をそのまま読み込める。
    """
    import torchvision

    resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    state = {k: v for k, v in resnet.state_dict().items() if not k.startswith("fc.")}
    result = model.model.backbone.model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"ImageNet 重みの読み込みでキーが一致しません: {result}")


def build_model(condition, allow_random_fallback=False, attn_implementation=None):
    """条件 A / B のモデルを作る。

    allow_random_fallback=True（--quick 用）のときは、ネットワーク制限などで事前学習済み重みを
    取得できなければランダム初期化で続行する。本番では False にして、取得失敗をエラーにする
    （「事前学習済みのつもりがランダム初期化だった」という事故を防ぐため）。

    戻り値: (model, info)。info["weights"] に実際に使った初期値の説明が入る。
    """
    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    info = {"condition": condition, "name": CONDITIONS[condition]}

    if condition == "A":
        try:
            model, loading_info = DetrForObjectDetection.from_pretrained(
                HUB_ID,
                ignore_mismatched_sizes=True,  # 分類ヘッド 92 -> 21 クラスはサイズが違うので作り直す
                output_loading_info=True,
                **_common_config_kwargs(),
                **extra,
            )
            # 分類ヘッド以外が1つでも読み込めていなければ、条件 A として成立しないのでエラーにする
            missing = [k for k in loading_info.get("missing_keys", []) if "class_labels_classifier" not in k]
            mismatched = [str(k) for k in loading_info.get("mismatched_keys", [])
                          if "class_labels_classifier" not in str(k)]
            if missing or mismatched:
                raise RuntimeError(f"COCO 事前学習済み重みの読み込みが不完全です: missing={missing}, "
                                   f"mismatched={mismatched}")
            info["weights"] = f"{HUB_ID}（COCO 事前学習済み）。分類ヘッドのみ新規（21クラス）"
        except Exception as e:
            if not allow_random_fallback:
                raise
            print(f"[model] 事前学習済み重みを取得できないためランダム初期化で続行します（動作確認用）: {e}")
            model = DetrForObjectDetection(DetrConfig(**_common_config_kwargs(), **extra))
            info["weights"] = "random init（フォールバック: 事前学習済み重みを取得できなかった。動作確認専用）"

    elif condition == "B":
        # アーキテクチャは A と同じ config から作り、重みは読み込まない（= Transformer 部分はランダム初期化）
        try:
            config = DetrConfig.from_pretrained(HUB_ID, **_common_config_kwargs(), **extra)
        except Exception as e:
            if not allow_random_fallback:
                raise
            print(f"[model] config を取得できないため DetrConfig の既定値（detr-resnet-50 と同じ構成）を使います: {e}")
            config = DetrConfig(**_common_config_kwargs(), **extra)
        model = DetrForObjectDetection(config)
        try:
            load_imagenet_resnet50_into_backbone(model)
            info["weights"] = ("backbone: torchvision ResNet-50 IMAGENET1K_V1 / "
                               "input_projection・Transformer・クエリ・ヘッド: ランダム初期化")
        except Exception as e:
            if not allow_random_fallback:
                raise
            print(f"[model] ImageNet 重みを取得できないためバックボーンもランダム初期化で続行します（動作確認用）: {e}")
            info["weights"] = "random init（フォールバック: ImageNet 重みを取得できなかった。動作確認専用）"
    else:
        raise ValueError(f"unknown condition: {condition}")

    set_backbone_trainable(model)
    return model, info


def build_model_from_checkpoint(ckpt, attn_implementation=None):
    """train.py が保存したチェックポイント（config と state_dict）からモデルを復元する。"""
    config = DetrConfig.from_dict(ckpt["config"])
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    model = DetrForObjectDetection(config)
    model.load_state_dict(ckpt["model"])
    return model


def detr_loss(model, pixel_values, pixel_mask, labels, use_amp=False):
    """順伝播して DETR の損失を計算する。

    model(pixel_values, labels=...) を呼べばライブラリが損失まで計算してくれるが、ここでは
      1. バックボーン + Transformer（重い部分）は混合精度（fp16）で計算し、
      2. 予測ヘッドとハンガリアンマッチング・損失は fp32 で計算する
    ように分けている（fp16 だとマッチングのコストや GIoU が不安定になりうるため）。
    中身はライブラリの DetrForObjectDetection.forward と同じ計算（hungarian_check.py で一致を確認）。

    labels: 長さ B のリスト。各要素は {"class_labels": (M,), "boxes": (M, 4) 正規化 cxcywh}
    戻り値: (loss スカラー, loss_dict, logits (B, Q, C+1), pred_boxes (B, Q, 4))
    """
    device_type = pixel_values.device.type
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        # out.last_hidden_state: (B, Q, 256) … デコーダ最終層の出力（クエリごとの特徴）
        # out.intermediate_hidden_states: (L=6, B, Q, 256) … auxiliary_loss=True のときの各層の出力
        out = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    hs = out.last_hidden_state.float()
    logits = model.class_labels_classifier(hs)        # (B, Q, 256) -> (B, Q, 21)
    pred_boxes = model.bbox_predictor(hs).sigmoid()    # (B, Q, 256) -> (B, Q, 4)、[0,1] の cxcywh

    outputs_class, outputs_coord = None, None
    if model.config.auxiliary_loss:
        inter = out.intermediate_hidden_states.float()   # (6, B, Q, 256)
        outputs_class = model.class_labels_classifier(inter)       # (6, B, Q, 21)
        outputs_coord = model.bbox_predictor(inter).sigmoid()      # (6, B, Q, 4)

    # ライブラリの損失関数: ハンガリアンマッチング -> 分類 CE + L1 + GIoU（＋各中間層の補助損失）の重み付き和
    loss, loss_dict, _ = ForObjectDetectionLoss(
        logits, labels, logits.device, pred_boxes, model.config, outputs_class, outputs_coord
    )
    return loss, loss_dict, logits, pred_boxes


def box_cxcywh_to_xyxy(b):
    """(…, 4) cxcywh -> (…, 4) xyxy"""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


@torch.no_grad()
def postprocess(logits, pred_boxes, orig_sizes):
    """モデル出力を「元画像ピクセル座標の検出結果」に変換する。NMS は使わない。

    logits: (B, Q, 21), pred_boxes: (B, Q, 4) 正規化 cxcywh, orig_sizes: (B, 2) = (H, W)
    各クエリについて「物体なし」を除いた20クラスの中で最大の確率をスコア、そのクラスをラベルとする。
    戻り値: 長さ B のリスト。各要素は {"boxes": (Q, 4) xyxy, "scores": (Q,), "labels": (Q,)}
    """
    probs = logits.softmax(-1)[..., :-1]          # (B, Q, 21) -> (B, Q, 20)。最後の列が「物体なし」
    scores, labels = probs.max(-1)                # (B, Q), (B, Q)
    boxes = box_cxcywh_to_xyxy(pred_boxes)        # (B, Q, 4) 正規化 xyxy
    h, w = orig_sizes[:, 0], orig_sizes[:, 1]
    scale = torch.stack([w, h, w, h], dim=-1).to(boxes)   # (B, 4)
    boxes = boxes * scale[:, None, :]             # (B, Q, 4) ピクセル座標
    return [{"boxes": b, "scores": s, "labels": l} for b, s, l in zip(boxes, scores, labels)]


def resnet_feature_size(h, w, num_stride2=5):
    """ResNet-50 の出力特徴マップのサイズ。stride 2 の層（conv1, maxpool, layer2〜4）を5回通るので
    各回で ceil(x/2) になる（padding があるため切り上げ）。cross-attention マップを (h, w) に戻すのに使う。"""
    for _ in range(num_stride2):
        h, w = (h + 1) // 2, (w + 1) // 2
    return h, w
