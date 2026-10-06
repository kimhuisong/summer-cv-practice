"""Pascal VOC 2007 の読み込み・前処理・バッチ化。

- trainval（5,011枚）で学習し、test（4,952枚）で評価する。
- DETR の損失は「画像サイズで正規化した (cx, cy, w, h)」形式のボックスを受け取るので、
  ここで VOC の (xmin, ymin, xmax, ymax) ピクセル座標から変換する。
- ネットワーク制限などで VOC を取得できない場合のために、図形を描いただけの
  ダミーデータセット（動作確認専用）も用意する。
"""

import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# VOC の20クラス（公式の並び順）。インデックス 0..19 をそのままクラスIDとして使う。
# DETR では「物体なし（no-object）」が追加の21番目（インデックス20）になる。
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ImageNet の平均・標準偏差（ResNet-50 の事前学習時と同じ正規化。DetrImageProcessor も同じ値）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# VOC 2007 の取得元。公式（Oxford の thor サーバ）を優先し、使えなければミラーを試す。
# md5 は torchvision.datasets.VOCDetection に記載された公式アーカイブの値。
VOC2007_ARCHIVES = {
    "trainval": {
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
    },
    "test": {
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "b6e924de25625d8de591ea690078ad9f",
    },
}
VOC2007_URL_BASES = [
    "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/",  # 公式（Oxford VGG）
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/",   # 公式の旧ホスト名
    "https://pjreddie.com/media/files/",                 # ミラー（YOLO の作者が公開しているもの）
]


def voc_root(data_dir):
    """展開後の VOC2007 フォルダのパスを返す。"""
    return os.path.join(data_dir, "VOCdevkit", "VOC2007")


def voc_available(data_dir, image_set):
    """指定した split の画像リストが手元にあるかどうか。"""
    return os.path.isfile(os.path.join(voc_root(data_dir), "ImageSets", "Main", f"{image_set}.txt"))


def download_voc2007(data_dir, image_set):
    """VOC 2007 の tar をダウンロードして展開する（md5 で中身を検証する）。

    公式サーバが落ちていることがあるので、URL を順に試す。
    """
    from torchvision.datasets.utils import download_and_extract_archive

    if voc_available(data_dir, image_set):
        return
    archive = VOC2007_ARCHIVES[image_set]
    errors = []
    for base in VOC2007_URL_BASES:
        url = base + archive["filename"]
        try:
            print(f"[data] downloading {url}")
            download_and_extract_archive(url, data_dir, filename=archive["filename"], md5=archive["md5"],
                                         remove_finished=True)
            if voc_available(data_dir, image_set):
                print(f"[data] VOC2007 {image_set} を取得しました: {url}")
                return
        except Exception as e:  # ネットワークエラー・md5 不一致など
            errors.append(f"{url}: {e}")
            print(f"[data] 失敗: {url}: {e}")
    raise RuntimeError("VOC2007 をどの取得元からもダウンロードできませんでした:\n" + "\n".join(errors))


def parse_voc_xml(xml_path):
    """アノテーション XML を読み、ボックス・ラベル・difficult フラグを返す。

    VOC の座標は 1 始まりのピクセル座標なので、xmin/ymin から 1 を引いて 0 始まりにする。
    戻り値:
        boxes: (M, 4) float32, xyxy（元画像のピクセル座標）
        labels: (M,) int64, 0..19
        difficult: (M,) bool
    """
    root = ET.parse(xml_path).getroot()
    boxes, labels, difficult = [], [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        diff = obj.find("difficult")
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text) - 1
        y1 = float(bb.find("ymin").text) - 1
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        boxes.append([x1, y1, x2, y2])
        labels.append(VOC_CLASSES.index(name))
        difficult.append(diff is not None and int(diff.text) == 1)
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(difficult, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# 前処理（画像とボックスを一緒に変換する）
# ---------------------------------------------------------------------------

def get_resize_size(w, h, min_size, max_size):
    """短辺を min_size に揃え、長辺が max_size を超えないように縮小したサイズ (new_w, new_h) を返す。"""
    scale = min_size / min(w, h)
    if max(w, h) * scale > max_size:
        scale = max_size / max(w, h)
    return int(round(w * scale)), int(round(h * scale))


class DetrTransform:
    """学習用: ランダム左右反転 + ランダムな短辺サイズへのリサイズ。評価用: 固定サイズへのリサイズのみ。

    出力のボックスは「リサイズ後の画像サイズで正規化した (cx, cy, w, h)」（DETR の損失が期待する形式）。
    """

    def __init__(self, train, min_sizes, max_size):
        self.train = train
        self.min_sizes = list(min_sizes)
        self.max_size = max_size

    def __call__(self, img, boxes):
        # img: PIL (W, H), boxes: (M, 4) xyxy ピクセル座標
        w, h = img.size
        boxes = boxes.clone()

        # 左右反転（学習時のみ、確率 0.5）。x 座標を w - x に写し、xmin と xmax を入れ替える。
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # マルチスケール学習: 短辺サイズをランダムに選ぶ（評価時は先頭の1つで固定）
        min_size = random.choice(self.min_sizes) if self.train else self.min_sizes[0]
        new_w, new_h = get_resize_size(w, h, min_size, self.max_size)
        img = TF.resize(img, [new_h, new_w], antialias=True)

        # テンソル化と正規化: PIL -> (3, H', W'), 値は ImageNet の平均・分散で標準化
        pixel_values = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)

        # ボックスを [0, 1] に正規化した cxcywh に変換する。
        # 正規化してしまえばリサイズの倍率は相殺されるので、元画像サイズ (w, h) で割ればよい。
        # boxes: (M, 4) xyxy -> (M, 4) cxcywh in [0, 1]
        x1, y1, x2, y2 = boxes.unbind(-1)
        cxcywh = torch.stack([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], dim=-1)
        return pixel_values, cxcywh.clamp(0, 1)


# ---------------------------------------------------------------------------
# データセット
# ---------------------------------------------------------------------------

class VOCDetectionDataset(Dataset):
    """Pascal VOC 2007 の物体検出データセット。

    1サンプルは (pixel_values, target) で、target は
        class_labels: (M,) int64        … DETR の損失に渡す
        boxes:        (M, 4) float32    … 正規化 cxcywh（DETR の損失に渡す）
        orig_boxes:   (M', 4) float32   … 元画像ピクセル座標 xyxy（mAP 評価用）
        orig_labels:  (M',) int64
        orig_difficult: (M',) bool      … mAP では difficult を「無視する GT」として扱う
        orig_size:    (2,) = (H, W)
        image_id:     str
    学習時は difficult の物体を教師から除く（VOC の慣習）。評価用の orig_* には difficult も残す。
    """

    def __init__(self, data_dir, image_set, transform, max_images=None, use_difficult_in_train=False):
        self.root = voc_root(data_dir)
        self.transform = transform
        self.use_difficult = use_difficult_in_train
        with open(os.path.join(self.root, "ImageSets", "Main", f"{image_set}.txt")) as f:
            ids = [line.strip() for line in f if line.strip()]
        if max_images is not None:
            ids = ids[:max_images]
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = Image.open(os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")).convert("RGB")
        boxes, labels, difficult = parse_voc_xml(os.path.join(self.root, "Annotations", f"{image_id}.xml"))
        w, h = img.size

        # 学習に使うボックス（difficult を除く）
        keep = torch.ones_like(difficult) if self.use_difficult else ~difficult
        pixel_values, cxcywh = self.transform(img, boxes[keep])
        target = {
            "class_labels": labels[keep],
            "boxes": cxcywh,
            "orig_boxes": boxes,
            "orig_labels": labels,
            "orig_difficult": difficult,
            "orig_size": torch.tensor([h, w]),
            "image_id": image_id,
        }
        return pixel_values, target


class DummyDetectionDataset(Dataset):
    """動作確認専用のダミーデータ（VOC を取得できない環境向け）。

    灰色の背景に 1〜3 個の色付き矩形を描き、矩形ごとにランダムな VOC クラスを割り当てる。
    **ここで得られる数値には何の意味もない**（パイプラインが最後まで動くかの確認用）。
    """

    def __init__(self, num_images, transform, seed=0):
        self.num_images = num_images
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)
        w, h = int(rng.randint(300, 500)), int(rng.randint(300, 500))
        img = Image.new("RGB", (w, h), (127, 127, 127))
        draw = ImageDraw.Draw(img)
        boxes, labels = [], []
        for _ in range(rng.randint(1, 4)):
            bw, bh = rng.randint(40, w // 2), rng.randint(40, h // 2)
            x1, y1 = rng.randint(0, w - bw), rng.randint(0, h - bh)
            cls = int(rng.randint(0, len(VOC_CLASSES)))
            color = tuple(int(c) for c in rng.randint(0, 255, size=3))
            draw.rectangle([x1, y1, x1 + bw, y1 + bh], fill=color)
            boxes.append([x1, y1, x1 + bw, y1 + bh])
            labels.append(cls)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        pixel_values, cxcywh = self.transform(img, boxes)
        target = {
            "class_labels": labels,
            "boxes": cxcywh,
            "orig_boxes": boxes,
            "orig_labels": labels,
            "orig_difficult": torch.zeros_like(labels, dtype=torch.bool),
            "orig_size": torch.tensor([h, w]),
            "image_id": f"dummy_{idx:05d}",
        }
        return pixel_values, target


def collate_fn(batch):
    """サイズの違う画像を右下にゼロ詰め（パディング）して1つのバッチにする。

    pixel_values: B 個の (3, H_i, W_i) -> (B, 3, H_max, W_max)
    pixel_mask:   (B, H_max, W_max)。本物の画素 = 1、パディング = 0。
                  DETR はこのマスクを使って、パディング部分を Attention で見ないようにする。
    """
    images, targets = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    pixel_values = torch.zeros(len(images), 3, max_h, max_w)
    pixel_mask = torch.zeros(len(images), max_h, max_w, dtype=torch.long)
    for i, img in enumerate(images):
        _, h, w = img.shape
        pixel_values[i, :, :h, :w] = img
        pixel_mask[i, :h, :w] = 1
    return pixel_values, pixel_mask, list(targets)


def build_datasets(args):
    """train / val（= test の一部、毎エポックの監視用）/ test のデータセットを作る。

    戻り値: (train_ds, val_ds, test_ds, data_source の説明文字列)
    """
    train_tf = DetrTransform(train=True, min_sizes=args.train_min_sizes, max_size=args.max_size)
    eval_tf = DetrTransform(train=False, min_sizes=[args.eval_min_size], max_size=args.max_size)

    use_dummy = args.dummy
    if not use_dummy:
        if args.quick and not (voc_available(args.data_dir, "trainval") and voc_available(args.data_dir, "test")):
            # --quick では数GBのダウンロードを待たない。手元に無ければダミーで動作確認する。
            print("[data] --quick: VOC2007 が手元に無いのでダミーデータで動作確認します")
            use_dummy = True
        else:
            download_voc2007(args.data_dir, "trainval")
            download_voc2007(args.data_dir, "test")

    if use_dummy:
        n_train = args.max_train_images or 16
        n_test = args.max_test_images or 8
        train_ds = DummyDetectionDataset(n_train, train_tf, seed=0)
        test_ds = DummyDetectionDataset(n_test, eval_tf, seed=10_000)
        val_ds = DummyDetectionDataset(min(n_test, args.val_images), eval_tf, seed=10_000)
        return train_ds, val_ds, test_ds, "dummy (動作確認用の合成データ。数値に意味はない)"

    train_ds = VOCDetectionDataset(args.data_dir, "trainval", train_tf, max_images=args.max_train_images)
    test_ds = VOCDetectionDataset(args.data_dir, "test", eval_tf, max_images=args.max_test_images)
    # 毎エポックの監視には test の先頭 val_images 枚だけを使う（時間節約のため）。
    # モデル選択（ベストエポックの選択など）には使わず、最終エポックのモデルを評価する。
    val_ds = VOCDetectionDataset(args.data_dir, "test", eval_tf, max_images=args.val_images)
    return train_ds, val_ds, test_ds, "Pascal VOC 2007"
