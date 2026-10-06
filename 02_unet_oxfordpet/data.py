"""Oxford-IIIT Pet のセグメンテーションデータ（128x128、3クラス）。

クラス定義（torchvision の trimap 1/2/3 を 0 始まりに振り替える）:
  0 = 前景（ペット）, 1 = 背景, 2 = 境界
前処理（リサイズ）は最初の1回だけ行い、uint8 のテンソルとして data/ にキャッシュする。
ダウンロード元が使えない環境向けに、動作確認用のダミーデータ（楕円＋境界リング）も用意する。
"""
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

IMG_SIZE = 128
CLASS_NAMES = ["foreground", "background", "boundary"]
N_CLASSES = 3


def _load_split_from_torchvision(root, split, download):
    """torchvision から読み、128x128 にリサイズした (images, masks) を返す。

    images: uint8 (N, 3, 128, 128)、masks: uint8 (N, 128, 128)（値は 0/1/2）。
    """
    from PIL import Image
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root=root, split=split, target_types="segmentation", download=download)
    imgs, masks = [], []
    for i in range(len(ds)):
        img, mask = ds[i]  # PIL (RGB) と PIL (trimap: 1=前景, 2=背景, 3=境界)
        # 画像は bilinear、マスクはクラス値が混ざらないよう nearest でリサイズ
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        imgs.append(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))  # (H,W,3) -> (3,H,W)
        masks.append(np.asarray(mask, dtype=np.uint8) - 1)               # 1,2,3 -> 0,1,2
    return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(masks))


def load_oxford_pet(root, split, download=True):
    """キャッシュ（data/pet_cache/）があればそれを使い、無ければ作る。split: trainval / test"""
    cache = os.path.join(root, "pet_cache", f"{split}_{IMG_SIZE}.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["images"], d["masks"]
    images, masks = _load_split_from_torchvision(root, split, download)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save({"images": images, "masks": masks}, cache)
    return images, masks


def make_dummy(n, seed):
    """動作確認用ダミー: ランダムな楕円（前景）＋その縁の帯（境界）＋ノイズ背景。

    実データではないので、このデータでの数値は結果として扱わない。
    """
    g = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    images = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), np.uint8)
    masks = np.ones((n, IMG_SIZE, IMG_SIZE), np.uint8)  # 既定は背景(1)
    for i in range(n):
        cy, cx = g.uniform(40, 88, 2)
        ry, rx = g.uniform(15, 35, 2)
        d = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)  # 楕円の正規化距離
        masks[i][d < 1.0] = 0                     # 前景
        masks[i][(d >= 0.85) & (d < 1.15)] = 2    # 縁の帯 = 境界
        base = g.randint(0, 255, 3)[:, None, None]
        img = np.where(masks[i][None] == 1, 255 - base, base) + g.randn(3, IMG_SIZE, IMG_SIZE) * 20
        images[i] = np.clip(img, 0, 255).astype(np.uint8)
    return torch.from_numpy(images), torch.from_numpy(masks)


def get_datasets(root, seed, quick=False, dummy=False, val_ratio=0.1):
    """(train, val, test) の TensorDataset と、実データかどうかのフラグを返す。

    train/val は公式 trainval を seed で 9:1 に分割（val は best epoch の選択用）。
    test は公式 test split。quick のときはそれぞれ先頭の一部だけを使う。
    実データが取得できず quick のときだけダミーデータにフォールバックする
    （本番学習では失敗させて、ダミーで学習してしまう事故を防ぐ）。
    """
    is_real = not dummy
    if not dummy:
        try:
            tv_x, tv_y = load_oxford_pet(root, "trainval")
            te_x, te_y = load_oxford_pet(root, "test")
        except Exception as e:  # ネットワーク制限などで取得できない場合
            if not quick:
                raise RuntimeError(f"Oxford-IIIT Pet を取得できません: {e}") from e
            print(f"[warn] 実データを取得できないためダミーデータで動作確認します: {type(e).__name__}")
            is_real = False
    if not is_real:
        tv_x, tv_y = make_dummy(200, seed)
        te_x, te_y = make_dummy(80, seed + 1000)

    # trainval を seed 固定でシャッフルして train / val に分ける
    perm = torch.randperm(len(tv_x), generator=torch.Generator().manual_seed(seed))
    n_val = int(len(tv_x) * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tr = (tv_x[train_idx], tv_y[train_idx])
    va = (tv_x[val_idx], tv_y[val_idx])
    te = (te_x, te_y)
    if quick:  # 数分で終わるよう一部だけ使う
        tr, va, te = (tr[0][:96], tr[1][:96]), (va[0][:32], va[1][:32]), (te[0][:32], te[1][:32])
    return TensorDataset(*tr), TensorDataset(*va), TensorDataset(*te), is_real


# 画像の正規化（uint8 -> float）。ImageNet の平均・分散は使わず、[0,1] -> [-1,1] に揃える
def normalize(x_uint8):
    # x: (B, 3, H, W) uint8 -> float32 in [-1, 1]
    return x_uint8.float() / 127.5 - 1.0
