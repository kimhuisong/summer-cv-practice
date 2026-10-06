"""ModelNet40 の読み込み・ダウンロード・ダミーデータ生成・データ拡張。

本番データ: modelnet40_ply_hdf5_2048（PointNet 論文が使った標準形式）
  - 学習 9,840 / テスト 2,468 形状、1形状あたり 2048 点（表面を一様サンプリング済み）
  - ply_data_train{0..4}.h5 / ply_data_test{0,1}.h5 に data:(n,2048,3), label:(n,1)
取得元（公式配布）: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
  ※ 配布元が落ちている場合は --data_url で別のミラーを指定するか、zip を手で
    data/ に置いて展開する（README.md 参照）。

ネットワーク制限で取得できない場合は、--quick / --dummy のときに限り
ダミーデータ（手続き的に作った40クラスの点群）で動作確認だけ行う。
"""
import glob
import os
import socket
import urllib.request
import zipfile

import numpy as np
import torch

from utils import normalize_unit_sphere

DEFAULT_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
H5_DIRNAME = "modelnet40_ply_hdf5_2048"

# ModelNet40 の40クラス名（h5 のラベル番号 0..39 に対応するアルファベット順）
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]


# ---------------------------------------------------------------- 本番データ
def _find_h5_dir(root):
    """root の下から ply_data_train*.h5 のあるディレクトリを探す。"""
    for d in (os.path.join(root, H5_DIRNAME), root):
        if glob.glob(os.path.join(d, "ply_data_train*.h5")):
            return d
    return None


def _download(root, url, timeout=30):
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, os.path.basename(url))
    print(f"[data] ダウンロード: {url}")
    socket.setdefaulttimeout(timeout)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    os.remove(zip_path)


def _load_h5_split(h5_dir, split):
    import h5py  # 本番データを読むときだけ必要

    xs, ys = [], []
    for path in sorted(glob.glob(os.path.join(h5_dir, f"ply_data_{split}*.h5"))):
        with h5py.File(path, "r") as f:
            xs.append(f["data"][:])    # (n, 2048, 3)
            ys.append(f["label"][:])   # (n, 1)
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32))        # (M, 2048, 3)
    y = torch.from_numpy(np.concatenate(ys).reshape(-1).astype(np.int64))  # (M,)
    return x, y


# ---------------------------------------------------------------- ダミーデータ
def _surface_points(shape_id, n, rng):
    """8種類の基本形状の表面から n 点をサンプルする。返り値: (n, 3)"""
    if shape_id == 0:    # 球
        p = rng.normal(size=(n, 3))
        return p / np.linalg.norm(p, axis=1, keepdims=True)
    if shape_id == 1:    # 立方体の表面
        p = rng.uniform(-1, 1, size=(n, 3))
        axis = rng.integers(0, 3, size=n)
        p[np.arange(n), axis] = rng.choice([-1.0, 1.0], size=n)
        return p
    if shape_id == 2:    # 円柱の側面
        t, h = rng.uniform(0, 2 * np.pi, n), rng.uniform(-1, 1, n)
        return np.stack([np.cos(t), np.sin(t), h], 1)
    if shape_id == 3:    # 円錐の側面
        t, h = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 1, n)
        return np.stack([(1 - h) * np.cos(t), (1 - h) * np.sin(t), 2 * h - 1], 1)
    if shape_id == 4:    # トーラス
        u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        return np.stack([(1 + 0.35 * np.cos(v)) * np.cos(u),
                         (1 + 0.35 * np.cos(v)) * np.sin(u),
                         0.35 * np.sin(v)], 1)
    if shape_id == 5:    # 薄い円盤
        t, r = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(0, 1, n))
        return np.stack([r * np.cos(t), r * np.sin(t), rng.normal(0, 0.02, n)], 1)
    if shape_id == 6:    # らせん
        t = rng.uniform(0, 6 * np.pi, n)
        return np.stack([np.cos(t), np.sin(t), t / (3 * np.pi) - 1], 1) + rng.normal(0, 0.02, (n, 3))
    # shape_id == 7: 3軸の十字
    p = rng.normal(0, 0.03, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    p[np.arange(n), axis] = rng.uniform(-1, 1, size=n)
    return p


_SCALE_PROFILES = np.array([[1, 1, 1], [1.6, 1, 1], [1, 1.6, 1], [1, 1, 1.6], [1.5, 1.5, 0.6]],
                           dtype=np.float32)


def make_dummy_dataset(n_train_per_class, n_test_per_class, n_points=1024, seed=0):
    """8形状 × 5種の縦横比 = 40クラスのダミー点群。本物のModelNet40ではない。"""
    rng = np.random.default_rng(seed)

    def build(n_per_class):
        xs, ys = [], []
        for c in range(40):
            shape_id, scale_id = divmod(c, 5)
            for _ in range(n_per_class):
                p = _surface_points(shape_id, n_points, rng)
                p = p * _SCALE_PROFILES[scale_id] * rng.uniform(0.9, 1.1, size=3)
                xs.append(p.astype(np.float32))
                ys.append(c)
        return torch.from_numpy(np.stack(xs)), torch.tensor(ys, dtype=torch.long)

    train_x, train_y = build(n_train_per_class)
    test_x, test_y = build(n_test_per_class)
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------- 入口
def load_modelnet40(root, quick=False, dummy=False, url=DEFAULT_URL, seed=0):
    """データを読み込んで dict で返す。

    返り値: {train_x:(M,P,3), train_y:(M,), test_x, test_y, class_names, source}
      P は1形状あたりの保存点数（本番は2048、ダミーは1024）。
      source は "modelnet40" か "dummy"（結果JSONに必ず記録して区別する）。
    """
    if not dummy:
        h5_dir = _find_h5_dir(root)
        if h5_dir is None:
            try:
                _download(root, url)
                h5_dir = _find_h5_dir(root)
            except Exception as e:  # ネットワーク不通・配布元停止など
                print(f"[data] ModelNet40 を取得できませんでした: {type(e).__name__}: {e}")
        if h5_dir is not None:
            train_x, train_y = _load_h5_split(h5_dir, "train")
            test_x, test_y = _load_h5_split(h5_dir, "test")
            if quick:  # 動作確認用にデータの一部だけ使う（シード固定のランダム抽出）
                g = torch.Generator().manual_seed(seed)
                tr = torch.randperm(len(train_x), generator=g)[:800]
                te = torch.randperm(len(test_x), generator=g)[:400]
                train_x, train_y, test_x, test_y = train_x[tr], train_y[tr], test_x[te], test_y[te]
            return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                        class_names=MODELNET40_CLASSES, source="modelnet40")
        if not quick:
            raise RuntimeError(
                "ModelNet40 が見つからず、ダウンロードにも失敗しました。"
                "README.md の手順で data/ に配置するか --data_url を指定してください。"
                "（動作確認だけなら --quick か --dummy）")
        print("[data] !!! ダミーデータにフォールバックします（本物の ModelNet40 ではありません）")

    train_x, train_y, test_x, test_y = make_dummy_dataset(10, 5, seed=seed)
    return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                class_names=[f"dummy_{i}" for i in range(40)], source="dummy")


# ---------------------------------------------------------------- 前処理・拡張
def sample_train_points(pool, num_points):
    """保存点(B, P, 3)から、各形状ごとにランダムに num_points 点を選ぶ（学習用）。

    pool: (B, P, 3) -> (B, num_points, 3)   ※ 選んだ点の並びもランダムになる
    """
    B, P, _ = pool.shape
    # 各形状で P 点のランダム置換を作り、先頭 num_points 個を使う: (B, num_points)
    idx = torch.rand(B, P).argsort(dim=1)[:, :num_points]
    return pool.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def sample_test_points(pool, num_points):
    """テスト用：保存点の先頭 num_points 点を使う（固定・再現可能）。(B, P, 3) -> (B, num_points, 3)"""
    return pool[:, :num_points]


def subsample_points(x, n, generator):
    """入力点数を減らす実験用：x の各点群からランダムに n 点を選ぶ。(B, N, 3) -> (B, n, 3)"""
    B, N, _ = x.shape
    idx = torch.rand(B, N, generator=generator).argsort(dim=1)[:, :n]
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def augment(x, jitter_sigma=0.01, jitter_clip=0.05):
    """学習時のデータ拡張：ランダムなスケール・平行移動・微小ノイズ。(B, N, 3) -> (B, N, 3)"""
    B = x.shape[0]
    scale = torch.empty(B, 1, 1, device=x.device).uniform_(0.8, 1.25)   # (B, 1, 1)
    shift = torch.empty(B, 1, 3, device=x.device).uniform_(-0.1, 0.1)   # (B, 1, 3)
    x = x * scale + shift
    noise = (torch.randn_like(x) * jitter_sigma).clamp(-jitter_clip, jitter_clip)
    return x + noise


def prepare_points(pool, num_points, train):
    """保存点 -> 正規化済みの入力点群。学習時はランダムサンプル、評価時は固定サンプル。"""
    x = sample_train_points(pool, num_points) if train else sample_test_points(pool, num_points)
    return normalize_unit_sphere(x)
