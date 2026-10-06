"""共通ユーティリティ：乱数シード固定、点群の正規化、点の並べ替え。

テンソルの形状の表記：B=バッチ、N=点の数、3=(x, y, z) 座標。
"""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """random / numpy / torch / CUDA の乱数シードをすべて固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 再現性のため cuDNN の非決定的な最適化を切る（少し遅くなる）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def normalize_unit_sphere(x: torch.Tensor) -> torch.Tensor:
    """各点群を重心が原点・最遠点が半径1の球に収まるように正規化する。

    x: (B, N, 3) -> (B, N, 3)
    """
    # 重心を原点へ: (B, N, 3) - (B, 1, 3)
    x = x - x.mean(dim=1, keepdim=True)
    # 原点から最も遠い点までの距離: (B, N) -> (B, 1, 1)
    radius = x.norm(dim=2).max(dim=1)[0].view(-1, 1, 1)
    return x / radius.clamp_min(1e-8)


def canonical_order(x: torch.Tensor) -> torch.Tensor:
    """点を x 座標の昇順に並べて「固定順」にする（ベースラインMLP用）。

    点群は本来「順序のない集合」なので、MLPに入れるには何らかの規則で並べる必要がある。
    ここでは最も単純な規則として x 座標でソートする。
    x: (B, N, 3) -> (B, N, 3)
    """
    # 各点群ごとに x 座標の昇順インデックスを得る: (B, N)
    idx = x[:, :, 0].argsort(dim=1)
    # (B, N) -> (B, N, 3) に広げて、点ごと（3座標まとめて）並べ替える
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def shuffle_points(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """各点群の点の順序をランダムに入れ替える（点の集合としては同一）。

    x: (B, N, 3) -> (B, N, 3)
    """
    B, N, _ = x.shape
    # 各点群ごとに独立なランダム置換: (B, N)
    perm = torch.rand(B, N, generator=generator).argsort(dim=1).to(x.device)
    return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, 3))


def order_points(x: torch.Tensor, model_type: str, shuffle: bool = False,
                 generator: torch.Generator = None) -> torch.Tensor:
    """モデルの種類に応じて点の並びを決める。

    - mlp（ベースライン）: まず x 座標で固定順に並べる。shuffle=True ならその後ランダムに入れ替える。
    - pointnet: 並び順は何でもよい設計。shuffle=True のときだけ入れ替える。
    """
    if model_type == "mlp":
        x = canonical_order(x)
    if shuffle:
        x = shuffle_points(x, generator)
    return x
