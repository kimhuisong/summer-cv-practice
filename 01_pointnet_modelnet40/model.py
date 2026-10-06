"""PointNet（条件A）とフラット化MLP（条件B：ベースライン）。

入力はどちらも x: (B, N, 3)（B=バッチ, N=点の数, 3=座標）。
出力は (logits: (B, num_classes), reg: スカラー)。reg は T-Net の正則化項（T-Net不使用なら0）。
"""
import torch
import torch.nn as nn


class TNet(nn.Module):
    """入力（または特徴）を k×k の行列で変換するミニネットワーク（空間変換ネットワーク）。

    点ごとの共有MLP → max pooling → 全結合 で k×k 行列を回帰する。
    max pooling を使っているので、この T-Net 自体も点の順序に不変。
    """

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(                       # 点ごとに共有されるMLP（Conv1d, kernel=1）
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )
        # 初期状態で「何もしない変換（単位行列）」になるよう最終層を初期化する
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        # x: (B, k, N)
        B = x.shape[0]
        h = self.mlp(x)                  # (B, k, N) -> (B, 1024, N)
        h = h.max(dim=2)[0]              # 点方向に max: (B, 1024, N) -> (B, 1024)
        h = self.fc(h)                   # (B, 1024) -> (B, k*k)
        eye = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        return (h + eye).view(B, self.k, self.k)   # 単位行列 + 残差: (B, k, k)


def orthogonality_regularizer(mat):
    """||I - A A^T||_F^2 の平均。特徴変換行列を直交行列に近づける（PointNet論文の正則化）。

    mat: (B, k, k) -> スカラー
    """
    k = mat.shape[1]
    eye = torch.eye(k, device=mat.device).unsqueeze(0)       # (1, k, k)
    diff = eye - torch.bmm(mat, mat.transpose(1, 2))         # (B, k, k)
    return (diff ** 2).sum(dim=(1, 2)).mean()


class PointNetCls(nn.Module):
    """PointNet 分類器：共有MLP → max pooling（対称関数）→ 分類MLP。"""

    def __init__(self, num_classes=40, use_tnet=False, dropout=0.3):
        super().__init__()
        self.use_tnet = use_tnet
        if use_tnet:
            self.input_tnet = TNet(3)     # 入力座標(3次元)の整列
            self.feature_tnet = TNet(64)  # 64次元特徴の整列
        # 点ごとの共有MLP（Conv1d の kernel=1 は「全点に同じ全結合を適用」と等価）
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)  Conv1d は (B, チャンネル, 点) の並びを期待する
        x = x.transpose(1, 2)
        reg = x.new_zeros(())
        if self.use_tnet:
            t_in = self.input_tnet(x)            # (B, 3, 3)
            x = torch.bmm(t_in, x)               # (B, 3, 3) x (B, 3, N) -> (B, 3, N)
        x = self.mlp1(x)                         # (B, 3, N) -> (B, 64, N)
        if self.use_tnet:
            t_feat = self.feature_tnet(x)        # (B, 64, 64)
            x = torch.bmm(t_feat, x)             # (B, 64, N)
            reg = orthogonality_regularizer(t_feat)
        x = self.mlp2(x)                         # (B, 64, N) -> (B, 1024, N)
        # ★ 対称関数：点の軸(N)方向に max を取る。N 個の点の並びを変えても結果は同じ。
        g = x.max(dim=2)[0]                      # (B, 1024, N) -> (B, 1024)  大域特徴
        return self.classifier(g), reg           # (B, 1024) -> (B, num_classes)


class FlattenMLP(nn.Module):
    """ベースライン：点を固定順に並べて N*3 次元のベクトルに平坦化し、普通のMLPに入れる。

    点の順序に関する不変性を一切持たない（i番目の入力は常に「i番目の点」として扱われる）。
    """

    def __init__(self, num_points=1024, num_classes=40, hidden=(1024, 512, 256), dropout=0.3):
        super().__init__()
        layers, d = [], num_points * 3
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, 3) -> (B, N*3)  点の並び順がそのまま入力次元の並びになる
        x = x.flatten(1)
        return self.net(x), x.new_zeros(())      # (B, N*3) -> (B, num_classes)


def build_model(cfg):
    """cfg: {"model": "pointnet"|"mlp", "use_tnet", "num_points", "num_classes", "dropout"}"""
    if cfg["model"] == "pointnet":
        return PointNetCls(cfg["num_classes"], cfg["use_tnet"], cfg["dropout"])
    if cfg["model"] == "mlp":
        return FlattenMLP(cfg["num_points"], cfg["num_classes"], dropout=cfg["dropout"])
    raise ValueError(f"unknown model: {cfg['model']}")
