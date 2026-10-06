"""U-Net（スキップ接続あり）と、同じ構造でスキップ接続だけ除いたエンコーダ・デコーダ。

比較したいのは「スキップ接続の有無」だけなので、エンコーダ・ボトルネック・
アップサンプリング・出力層はまったく同じにしてある。
唯一の違いはデコーダの DoubleConv の入力チャネル数:
  - use_skip=True : 転置畳み込みの出力 (C) とエンコーダ特徴 (C) を concat -> 2C 入力
  - use_skip=False: 転置畳み込みの出力 (C) のみ                          ->  C 入力
そのため use_skip=False の方がパラメータ数が少ない（README に差を明記）。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """3x3 畳み込み -> BatchNorm -> ReLU を2回繰り返す基本ブロック（空間サイズは保つ）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, in_ch, H, W) -> (B, out_ch, H, W)
        return self.block(x)


class UNet(nn.Module):
    """U-Net。use_skip=False でスキップ接続なしのエンコーダ・デコーダになる。

    入力 (B, 3, 128, 128)、base=32、depth=4 のときの形状:
      enc0: 32 @128  -> pool -> enc1: 64 @64 -> pool -> enc2: 128 @32 -> pool
      -> enc3: 256 @16 -> pool -> bottleneck: 512 @8
      dec3: 256 @16 -> dec2: 128 @32 -> dec1: 64 @64 -> dec0: 32 @128 -> 1x1conv: n_classes @128
    """

    def __init__(self, n_classes=3, in_ch=3, base=32, depth=4, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        chs = [base * 2 ** i for i in range(depth + 1)]  # 例: [32, 64, 128, 256, 512]

        # --- エンコーダ: DoubleConv と 2x2 max pooling（解像度を 1/2 にする）---
        self.encoders = nn.ModuleList()
        prev = in_ch
        for c in chs[:-1]:
            self.encoders.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1])

        # --- デコーダ: 転置畳み込みで解像度を2倍に -> (skip と concat) -> DoubleConv ---
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            up_out = chs[i - 1]
            self.ups.append(nn.ConvTranspose2d(chs[i], up_out, kernel_size=2, stride=2))
            dec_in = up_out * 2 if use_skip else up_out  # ← 唯一の構造差
            self.decoders.append(DoubleConv(dec_in, up_out))

        # 1x1 畳み込みでチャネルをクラス数に写す（ピクセルごとのロジット）
        self.head = nn.Conv2d(chs[0], n_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        skips = []
        for enc in self.encoders:
            x = enc(x)             # (B, C, h, w)  C は 32,64,128,256 と増える
            skips.append(x)        # スキップ用にプーリング前の特徴を保存
            x = self.pool(x)       # (B, C, h, w) -> (B, C, h/2, w/2)
        x = self.bottleneck(x)     # (B, 256, H/16, W/16) -> (B, 512, H/16, W/16)

        for up, dec in zip(self.ups, self.decoders):
            x = up(x)              # (B, 2C, h, w) -> (B, C, 2h, 2w)
            skip = skips.pop()     # 同じ解像度のエンコーダ特徴 (B, C, 2h, 2w)
            if self.use_skip:
                x = torch.cat([skip, x], dim=1)  # (B, C, 2h, 2w) x2 -> (B, 2C, 2h, 2w)
            x = dec(x)             # (B, 2C or C, 2h, 2w) -> (B, C, 2h, 2w)

        return self.head(x)        # (B, 32, H, W) -> (B, n_classes, H, W)


def count_params(model):
    """学習対象パラメータ数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant, **kwargs):
    """variant: 'unet'（スキップあり, 条件A）/ 'noskip'（スキップなし, 条件B）。"""
    if variant not in ("unet", "noskip"):
        raise ValueError(f"unknown variant: {variant}")
    return UNet(use_skip=(variant == "unet"), **kwargs)


if __name__ == "__main__":
    # 形状とパラメータ数の確認: python model.py
    for v in ("unet", "noskip"):
        m = build_model(v)
        y = m(torch.randn(2, 3, 128, 128))
        print(f"{v:7s} out={tuple(y.shape)} params={count_params(m):,}")
