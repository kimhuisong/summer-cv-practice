# U-Net（Oxford-IIIT Pet）— スキップ接続は何を担っているのか？

U-Net（条件A）と、**スキップ接続だけを取り除いた同構造のエンコーダ・デコーダ**（条件B）を比較し、
スキップ接続が細かい位置情報（特に「境界」クラス）の復元に効くかを検証する。
仮説: スキップ接続を外すと、全体の精度より境界クラスの精度が大きく落ちる。
詳細は [`report.md`](report.md)。

## 状態
- 実装済み / `--quick`（ダミーデータ）での動作確認済み。
- **本番学習（Colab T4）は未実行**。結果は `report.md` で「TODO: Colab実行後に記入」のまま。
- 実データでの前処理（torchvision からの読み込み）は、この開発環境が配布元に接続できなかったため**未検証**。Colab での初回実行時に確認する。

## ファイル
| ファイル | 内容 |
|---|---|
| `model.py` | `UNet(use_skip=True/False)`。`python model.py` で形状とパラメータ数を表示 |
| `data.py` | Oxford-IIIT Pet の読み込み（128×128、3クラス）とダミーデータ |
| `utils.py` | シード固定、混同行列、IoU |
| `train.py` | 学習（`--variant unet|noskip`, `--seed`, `--quick`） |
| `evaluate.py` | test 評価（mIoU・クラス別IoU）と A/B の予測比較図 |
| `colab.ipynb` | Colab 用。clone → quick確認 → 学習 → 評価 → results を zip でダウンロード |
| `report.md` | レポート |

## パラメータ数（`python model.py` の出力、base=32）
| 条件 | パラメータ数 |
|---|---|
| A: U-Net（スキップあり） | 7,763,107 |
| B: スキップなし | 6,979,747 |
| 差（A − B） | 783,360（B は約10%少ない） |

差の原因: デコーダ各段の DoubleConv の入力が、A は転置畳み込み出力とエンコーダ特徴の連結（2C チャネル）、B は転置畳み込み出力のみ（C チャネル）になるため。それ以外の層は同一。
B が劣る場合、スキップがないことと、パラメータが少ないことの影響は本実験だけでは完全には切り分けられない（`report.md` の限界を参照）。

## 実行方法
```bash
pip install -r ../requirements.txt
cd 02_unet_oxfordpet

# 動作確認（CPU・データの一部・1エポック。数分以内）。結果は results/quick/（git管理外）
python train.py --variant unet   --quick
python train.py --variant noskip --quick
python evaluate.py --quick

# 本番（GPU 推奨。Colab では colab.ipynb を実行）
for s in 0 1 2; do
  python train.py --variant unet   --seed $s --epochs 30
  python train.py --variant noskip --seed $s --epochs 30
done
python evaluate.py --seeds 0 1 2
```
Colab: `colab.ipynb` を開き、ランタイムを GPU（T4）にして上から実行する。最後のセルで `results.zip` をダウンロードできる。
1条件・1 seed あたり40分以内を想定した設定（30 epoch, AMP）。**実測時間は未計測**（`train_info.json` の `train_sec` を見る）。

## 出力（`results/`）
```
results/<variant>_seed<k>/        # variant = unet | noskip
  history.json                    # エポックごとの train loss / val mIoU / val クラス別IoU
  train_info.json                 # パラメータ数・best val mIoU・学習時間
  learning_curves.png             # 学習曲線
  checkpoints/best.pth            # git 管理しない
results/comparison/
  metrics.json                    # test の mIoU・クラス別IoU（seed 平均±標準偏差・A に対する B の相対低下率）
  per_class_iou.png               # クラス別 IoU の棒グラフ
  prediction_comparison.png       # 同じテスト画像に対する A・B の予測マスクの並置
```

## データセットと取得元
- **Oxford-IIIT Pet**（trainval 3,680 / test 3,669 枚）。`torchvision.datasets.OxfordIIITPet(target_types="segmentation", download=True)` が公式配布元（Oxford VGG, `https://www.robots.ox.ac.uk/~vgg/data/pets/`）から取得する。
- マスクは trimap（1=前景, 2=背景, 3=境界）を 0=前景, 1=背景, 2=境界 に振り替えて使う。128×128 に画像は bilinear、マスクは nearest でリサイズし、`data/pet_cache/` にキャッシュする（git 管理外）。
- 分割: 公式 trainval を seed 固定で 9:1 に分けて train/val（val は best epoch の選択用）、公式 test で最終評価。
- **ネットワーク制限について**: この実装を書いた開発環境では公式配布元（robots.ox.ac.uk / thor.robots.ox.ac.uk）と Hugging Face への接続が拒否された。そのため開発環境では**ダミーデータ**（ランダムな楕円＋縁の帯を境界とした合成画像）で `--quick` の動作確認だけを行った。ダミーデータの数値は結果として扱わない。Colab で公式配布元に接続できない場合は、信頼できるミラーを使い、ここに取得元を追記すること。
- 本番学習（`--quick` なし）で実データを取得できない場合は、ダミーで学習してしまわないようエラーで止まる。
