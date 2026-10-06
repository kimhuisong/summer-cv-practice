# DETR（Pascal VOC 2007）

**問い:** 事前学習はどれだけ効くのか。DETR はなぜ NMS なしで動くのか？

Hugging Face transformers の `DetrForObjectDetection` を使い、VOC 2007 trainval で学習・test で評価する。
同じエポック数・同じ設定で、**初期値だけが違う2条件**を比較する。

| 条件 | フォルダ名 | 初期値 |
|---|---|---|
| A | `results/A_coco_pretrained/` | COCO 事前学習済み DETR 全体（`facebook/detr-resnet-50`）。分類ヘッドだけ 20+1 クラス用に新規作成 |
| B | `results/B_imagenet_backbone/` | バックボーン（ResNet-50）だけ ImageNet 事前学習（torchvision `IMAGENET1K_V1`）。入力射影・Transformer・物体クエリ・ヘッドはランダム初期化 |

解説・考察は [`report.md`](report.md) を参照。

## ファイル構成

| ファイル | 内容 |
|---|---|
| `dataset.py` | VOC 2007 のダウンロード（取得元のフォールバックつき）・前処理・パディングつきバッチ化・ダミーデータ |
| `model.py` | 条件 A / B のモデル構築、損失計算（バックボーン+Transformer は fp16、ヘッドと損失は fp32）、後処理 |
| `train.py` | 学習。エポックごとに学習損失・監視用損失・監視用 mAP を記録 |
| `evaluate.py` | test 全体での mAP、NMS を後から足したときの mAP、重複検出の統計、検出結果・cross-attention・クエリの分業の可視化、A/B の比較 |
| `hungarian_check.py` | ハンガリアンマッチングと損失の自前実装を、ライブラリの結果と照合する（CPU で数十秒） |
| `utils.py` | シード固定・推論・mAP 計算などの共通処理 |
| `colab.ipynb` | Colab で clone → 学習 → 評価 → `results/` を zip でダウンロード |

## 実行方法

### 動作確認（CPU・数分以内）
```bash
cd 03_detr_voc2007
python hungarian_check.py                 # 自前実装とライブラリの一致確認
python train.py --condition A --quick     # CPU・データ16枚・1エポック
python train.py --condition B --quick
python evaluate.py --condition A --quick
python evaluate.py --condition B --quick
python evaluate.py --compare --quick
```
`--quick` は VOC が手元（`data/`）に無ければ**合成ダミーデータ**で動き、事前学習済み重みを取得できなければ
**ランダム初期化**で続行する（どちらもログと `train_config.json` に記録される）。数値には意味がない。
結果は `results/*_quick/` に出る（git 管理しない）。

### 本番（Google Colab, T4 GPU）
`colab.ipynb` を Colab で開いて上から実行する。1条件あたりの設定は次のとおり（A と B で共通）。

| 項目 | 値 |
|---|---|
| エポック数 | 5（4エポック終了後に学習率を1/10） |
| バッチサイズ | 8 |
| 最適化 | AdamW、学習率 1e-4（バックボーンは 1e-5）、weight decay 1e-4、勾配クリッピング 0.1 |
| 入力サイズ | 学習: 短辺を 384〜576 からランダム（長辺 ≤ 800）＋左右反転 / 評価: 短辺 512（長辺 ≤ 800） |
| 損失 | 分類 CE（物体なしの重み 0.1）×1 + L1 ×5 + GIoU ×2、デコーダ全6層に補助損失 |
| 混合精度 | バックボーン+Transformer を fp16、ヘッド・マッチング・損失は fp32 |

**時間の目安は未計測の見積もり**（T4 で1エポック数分〜10分程度、5エポック＋評価で約40分を想定）。
1エポック目のログに出る `eta` を見て、40分を大きく超えそうなら `--epochs` や `--max_size` を下げ、
A と B を**同じ設定で**実行し直すこと。

単体で実行する場合:
```bash
python train.py --condition A && python evaluate.py --condition A
python train.py --condition B && python evaluate.py --condition B
python evaluate.py --compare
```

## 結果の置き場所
```
results/
  A_coco_pretrained/          history.json, loss_curve.png, train_config.json,
                              eval_metrics.json, detections.png, cross_attention_*.png,
                              query_specialization.png, checkpoints/last.pt（git 管理しない）
  B_imagenet_backbone/        同上
  comparison/                 comparison_curves.png, summary.json, summary.md
  hungarian_check/check.json  自前実装とライブラリの照合結果
```

## データセットと事前学習済み重みの取得元
- **Pascal VOC 2007**（trainval 5,011枚 / test 4,952枚、20クラス）
  - 公式: `https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/`（Oxford VGG。torchvision と同じ URL）
  - 公式の旧ホスト名: `http://host.robots.ox.ac.uk/pascal/VOC/voc2007/`
  - ミラー: `https://pjreddie.com/media/files/`（YOLO の作者による公開ミラー）
  - 上から順に試し、いずれも torchvision に記載された公式アーカイブの **md5 で中身を検証**する。
    実際にどこから取得したかは学習ログに `[data] VOC2007 ... を取得しました: <URL>` と出る。
- **COCO 事前学習済み DETR**: Hugging Face Hub `facebook/detr-resnet-50`
- **ImageNet 事前学習済み ResNet-50**: torchvision `ResNet50_Weights.IMAGENET1K_V1`（`download.pytorch.org`。元論文の DETR と同じ重み）

## この環境での動作確認について（正直な記録）
この実装を書いた環境（クラウドのコンテナ）では、ネットワークポリシーにより
`huggingface.co`・`host.robots.ox.ac.uk`・`pjreddie.com`・`download.pytorch.org` へのアクセスが拒否された（HTTP 403）。
そのため、
- `--quick` の動作確認は **合成ダミーデータ＋ランダム初期化の重み** で行った（学習・評価・比較・可視化が最後まで動くことのみ確認）。
- 条件 A の重み読み込み処理は、ローカルに保存した 92 クラス出力の DETR（ランダム重み）を Hub の代わりに読ませて、
  「分類ヘッド以外の 528 個のテンソルがすべてコピーされ、分類ヘッドだけ作り直される」ことを確認した。
  本物の `facebook/detr-resnet-50` の読み込みは Colab で初めて行われる。読み込みが不完全な場合は
  エラーで止まるようにしてある（黙ってランダム初期化で学習しない）。
- `hungarian_check.py` はネットワーク不要なので、この環境で実行して全項目一致を確認した（`results/hungarian_check/check.json`）。

本番の数値（mAP など）は **まだ1つも得られていない**。Colab で実行後に `report.md` に記入する。

## 動作確認したバージョン
Python 3.11 / torch 2.14.1（CPU 実行）/ transformers 5.18.0 / timm 1.0.30 / torchmetrics 1.9.0 / pycocotools 2.0.11。
transformers は v4 と v5 で DETR のバックボーン読み込みや Attention の実装が変わっているため、
`colab.ipynb` では `transformers==5.18.0` に固定している。

状態: 実装済み・動作確認済み（`--quick`、ダミーデータ）。本番学習は未実行。
