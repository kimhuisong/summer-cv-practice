# PointNet（ModelNet40）

**問い：点の並び順に依存しない設計は、本当に必要なのか？**
点群（順序のない点の集合）に対し、順序不変な設計（PointNet）と、点を固定順に並べて平坦化した普通のMLPを比べ、
「テスト時に点の順序をシャッフルしたら精度がどうなるか」で検証する。

## 状態
- 実装: 完了。`--quick`（ダミーデータ・CPU）でパイプライン全体の動作を確認済み。
- **本番学習（Colab T4）: 未実行。** report.md の結果欄は `TODO: Colab実行後に記入`。
- この環境（サンドボックス）では ModelNet40 の配布元に接続できなかったため、**実データでの実行は一度もしていない**。
  動作確認はダミーデータ（後述）で行った。

## 比較条件
| 条件名 | モデル | 実行 |
|---|---|---|
| `pointnet` | 条件A：共有MLP → max pooling → 分類MLP | `python train.py --model pointnet` |
| `pointnet_tnet` | 条件A + T-Net（入力3×3・特徴64×64） | `python train.py --model pointnet --use_tnet` |
| `mlp_baseline` | 条件B：点を x 座標昇順に並べて平坦化（N×3=3072次元）→ 普通のMLP | `python train.py --model mlp` |

## 評価（`evaluate.py`、学習後に自動実行）
- テスト精度（通常） と、テスト時に点の順序をランダムにシャッフルした精度（5シードの平均±標準偏差）
- 順序不変性の数値チェック：同じ点群を並べ替えたときの logits の最大絶対差
- 入力点数 1024 / 512 / 256 での精度（PointNetのみ。MLPは入力次元が固定のため対象外。5回の乱数間引きの平均±標準偏差）
- 混同行列（`confusion_matrix.png`）と誤分類の多いペア、点群と予測ラベルの3D可視化（`predictions_3d.png`）
- `python evaluate.py --compare` で3条件の比較図 `results/comparison.png` と `comparison.json` を作る

## データ
- ModelNet40 の `modelnet40_ply_hdf5_2048`（PointNet 論文と同じ標準配布形式。学習 9,840 / テスト 2,468 形状、各2048点）。
- 前処理：1形状から**1024点**をサンプル（学習は毎回ランダム、テストは先頭1024点で固定）→ 重心を原点に移し、最遠点が半径1になるよう正規化。
- 学習時の拡張：ランダムなスケール(0.8–1.25)・平行移動(±0.1)・ガウスノイズ(σ=0.01, 上限0.05)。回転はしない（ModelNetは向きが揃っているため）。
- 取得元（`dataset.py` の `MIRROR_URLS` を先頭から順に試す。`--data_url` を渡すとそれを最優先で試す）：
  1. 公式配布元（Stanford）: `https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip`
     — Colab からタイムアウトしたとの報告あり。
  2. Hugging Face ミラー: `https://huggingface.co/datasets/zhangtao-whu/point_cloud_datasets/resolve/main/modelnet40_ply_hdf5_2048.zip`
     — データセットリポジトリ `zhangtao-whu/point_cloud_datasets` のルートに同名zip（約435MB）があることは検索結果で確認したが、
       **この直URLでのダウンロードは未検証**（この環境は HF に接続できない）。使えなければ次の手動配置へ。
  - 通信失敗は同じURLを2回まで再試行し、zip破損・検証不一致は次のミラーへ進む。全て失敗したらエラーで止まる。
  - **取得後（と既存データの使用前）に必ず検証する**：`ply_data_train0-4.h5` / `ply_data_test0-1.h5` の存在、
    `data` の形状 `(n, 2048, 3)`、サンプル数 **train 9840 / test 2468**。1つでも違えばエラー。
  - 手動配置：zip を別経路で入手して `01_pointnet_modelnet40/data/` に展開する
    （`data/modelnet40_ply_hdf5_2048/ply_data_train0.h5` などが並ぶ形）。次回の実行時に上記の検証が走る。
- **本番実行（`--quick` なし）でデータを用意できない／検証に通らない場合は、ダミーデータに切り替えず、エラーで停止する**
  （`--quick` のときだけ、警告つきでダミーにフォールバックする）。
- **ダミーデータ**（`--quick` / `--dummy` 専用）：8種の基本形状（球・立方体・円柱・円錐・トーラス・円盤・らせん・十字）× 5種の縦横比 = 40クラスを手続き的に生成。
  本物の ModelNet40 ではなく、**動作確認にしか使わない**（`--quick` か `--dummy` 指定時のみ使われる）。結果JSONの `data_source` が `"dummy"` ならダミー。
- `data/` は git 管理しない。

## 使い方
```bash
pip install -r ../requirements.txt

# 動作確認（CPUのみ・データの一部・1エポック。数分以内。実データが無ければダミーデータにフォールバック）
python train.py --quick                       # PointNet
python train.py --quick --model mlp           # ベースライン

# 本番（Colab T4。colab.ipynb で一括実行できる）
python train.py --model pointnet
python train.py --model pointnet --use_tnet
python train.py --model mlp
python evaluate.py --compare
```
主な引数：`--epochs 100` `--batch_size 32` `--lr 1e-3` `--weight_decay 1e-4` `--num_points 1024` `--seed 0`。

## Colab
`colab.ipynb` を開き、T4 GPU を選んで上から実行する。リポジトリを clone → 3条件を学習 → 比較 → `results.zip` をダウンロード。
「1条件あたり40分以内」は設計上の目標で、**T4での実測時間は未確認**（最初のエポックの `elapsed_sec` を見て `--epochs` を調整する）。

## 結果の置き場所
`results/<条件名>/` に以下を保存（`quick_*` は動作確認用で git 管理しない）。
- `history.json`（エポックごとの loss / 精度）、`training_curves.png`
- `metrics.json`（上記の評価指標）、`confusion_matrix.png`/`.npy`、`predictions_3d.png`
- `checkpoints/last.pt`（git 管理しない）

チェックポイントは**最終エポック**を使う。検証用の分割を作っていないため、テスト精度でベストエポックを選ぶとテストへのリークになるから。
エポックごとのテスト精度は学習曲線の監視用で、モデル選択には使っていない。

## ファイル
`model.py`（PointNet / T-Net / FlattenMLP）、`dataset.py`（読込・DL・ダミー・拡張）、`utils.py`（シード・正規化・並べ替え）、
`train.py`、`evaluate.py`、`colab.ipynb`、`report.md`。

## 設計上の注意
- ベースラインの「固定順」は **x座標の昇順ソート**。h5 の点の並びはランダムなので、何も並べないとMLPは学習のしようがない。
  ソートはより素朴な規則であり、別の規則（Z曲線など）で結論が変わる可能性がある（report.md の限界に記載）。
- PointNet は max pooling の軸が点の軸なので、並べ替えに対して**厳密に**不変（浮動小数点でも logits の差は0になるはず）。
  シャッフルで精度が変わるとしたらバグなので、`max_abs_logit_diff_under_shuffle` で確認する。
