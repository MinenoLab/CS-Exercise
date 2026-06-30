# YOLOファインチューニングテンプレート

Ultralytics YOLOを使って、任意の物体検出データセットをファインチューニングするための公開用テンプレートです。データ本体は含めず、利用者が自分の動画、画像、YOLO形式ラベルを配置して実行する構成です。

本READMEでは、Ubuntu/Linux環境でGPUを使って学習する前提です。


## 前提条件

本テンプレートは、以下の環境を前提とします。

- OS: Ubuntu / Linux
- Python: 3.11
- パッケージ管理: `uv`
- GPU: NVIDIA GPUを使用できる環境
- データ形式: YOLO形式の物体検出データセット
- アノテーション形式: Label StudioからYOLO形式でエクスポートしたラベル

Label Studioを使ったアノテーション環境の構築やエクスポート方法は、[Label Studio 環境構築まとめ](../../label-studio_setup/label-studio_setup) を参照してください。

GPUが認識されているかは、次のコマンドで確認できます。

```bash
# NVIDIA GPUが認識されているか確認する
nvidia-smi
```

## ディレクトリ構成

```text
yolo_finetune_template/
├── configs/
│   └── dataset.yaml.example
├── videos/input/
├── frames/raw/
├── annotations/yolo_raw/
├── datasets/
├── scripts/
│   ├── extract_frames.py
│   ├── split_dataset.py
│   ├── make_data_yaml.py
│   ├── check_dataset.py
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── predict_images.py
│   ├── predict_video.py
│   └── list_prediction_errors.py
├── runs/experiments/
├── pyproject.toml
└── README.md
```

## 環境構築

Python 3.11の仮想環境を作成し、必要なライブラリをインストールします。

```bash
# Python 3.11をインストールする
uv python install 3.11

# Python 3.11の仮想環境を作成する
uv venv --python 3.11

# 仮想環境を有効化する
source .venv/bin/activate

# pyproject.tomlに書かれた依存ライブラリをインストールする
uv sync
```

## 方法A: YOLO形式データセットから始める

すでにYOLO形式データセットがある場合は、`datasets/custom_yolo/` に配置します。
ここでは、検証用フォルダ名が `valid` のデータセットを例にします。
実際には自分のデータセットのディレクトリ構成に合わせて設定してください。

```text
datasets/custom_yolo/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
```

画像ファイルとラベルファイルは、拡張子を除いたファイル名が対応している必要があります。

```text
datasets/custom_yolo/images/train/sample_000001.jpg
datasets/custom_yolo/labels/train/sample_000001.txt
```
※一般的にdata.yamlは設定されていますが、ない場合は以下を参考にしてください。

`data.yaml` を生成します。

```bash
# images/train, images/valid, images/test を使うdata.yamlを生成する
uv run python scripts/make_data_yaml.py \
  --dataset-dir datasets/custom_yolo \
  --train images/train \
  --val images/valid \
  --test images/test \
  --class-names target_object
```

引数:

- `--dataset-dir`: `data.yaml`を作成するデータセットディレクトリ
- `--train`: 学習用画像ディレクトリ。`--dataset-dir`からの相対パス
- `--val`: 検証用画像ディレクトリ。`--dataset-dir`からの相対パス
- `--test`: test用画像ディレクトリ。`--dataset-dir`からの相対パス
- `--class-names`: クラスID順のクラス名

`test` データがない場合は、`--test ""` と指定します。

```bash
# test splitを書かないdata.yamlを生成する
uv run python scripts/make_data_yaml.py \
  --dataset-dir datasets/custom_yolo \
  --train images/train \
  --val images/valid \
  --test "" \
  --class-names target_object
```

引数:

- `--dataset-dir`: `data.yaml`を作成するデータセットディレクトリ
- `--train`: 学習用画像ディレクトリ相対パス
- `--val`: 検証用画像ディレクトリ相対パス
- `--test ""`: test splitを `data.yaml` に書かない指定
- `--class-names`: クラスID順のクラス名

複数クラスの場合は、クラスIDの順番にクラス名を並べます。

```bash
# 3クラス用のdata.yamlを生成する
uv run python scripts/make_data_yaml.py \
  --dataset-dir datasets/custom_yolo \
  --train images/train \
  --val images/valid \
  --test images/test \
  --class-names class_a class_b class_c
```

引数:

- `--dataset-dir`: `data.yaml`を作成するデータセットディレクトリ
- `--train`: 学習用画像ディレクトリ相対パス
- `--val`: 検証用画像ディレクトリ相対パス
- `--test`: test用画像ディレクトリ相対パス
- `--class-names`: クラスID順のクラス名。ここでは3クラス分を指定

生成される `data.yaml` の例です。

```yaml
path: datasets/custom_yolo
train: images/train
val: images/valid
test: images/test
names:
  0: target_object
```

## 方法B: 動画から画像を切り出して始める

動画を `videos/input/` に置きます。

```text
videos/input/sample01.mp4
videos/input/sample02.mp4
```

1秒に1枚の間隔で画像を切り出します。

```bash
# videos/input/の動画から1秒ごとに画像を切り出す
uv run python scripts/extract_frames.py \
  --video-dir videos/input \
  --output-dir frames/raw \
  --interval-sec 1.0
```

引数:

- `--video-dir`: 入力動画を置いたディレクトリ
- `--output-dir`: 切り出した画像の保存先ディレクトリ
- `--interval-sec`: 画像を切り出す間隔。単位は秒

切り出した画像をLabel Studioでアノテーションし、YOLO形式でエクスポートしたラベルを `annotations/yolo_raw/` に保存します。

```text
frames/raw/sample01_000001.jpg
annotations/yolo_raw/sample01_000001.txt
```

未分割データを `train/val/test` に分割します。このスクリプトで作成する検証用フォルダ名は `val` です。

```bash
# frames/raw と annotations/yolo_raw を train/val/test に分割する
uv run python scripts/split_dataset.py \
  --images-dir frames/raw \
  --labels-dir annotations/yolo_raw \
  --output-dir datasets/custom_yolo \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

引数:

- `--images-dir`: 分割前の画像ディレクトリ
- `--labels-dir`: 分割前のYOLO形式ラベルディレクトリ
- `--output-dir`: 分割後データセットの保存先ディレクトリ
- `--val-ratio`: 検証用データの割合
- `--test-ratio`: test用データの割合
- `--seed`: 分割結果を固定するための乱数シード

グループ単位で分割したい場合は、ファイル名の接頭辞をグループIDとして使えます。例えば `sample01_000001.jpg` なら `sample01` がグループIDです。

```bash
# sample01をtest、sample02をvalに固定して分割する
uv run python scripts/split_dataset.py \
  --images-dir frames/raw \
  --labels-dir annotations/yolo_raw \
  --output-dir datasets/custom_yolo \
  --split-by-group \
  --test-groups sample01 \
  --val-groups sample02
```

引数:

- `--images-dir`: 分割前の画像ディレクトリ
- `--labels-dir`: 分割前のYOLO形式ラベルディレクトリ
- `--output-dir`: 分割後データセットの保存先ディレクトリ
- `--split-by-group`: ファイル名の接頭辞をグループIDとして分割する指定
- `--test-groups`: testに固定するグループID
- `--val-groups`: valに固定するグループID

`data.yaml` を生成します。

```bash
# split_dataset.pyで作成した images/val 構成に合わせてdata.yamlを生成する
uv run python scripts/make_data_yaml.py \
  --dataset-dir datasets/custom_yolo \
  --train images/train \
  --val images/val \
  --test images/test \
  --class-names target_object
```

引数:

- `--dataset-dir`: `data.yaml`を作成するデータセットディレクトリ
- `--train`: 学習用画像ディレクトリ。`--dataset-dir`からの相対パス
- `--val`: 検証用画像ディレクトリ。`split_dataset.py`の出力に合わせて `images/val` を指定
- `--test`: test用画像ディレクトリ。`--dataset-dir`からの相対パス
- `--class-names`: クラスID順のクラス名

## データセット確認

学習前に、画像とラベルの対応、YOLO形式の値、クラス数を確認します。

```bash
# data.yamlを読み、train/val/testの画像とラベルを確認する
uv run python scripts/check_dataset.py \
  --dataset-dir datasets/custom_yolo
```

引数:

- `--dataset-dir`: 確認対象のデータセットディレクトリ。中の `data.yaml` を読み取り

動画から切り出した直後の未分割データを確認したい場合は、次のようにします。

```bash
# frames/raw と annotations/yolo_raw の対応を確認する
uv run python scripts/check_dataset.py \
  --check-raw
```

引数:

- `--check-raw`: `frames/raw` と `annotations/yolo_raw` の対応確認に切り替える指定

## 学習

```bash
# YOLOをGPU(device 0)でファインチューニングする
uv run python scripts/train_yolo.py \
  --data datasets/custom_yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --patience 30 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name custom_object_ep100
```

引数:

- `--data`: 学習に使うデータセット設定ファイル
- `--model`: ファインチューニング元のYOLO重み
- `--epochs`: 最大学習エポック数
- `--patience`: 改善が止まったときに早期終了するまでの待機エポック数
- `--imgsz`: 入力画像サイズ
- `--batch`: バッチサイズ
- `--device`: 使用するGPU番号。`0`は1枚目のGPU
- `--name`: 実験名。結果保存ディレクトリ名に使用

主な出力先です。

```text
runs/experiments/custom_object_ep100/train/results.csv
runs/experiments/custom_object_ep100/train/results.png
runs/experiments/custom_object_ep100/train/weights/best.pt
runs/experiments/custom_object_ep100/metadata/train_metadata.json
```

## test評価

```bash
# 学習済みbest.ptをtest splitで評価する
uv run python scripts/evaluate_yolo.py \
  --model runs/experiments/custom_object_ep100/train/weights/best.pt \
  --data datasets/custom_yolo/data.yaml \
  --split test \
  --conf 0.25 \
  --iou 0.7 \
  --name test_evaluation
```

引数:

- `--model`: 評価に使う学習済み重み
- `--data`: 評価対象のデータセット設定ファイル
- `--split`: 評価対象の分割。ここでは `test`
- `--conf`: 検出結果として採用する信頼度の閾値
- `--iou`: NMSで重なったBBoxを統合するときのIoU閾値
- `--name`: 評価結果の保存名

出力先です。

```text
runs/experiments/custom_object_ep100/test_evaluation/evaluation_metadata.json
runs/experiments/custom_object_ep100/test_evaluation/confusion_matrix.png
runs/experiments/custom_object_ep100/test_evaluation/confusion_matrix_normalized.png
runs/experiments/custom_object_ep100/test_evaluation/BoxPR_curve.png
```

## 画像推論

```bash
# test画像に対して推論し、検出結果画像を保存する
uv run python scripts/predict_images.py \
  --model runs/experiments/custom_object_ep100/train/weights/best.pt \
  --source datasets/custom_yolo/images/test \
  --conf 0.25 \
  --name test_images_conf025
```

引数:

- `--model`: 推論に使う学習済み重み
- `--source`: 推論対象の画像ディレクトリ
- `--conf`: 検出結果として採用する信頼度の閾値
- `--name`: 推論結果の保存名

## 動画推論

```bash
# 入力動画に対して推論し、検出結果動画をmp4で保存する
uv run python scripts/predict_video.py \
  --model runs/experiments/custom_object_ep100/train/weights/best.pt \
  --source videos/input/sample01.mp4 \
  --conf 0.25 \
  --name sample01_conf025 \
  --output-format mp4
```

引数:

- `--model`: 推論に使う学習済み重み
- `--source`: 推論対象の動画ファイル
- `--conf`: 検出結果として採用する信頼度の閾値
- `--name`: 推論結果の保存名
- `--output-format`: 出力動画の形式

## エラー分析

正解ラベルと推論結果を比較し、間違えた画像、FP画像、FN画像をCSVと画像で出力します。

```bash
# test splitの推論結果を正解ラベルと比較し、誤検出・未検出を出力する
uv run python scripts/list_prediction_errors.py \
  --model runs/experiments/custom_object_ep100/train/weights/best.pt \
  --dataset-dir datasets/custom_yolo \
  --split test \
  --conf 0.25 \
  --match-iou 0.5 \
  --name test_error_conf025_iou05 \
  --save-images
```

引数:

- `--model`: 推論に使う学習済み重み
- `--dataset-dir`: 正解ラベルと画像を含むデータセットディレクトリ
- `--split`: エラー分析対象の分割。ここでは `test`
- `--conf`: 検出結果として採用する信頼度の閾値
- `--match-iou`: 正解BBoxと推論BBoxを同一物体とみなすIoU閾値
- `--name`: エラー分析結果の保存名
- `--save-images`: エラー画像を保存する指定

## 結果の保存場所

```text
runs/experiments/<experiment_name>/
├── train/
├── test_evaluation/
├── predictions/
│   ├── images/
│   └── videos/
├── error_analysis/
└── metadata/
```
