# 第2部：画像認識パート

## ✅ 概要

このリポジトリは大学3年生向けの情報科学演習教材をJupyter Notebook形式でまとめたものです。
ノートブックを順に実行しながら、画像の取り扱い、前処理、基本的な機械学習モデルの実装と評価を学びます。

## 📚 学習目標
- OpenCVやMatplotlibを使った画像の読み込み・表示・保存・前処理ができる
- 画像の基本演算（リサイズ・回転・色空間変換・フィルタ処理）を理解する
- 単純なパーセプトロン/多層パーセプトロン（MLP）を実装して学習・評価ができる
- PyTorchを用いたモデル定義・学習ループ・評価（MNISTなど）が行える

## 📁 ディレクトリ構成

```
project_root/
├── intro.md                # この概要ページ
├── requirements.txt        # pip/condaで使う依存リスト
├── data/                   # データセット（画像や学習データ）
├── notebooks/              # Jupyter Notebook（教材本体）
│   ├── images/              # ノートブック内で使う画像ファイル
│   ├── 00_Setup.ipynb
│   ├── 01_ImageProcessing.ipynb
│   └── 02_MachineLearning_Mnist.ipynb
└── util.py                 # 教材で共通利用するユーティリティ関数
```


## 📊 データセット

- ノートブックで利用する小規模な画像ファイルは `data/` に配置します。大きなデータは外部から取得するようにしてください。

## 🚀 学習の進め方

1. `notebooks/` 内のノートブックを順に実行してください。
2. まず `00_Setup.ipynb` で環境整備（conda環境やPyTorchインストール方法の説明）を確認します。
3. `01_ImageProcessing.ipynb` で画像処理の基礎と演習を行ってください。
4. `02_MachineLearning_Mnist.ipynb` で簡単なニューラルネットワーク（MLP/CNN）の実装・学習・評価を試してみましょう。

## 💻 使用技術

- Python 3.9+
- NumPy, pandas, matplotlib, seaborn
- OpenCV (cv2)
- PyTorch, torchvision
- scikit-learn

## 📝 実行環境

環境を再現するためには `requirements.txt`（または `environment.yml` を使う場合はそちら）を参照し、仮想環境を作成してください。ノートブック内に実行例を記載しています。