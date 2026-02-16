# IoTパート

## ✅ 概要

このパートは大学3年生向けの情報科学演習教材で、IoT（Internet of Things）システムの構築を実践的に学びます。
Raspberry Pi、PiNode3、Spresenseカメラなどを用いて、センサーデータの収集からデータベースへの送信まで、実際のIoTシステムの一連のプロセスを体験します。

## 📚 学習目標

- Raspberry Piの基本的なセットアップおよびリモートアクセス方法を習得する
- IoTシステム（PiNode3）を構築し、センサーデータの収集とデータベースへの保存を実装する
- Spresenseカメラの設定と動作確認を行い、画像データの取得方法を学ぶ
- IoTシステム全体の構成と、エッジデバイスからクラウドへのデータフローを理解する
- 実際のハードウェアを使って、IoTシステムの物理的な設置と運用を体験する

## 📁 ディレクトリ構成

```
part4_iot/
├── intro.md                              # この概要ページ
├── data/                                 # 収集データ格納ディレクトリ
└── notebooks/                            # 学習用Jupyter Notebook
    ├── 01_raspi_setup.ipynb              # Raspberry Piのセットアップ
    ├── 02_pinode_setup.ipynb             # PiNode3システムの構築
    ├── 03_spresense_setup.ipynb          # Spresenseカメラのセットアップ
    └── 04_iot_physical_install.md        # IoTデバイスの物理的な設置手順
```

## 🔧 使用ハードウェア

- **Raspberry Pi**: メインコンピューティングユニット（Debian GNU/Linux 12 (bookworm)推奨）
- **PiNode3**: Raspberry Piベースのセンサーネットワークシステム
- **Spresense Camera**: Sony製の小型高性能カメラモジュール
- **各種センサー**: 温度、湿度、照度などの環境センサー

## 📊 データセット

本パートでは、以下のようなデータを収集・蓄積します：

- **センサーデータ**: 温度、湿度、照度などの時系列データ
- **画像データ**: Spresenseカメラで取得した画像
- **ログデータ**: システムの動作ログやエラーログ

データは`data/`ディレクトリに保存されるか、リモートデータベースに送信されます。

## 🚀 学習の進め方

`notebooks`ディレクトリ内のJupyter Notebookを以下の順序で実行し、内容を理解しながら進めてください。
各Notebookには、実機を使った実習が含まれています。

### Step 1: Raspberry Piのセットアップ

Raspberry Pi OSのインストールから始め、Wi-Fi設定、SSH接続、OpenCVによるカメラ制御まで、Raspberry Piを使用するための基本的な環境構築を学びます。

**主な内容:**
- Raspberry Pi Imagerを使ったOSのインストール
- ネットワーク設定（Wi-Fi、固定IPアドレス）
- SSH/VNC/sambaによるリモートアクセス設定
- OpenCVのインストールとカメラ動作確認

### Step 2: PiNode3システムの構築

Raspberry Piを用いて、実際のIoTシステム（PiNode3）を構築します。センサーデータの収集、ローカルまたはリモートデータベースへの保存、自動起動設定などを実装します。

**主な内容:**
- PiNode3リポジトリのクローンとインストール
- データ収集プログラムの設定と動作確認
- ローカルDB/リモートDBへのデータ送信設定
- systemdサービスによる自動起動設定

### Step 3: Spresenseカメラのセットアップ

Sony製Spresenseカメラの開発環境を構築し、Arduino IDEを使ってプログラムを書き込みます。PCからのアクセス方法やPiNodeとの連携方法を学びます。

**主な内容:**
- Arduino IDEのインストールと設定
- SPRESENSE Arduino board packageのインストール
- USBドライバのインストール
- カメラプログラムの書き込みと動作確認
- PiNodeとの連携設定

### Step 4: IoTデバイスの物理的な設置

実際にIoTデバイスを設置する際の注意点、ケーブル配線、電源管理、防水・防塵対策などの実践的な知識を習得します。

## 💻 使用技術

### ソフトウェア
- **Python**: IoTシステムのプログラミング言語
- **OpenCV**: カメラ画像の処理
- **InfluxDB**: 時系列データベース（オプション）
- **systemd**: Linuxサービス管理
- **Arduino IDE**: Spresenseプログラミング環境

### ハードウェア制御
- **GPIO制御**: センサーとの通信
- **I2C/SPI通信**: デバイス間通信
- **USB通信**: カメラとの接続

### ネットワーク
- **SSH**: リモートアクセス
- **HTTP/HTTPS**: データ送信

## 📝 実行環境

### Raspberry Pi側
- Raspberry Pi OS (64-bit) Debian GNU/Linux 12 (bookworm)
- Python 3.x
- OpenCV, InfluxDB client等のライブラリ

### PC側
- Windows/Mac/Linux
- Arduino IDE
- SSH/VNCクライアント
- テキストエディタ（VS Code推奨）

### ネットワーク要件
- Wi-Fi環境またはEthernet接続
- インターネット接続（パッケージインストールおよびリモートDB利用時）

## 🔗 関連リポジトリ

- [PiNode3](https://github.com/MinenoLab/PiNode3.git) - センサーネットワークシステム
- [PiNode3-SPRESENSE](https://github.com/MinenoLab/PiNode3-SPRESENSE.git) - Spresenseカメラ連携

## 注意事項

- 実機を使用するため、ハードウェアの取り扱いには十分注意してください
- 電源の接続/切断は慎重に行ってください
- ネットワーク設定は各自の環境に合わせて調整してください
- センサーやカメラの設置場所は、目的に応じて適切に選定してください
