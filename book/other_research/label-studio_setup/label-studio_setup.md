# Label Studio 環境構築・使用方法まとめ

## 1. 概要

Label Studio は，画像・動画・テキスト・音声・時系列データなどに対してアノテーションを行うためのツールである．本ページでは，画像に対して BBox(Bounding Box)・OBB(Oriented Bounding Box) などのアノテーションを行い，YOLO の学習用データとして利用する方法について解説する．また，作業を効率化する方法として，YOLO ML Backend を接続し，学習済みモデルによる予測結果を Label Studio 上に表示して修正しながらアノテーションを進める方法についても説明する．

### 1.1 Label Studio で扱える主なアノテーション形式

Label Studio では，Labeling Interface の設定を変更することで，様々な形式のアノテーションを行うことができる．

#### 画像アノテーション

| 形式                   | 主な用途                   | Label Studio の主なタグ |
| -------------------- | ---------------------- | ------------------ |
| BBox                 | 物体検出                   | `RectangleLabels`  |
| OBB                  | 回転矩形による物体検出            | `RectangleLabels`  |
| Polygon              | インスタンスセグメンテーション        | `PolygonLabels`    |
| Brush / Mask         | セマンティックセグメンテーション，マスク作成 | `BrushLabels`      |
| Bitmask              | ビットマスク形式の領域アノテーション     | `BitmaskLabels`    |
| Keypoint             | 姿勢推定，部位点アノテーション        | `KeyPointLabels`   |
| Ellipse              | 楕円形の領域指定               | `EllipseLabels`    |
| Vector               | ベクトル形式のアノテーション         | `VectorLabels`     |
| Image Classification | 画像単位の分類                | `Choices`          |

#### 動画アノテーション

| 形式              | 主な用途                      | Label Studio の主なタグ |
| --------------- | ------------------------- | ------------------ |
| Video Rectangle | 動画内の物体追跡，フレームごとの矩形アノテーション | `VideoRectangle`   |
| Timeline Labels | 動画内の時間区間ラベル付け             | `TimelineLabels`   |

#### テキスト・HTML・段落アノテーション

| 形式                 | 主な用途               | Label Studio の主なタグ |
| ------------------ | ------------------ | ------------------ |
| Text Labeling      | 固有表現抽出，テキスト分類      | `Labels`           |
| HyperText Labeling | HTML 文書中の範囲アノテーション | `HyperTextLabels`  |
| Paragraph Labeling | 段落単位のラベル付け         | `ParagraphLabels`  |
| TextArea           | 自由記述，文字起こし，説明文入力   | `TextArea`         |
| Choices            | 単一選択・複数選択の分類       | `Choices`          |
| Taxonomy           | 階層的な分類             | `Taxonomy`         |
| Rating             | 評価値の入力             | `Rating`           |
| Ranker             | 順位付け               | `Ranker`           |
| Pairwise           | 2つのデータの比較          | `Pairwise`         |

#### 音声アノテーション

| 形式                     | 主な用途        | Label Studio の主なタグ |
| ---------------------- | ----------- | ------------------ |
| Audio Transcription    | 音声の文字起こし    | `TextArea`         |
| Audio Classification   | 音声単位の分類     | `Choices`          |
| Audio Segment Labeling | 音声区間へのラベル付け | `Labels`           |

#### 時系列データアノテーション

| 形式                   | 主な用途                   | Label Studio の主なタグ |
| -------------------- | ---------------------- | ------------------ |
| Time Series Labeling | センサーデータなどの時系列範囲へのラベル付け | `TimeSeriesLabels` |

#### その他・補助的なアノテーション

| 形式         | 主な用途         | Label Studio の主なタグ      |
| ---------- | ------------ | ----------------------- |
| Number     | 数値入力         | `Number`                |
| DateTime   | 日時入力         | `DateTime`              |
| Relation   | 領域・ラベル間の関係付け | `Relation`, `Relations` |
| Magic Wand | 画像領域の半自動選択   | `Magicwand`             |

---

## 2. 使用環境

本ページで使用した環境を以下に示す．

* OS: Ubuntu 24.04.3 LTS
* GPU: NVIDIA RTX PRO 6000 Blackwell 
* CUDA バージョン: 12.9
* Python バージョン: 3.13.9
* pip バージョン: 25.2
* Label Studio バージョン: 1.23.0
* Label Studio ML Backend の Githubリポジトリ: https://github.com/HumanSignal/label-studio-ml-backend
* Docker バージョン: 29.3.0
* Docker Compose バージョン: v2.39.1

---

## 3. 環境構築手順

### 3.1 作業ディレクトリの作成

まず，VS Code で DNN サーバーに SSH 接続する．

※コマンドラインで SSH 接続する場合は，以下のコマンドを実行する．

```bash
ssh ユーザー名@サーバーのIPアドレス
```

接続後，作業用ディレクトリを作成する．

```bash
mkdir -p ~/label_studio
cd ~/label_studio
```

---

### 3.2 Label Studio の環境作成

Python の仮想環境を作成する．

```bash
python3 -m venv .venv
```

仮想環境を有効化する．

```bash
source .venv/bin/activate
```

pip を更新する．

```bash
pip install --upgrade pip
```

Label Studio をインストールする．

```bash
pip install label-studio
```

インストールできたか確認する．

```bash
label-studio --version
```

---

### 3.3 YOLO ML Backend の取得

YOLO ML Backend 用のリポジトリを `git clone` する．

```bash
cd ~
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
```

クローンしたディレクトリへ移動する．

```bash
cd ~/label-studio-ml-backend/label_studio_ml/examples/yolo
```

必要に応じて，使用する YOLO モデルの重みファイルを ML Backend 側の models ディレクトリにコピーする．

```bash
cp モデルファイルがあるパス/best.pt models/モデル名.pt
```

---

### 3.4 Dockerfile の修正

YOLO ML Backend のディレクトリに移動し，Dockerfile を編集する．

```bash
cd ~/label-studio-ml-backend/label_studio_ml/examples/yolo
```

Docker イメージ作成時に conda update が実行され，処理が終わらない場合があるため，Dockerfile 内に以下の行がある場合は，行頭に # を付けてコメントアウトする．

修正前:
```dockerfile
RUN conda update conda -y
```

修正後:

```dockerfile
# RUN conda update conda -y
```

---

### 3.5 Label Studio の API キーの取得

YOLO ML Backend から Label Studio に接続するためには，Label Studio の API キーが必要である．
まず Label Studio を起動し，ブラウザからアクセスする．

```bash
cd ~/label_studio
source .venv/bin/activate
label-studio start
```

手元の PC のブラウザで以下にアクセスする．

```text
http://localhost:8080
```

Label Studio にログイン後，以下の手順で API キーを取得する．

1. 画面右上のユーザーアイコンをクリックし，`Account & Settings` を開く

![getting_APIkey_step1](./images/getting_APIkey_step1.png)

2. `Personal Access Token` タブの `Create New Token` を開く

![getting_APIkey_step2](./images/getting_APIkey_step2.png)

3. 表示された API キーをコピー

![getting_APIkey_step3](./images/getting_APIkey_step3.png)

取得した API キーは，次の `.env` ファイルの設定で使用する．

---

### 3.6 `.env` ファイルの設定

YOLO ML Backend のディレクトリに移動する．

```bash
cd ~/label-studio-ml-backend/label_studio_ml/examples/yolo
```

`.env` ファイルを作成し，以下のように Label Studio の URL と API キーを記述する．

```env
LABEL_STUDIO_URL=http://<サーバーのIPアドレス>:8080
LABEL_STUDIO_API_KEY=<APIキー>
```

例:

```env
LABEL_STUDIO_URL=http://xxx.xxx.xxx.xxx:8080
LABEL_STUDIO_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

ここで指定する `LABEL_STUDIO_URL` は，YOLO ML Backend から見た Label Studio の URL である．
Docker コンテナ内から接続するため，`localhost` ではなく，DNN サーバーの IP アドレスを指定する．

---

### 3.7 docker-compose.yml の修正

YOLO ML Backend の `docker-compose.yml` の編集を行う．

まず，`build` の `args` に以下の設定がある場合は該当の行を削除する．

```yaml
args:
  TEST_ENV: ${TEST_ENV}
```

次に，environment の設定を修正する．
Label Studio の URL と API キーを .env ファイルから読み込むようにするため，以下のように変更する．

修正前:

```yaml
- LABEL_STUDIO_HOST=${LABEL_STUDIO_HOST:-http://host.docker.internal:8080}
- LABEL_STUDIO_API_KEY=${LABEL_STUDIO_API_KEY}
```

修正後:

```yaml
  - LABEL_STUDIO_URL=${LABEL_STUDIO_URL}
  - LABEL_STUDIO_API_KEY=${LABEL_STUDIO_API_KEY}
```

これにより，`.env` ファイルに記述した `LABEL_STUDIO_URL` と `LABEL_STUDIO_API_KEY` が，YOLO ML Backend の Docker コンテナ内で使用される．

---
### 3.8 YOLO ML Backend の起動

YOLO ML Backend のディレクトリへ移動する．

```bash
cd ~/label-studio-ml-backend/label_studio_ml/examples/yolo
```

Docker Compose で起動する．バックグラウンドで起動したい場合は，-d オプションを付ける．

```bash
docker compose up
```
---

### 3.9 Label Studio の起動

別のターミナルを開き，Label Studio の作業ディレクトリへ移動する．

```bash
cd ~/label_studio
```

仮想環境を有効化する．

```bash
source .venv/bin/activate
```

Label Studio を起動する．

```bash
label-studio start
```

---

### 3.10 SSH ポートフォワーディング

手元の PC から DNN サーバー上の Label Studio にアクセスするため，ポートフォワーディングを行う．
Label Studio では `8080`，YOLO ML Backend では `9090` を使用するため，両方のポートを転送する．

#### VS Code Remote SSH を使用する場合(推奨)

VS Code の Remote SSH で DNN サーバーに接続している場合は，VS Code のポート転送機能を使用できる．

1. VS Code で DNN サーバーに Remote SSH 接続する
2. 画面下部の `PORTS` タブを開く
3. `Forward a Port` をクリックする
4. `8080` を入力して転送する
5. 同様に `9090` も転送する

転送後，`PORTS` タブに以下のように表示されていればよい．

| Port   | 用途              |
| ------ | --------------- |
| `8080` | Label Studio    |
| `9090` | YOLO ML Backend |
---

#### PowerShell またはターミナルを使用する場合

手元の PC の PowerShell またはターミナルで以下を実行する．

```bash
ssh -L 8080:localhost:8080 -L 9090:localhost:9090 ユーザー名@サーバーのIPアドレス
```
---


### 3.11 ブラウザでアクセス

手元の PC のブラウザで以下にアクセスする．

```text
http://localhost:8080
```

Label Studio のログイン画面が表示されれば成功．

![label_studio_login](./images/label_studio_login.png)
---

## 4. Label Studio の基本的な使い方

### 4.1 プロジェクト作成

1. Label Studio にログインする

![creating_project_step1](./images/creating_project_step1.png)

2. `Create Project` をクリックする

![creating_project_step2](./images/creating_project_step2.png)

3. プロジェクト名を入力する

![creating_project_step3](./images/creating_project_step3.png)

4. アノテーション対象の画像をアップロードし，`Save`を押す

![creating_project_step4](./images/creating_project_step4.png)

5. Labeling Interface を設定し，`Sava` を押す

![creating_project_step5](./images/creating_project_step5.png)

---

### 4.2 Labeling Interface の設定

BBox や OBB など，アノテーション形式に応じて Labeling Interface を設定する．

#### BBox の例

```xml
<View>
<Image name="image" value="$image"/>
  <RectangleLabels
    name="label"
    toName="image"
    model_path="best.pt"
    model_score_threshold="0.5"
    opacity="0">

    <Label value="ace" background="red"/>
</RectangleLabels>
</View>
 
```

#### OBB の例

```xml
<View>
<Image name="image" value="$image"/>
  <RectangleLabels
    name="label"
    toName="image"
    canRotate="true"
    model_path="best.pt"
    model_score_threshold="0.5"
    model_obb="true"
    opacity="0">

    <Label value="ace" background="red"/>
</RectangleLabels>
</View>
 
```

---

### 4.3 ML Backend の接続

YOLO ML Backend を使う場合，Label Studio のプロジェクト設定から ML Backend を接続する．

1. プロジェクトを開き，`Settings` をクリックする

![connecting_MLBackend_step1](./images/connecting_MLBackend_step1.png)

2. `Model` タブを開き，`Connect Model`をクリックする

![connecting_MLBackend_step2](./images/connecting_MLBackend_step2.png)

3. `Name` に任意の名前，`Backend URL` に以下の URL を入力し，`Validate and Save` をクリックする

```text
http://localhost:9090
```

![connecting_MLBackend_step3](./images/connecting_MLBackend_step3.png)


4. 作成したモデルの右上の `･･･` をクリックし，`Send Test Request` → `Send Request` で接続テストを行う

![connecting_MLBackend_step4](./images/connecting_MLBackend_step4.png)

![connecting_MLBackend_step5](./images/connecting_MLBackend_step5.png)

5. 正常に接続できれば `Save` で保存する

![connecting_MLBackend_step6](./images/connecting_MLBackend_step6.png)

![connecting_MLBackend_step7](./images/connecting_MLBackend_step7.png)

6. `Annotation` タブを開き，`Use predictions to prelabel tasks` を有効にして，使用するモデルを選択し，`Save` をクリックする

![connecting_MLBackend_step8](./images/connecting_MLBackend_step8.png)

この設定を有効にすることで，ML Backend の予測結果がアノテーション画面に事前表示される．

---

## 5. アノテーション作業の流れ

### 5.1 画像を開く

プロジェクト内のタスクを開くと，画像が表示される．

![annotation_step1](./images/annotation_step1.png)

---

### 5.2 BBox / OBB を作成する

画像中の対象物を囲むように矩形を作成する．

#### BBox の場合

* 対象を水平な矩形で囲む

![annotation_BBox](./images/annotation_BBox.png)

#### OBB の場合

* 対象の向きに合わせて矩形を回転させる
* 通常の BBox よりも，個体の向きを反映しやすい

![annotation_OBB](./images/annotation_OBB.png)

---

### 5.3 Submit する

1 枚の画像のアノテーションが終わったら `Submit` を押す．

`Submit` を押すことで，現在のアノテーション結果が保存される．

![annotation_step3](./images/annotation_step3.png)

---

### 5.4 予測結果を修正する

YOLO ML Backend を接続している場合，学習済みモデルによる予測結果が Label Studio 上に表示される．  
予測された BBox / OBB が正しい場合はそのまま利用し，位置や大きさがずれている場合は手動で修正する．  
不要な予測がある場合は削除し，検出されていない対象がある場合は手動で追加する．

↓予測結果がずれている例

![misaligned_BBox](./images/misaligned_BBox.png)

BBoxの位置を手動で修正

![modified_BBox](./images/modified_BBox.png)

---

## 6. エクスポート方法

アノテーションが完了したら，学習用データとしてエクスポートする．

### 6.1 エクスポート手順

1. プロジェクトを開き，`Export` をクリックする

![export_step1](./images/export_step1.png)

2. 出力形式を選択し，`Export` をクリックする

Label Studio における主な出力形式を以下に示す．ただし，選択できる出力形式は，使用しているアノテーション形式や Labeling Interface の設定によって変わる場合がある．

| 出力形式                          | 主な用途                           | 対応しやすいアノテーション               |
| ----------------------------- | ------------------------------ | --------------------------- |
| JSON                          | Label Studio の情報を含む標準形式        | すべてのプロジェクト                  |
| JSON_MIN                      | Label Studio 固有の情報を減らした簡易 JSON | すべてのプロジェクト                  |
| CSV                           | 表形式での確認・集計                     | すべてのプロジェクト                  |
| TSV                           | タブ区切りの表形式                      | すべてのプロジェクト                  |
| YOLO                          | YOLO 系モデルの学習用                  | BBox，Keypoint，OBB など        |
| COCO                          | 物体検出，セグメンテーション，Keypoint など     | BBox，Polygon，Brush，Keypoint |
| Pascal VOC XML                | 物体検出用の XML 形式                  | BBox                        |
| Brush labels to NumPy and PNG | マスク画像・NumPy 配列として出力            | Brush / Mask                |
| ASR_MANIFEST                  | 音声認識用データ形式                     | 音声文字起こし                     |
| CoNLL2003                     | 固有表現抽出などの NLP 用形式              | テキストラベリング                   |
| spaCy                         | spaCy 用の NLP データ変換             | テキストラベリング                   |
---

## 7. 終了方法

### 7.1 Label Studio の停止

Label Studio を起動しているターミナルで `Ctrl + C` を押す．

---

### 7.2 YOLO ML Backend の停止

`docker compose up` で起動している場合は，ターミナルで `Ctrl + C` を押す．

バックグラウンド起動している場合は，以下を実行する．

```bash
cd ~/label-studio-ml-backend/label_studio_ml/examples/yolo
docker compose down
```

---

## 8. 2回目以降の起動方法

一度環境構築が完了している場合，2回目以降は以下の手順で Label Studio と YOLO ML Backend を起動する．

---

### 8.1 YOLO ML Backend の起動

YOLO ML Backend のディレクトリへ移動する．

```bash
cd ~/label-studio-ml-backend/label_studio_ml/examples/yolo
```

Docker Compose で YOLO ML Backend を起動する．バックグラウンドで起動したい場合は，-d オプションを付ける．

```bash
docker compose up
```
---

### 8.2 Label Studio の起動

別のターミナルを開き，Label Studio の作業ディレクトリへ移動する．

```bash
cd ~/label_studio
```

仮想環境を有効化する．

```bash
source .venv/bin/activate
```

Label Studio を起動する．

```bash
label-studio start
```

---

### 8.3 SSH ポートフォワーディング


#### 8.3.1 VS Code Remote SSH を使用する場合

`PORTS` タブから `8080` と `9090` を転送する．

#### 8.3.2 PowerShell またはターミナルを使用する場合
手元の PC から DNN サーバー上の Label Studio にアクセスするため，ポートフォワーディングを行う．手元の PC で以下を実行する．

```bash
ssh -L 8080:localhost:8080 -L 9090:localhost:9090 ユーザー名@サーバーのIPアドレス
```

---

### 8.4 ブラウザでアクセス

手元の PC のブラウザで以下にアクセスする．

```text
http://localhost:8080
```

Label Studio の画面が表示されれば起動完了である．

---
