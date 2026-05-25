# SO-ARM101 構築手順まとめ

## 概要

本記事では、SO-ARM101 を組み立てて動作確認するまでの手順をまとめる。  
目的は、同じ構成を再現できるようにすることと、途中で詰まりやすい点を残すことである。

本記事では、以下の流れで進める。

1. 参考情報
2. 使用環境
3. SO-ARM101 の概要
4. アーム組み立て
5. LeRobot の環境構築
6. USB ポート確認
7. モーター ID とボーレートの設定
8. キャリブレーション
9. テレオペレーション確認
10. まとめ

## 1. 参考情報

- [公式ドキュメント](https://huggingface.co/docs/lerobot/so101#so-101)

- [ABEJA Tech Blog「【初心者が】ロボットアーム SO-101組み立てレポート ※ 2025/06/16更新【手順更新しました】](https://tech-blog.abeja.asia/entry/so101-assembly-report-v2-202506)

## 2. 使用環境

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12

## 3. SO-ARM101 の概要

SO-ARM101 は、Hugging Face の LeRobot でサポートされているオープンソースのロボットアームである。  
部品調達、3D プリント、組み立て、制御方法が公開されており、比較的再現しやすい構成になっている。  
また、リーダーアームとフォロワーアームの 2 台構成をとることで、人が動かした操作を別のアームに追従させるテレオペレーションが可能である。

この構成により、単なるロボットアームの操作だけでなく、模倣学習のための動作データ収集や、LeRobot を用いたロボット制御実験を行いやすい点が特徴である。  
本記事では、SO-ARM101 を対象として、環境構築から組み立て、キャリブレーション、テレオペレーション確認までの流れを整理する。

:::{figure} ./images/leader.png
:width: 55%
:align: center

図 1: リーダーアーム
:::

:::{figure} ./images/follower.png
:width: 55%
:align: center

図 2: フォロワーアーム
:::

## 4. アーム組み立て

```{warning}
モーター配置を必ず確認すること

モーターは見た目が非常によく似ているが、関節ごとにギア比が異なる。  
取り付け位置を誤ると正常に動作せず、大きな手戻りや再分解が必要になる。  
組み立て前に、各モーターの配置を必ず確認すること。
```

:::{figure} ./images/motor_model_codes.png
:width: 70%
:align: center

図 3: モーター型番の確認例
:::

リーダーアームで使用するモーターとギア比は、以下の通りである。

| リーダーアーム軸 | モーター | ギア比 |
| --- | ---: | ---: |
| Joint 1 | 1 | 1/191 |
| Joint 2 | 2 | 1/345 |
| Joint 3 | 3 | 1/191 |
| Joint 4 | 4 | 1/147 |
| Joint 5 | 5 | 1/147 |
| Joint 6 | 6 | 1/147 |

組み立て方法は、以下の公式ドキュメントを参照する。

- <https://huggingface.co/docs/lerobot/so101#so-101>

## 5. LeRobot の環境構築

[公式ドキュメント](https://huggingface.co/docs/lerobot/installation) に従って LeRobot を導入する。  
この節では、LeRobot の公式インストールガイドに従って環境を構築する。  
ここでは、公式で案内されている `uv` を用いた方法を採用する。

### 5.1 `uv` のインストール

まず、Python の仮想環境管理に用いる `uv` をインストールする。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、ターミナルを開き直す。

### 5.2 仮想環境の作成

次に、Python 3.12 をインストールし、仮想環境を作成する。

```bash
uv python install 3.12
uv venv --python 3.12
```

仮想環境を有効化する。

```bash
source .venv/bin/activate
```

### 5.3 `ffmpeg` のインストール

LeRobot では動画デコードのために `ffmpeg` が必要になるため、Ubuntu 側にインストールする。

```bash
sudo apt install ffmpeg
```

### 5.4 LeRobot 本体のインストール

LeRobot のリポジトリを取得し、編集可能モードでインストールする。

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv pip install -e .
```

### 5.5 SO-ARM101 用の追加依存関係

SO-101 では Feetech モーターを使用するため、追加で Feetech 用の依存関係を導入する。

```bash
uv pip install -e ".[feetech]"
```

### 5.6 ここまでの最小構成

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

sudo apt install ffmpeg

git clone https://github.com/huggingface/lerobot.git
cd lerobot

uv pip install -e .
uv pip install -e ".[feetech]"
```

### 5.7 エラーが出た場合

`cmake` やビルド関連のエラーが出た場合は、以下の依存関係を追加する。

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
```

## 6. USB ポート確認
```{warning}
接続する電源の電圧を必ず確認すること

STS3215 7.4Vモーター → 5V電源　←リーダーアーム
STS3215 12Vモーター → 12V電源　←フォロワーアーム
```
組み立てやキャリブレーションの前に、リーダーアームとフォロワーアームそれぞれの MotorBus が、どの USB ポートとして認識されるかを確認する。

MotorBus を PC に USB 接続し、電源も接続した状態で、以下のコマンドを実行する。

```bash
lerobot-find-port
```

Linux では、USB ポートへのアクセス権限が必要になる場合がある。  
その場合は、以下を実行する。

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

実行すると、以下のように使用可能なポートが表示される。

```text
Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']
Remove the usb cable from your MotorsBus and press Enter when done.
```

指示に従い、確認したい MotorBus の USB ケーブルを一度抜き、Enter キーを押す。  
その後、以下のように該当するポートが表示される。

```text
The port of this MotorsBus is /dev/ttyACM1
Reconnect the USB cable.
```

この例では、該当する MotorBus のポートは `/dev/ttyACM1` である。

## 7. モーター ID とボーレートの設定

各モーターには、バス上で識別するための一意な ID を割り当てる必要がある。  
新品のモーターは、初期状態では同じ ID になっている場合があるため、そのままでは正しく通信できない。

また、コントローラと各モーターが通信するためには、ボーレートも一致している必要がある。  
この設定はモーター内部の EEPROM に保存されるため、基本的には最初に一度だけ行えばよい。

```{warning}
モーターは必ず 1 個ずつ接続して設定する

ID 設定時は、対象のモーターだけをコントローラボードに接続する。  
複数のモーターを同時に接続した状態で設定すると、意図しないモーターに ID が書き込まれる可能性がある。
```

### 7.1 フォロワーアーム

フォロワーアームのコントローラボードを PC に USB 接続し、電源も接続する。  
次に、前の手順で確認したポートを指定して、以下のコマンドを実行する。

```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1
```

実行すると、以下のような指示が表示される。

```text
Connect the controller board to the 'gripper' motor only and press enter.
```

指示に従い、`gripper` モーターだけをコントローラボードに接続し、Enter キーを押す。  
このとき、対象のモーターが他のモーターとデイジーチェーン接続されていないことを確認する。

設定が完了すると、以下のように表示される。

```text
'gripper' motor id set to 6
```

続いて、次のモーターを接続するように指示される。

```text
Connect the controller board to the 'wrist_roll' motor only and press enter.
```

同様に、表示された指示に従い、各モーターを 1 個ずつ接続して ID とボーレートを設定する。

```{warning}
Enter を押す前に接続を確認すること

ID 設定時は、対象のモーターだけをコントローラボードに接続する。  
複数のモーターを同時に接続すると、意図しないモーターに ID が書き込まれる可能性がある。  
また、作業中に電源ケーブルが抜けていないかも確認してから Enter キーを押す。
```

すべての設定が完了したら、各モーターを通常の配線順に接続する。  
最後に、最初のモーターである `shoulder_pan` からコントローラボードへ接続する。

### 7.2 リーダーアーム

リーダーアームも同様に、前の手順で確認したポートを指定して設定する。

```bash
lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0
```

表示される指示に従い、各モーターを 1 個ずつ接続して ID とボーレートを設定する。

### 7.3 ポート名について

上記の `/dev/ttyACM0` や `/dev/ttyACM1` は一例である。  
実際には、`lerobot-find-port` で確認したポート名に置き換える。

```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port={follower_port}
```

```bash
lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port={leader_port}
```

## 8. キャリブレーション

組み立て後、リーダーアームとフォロワーアームのキャリブレーションを行う。

### 8.1 リーダーアーム

```bash
python -m lerobot.calibrate \
  --teleop.type=so101_leader \
  --teleop.port={leader_port} \
  --teleop.id={leader_id}      #任意の名前を付ける
```

### 8.2 フォロワーアーム

```bash
python -m lerobot.calibrate \
  --robot.type=so101_follower \
  --robot.port={follower_port} \
  --robot.id={follower_id}　　　#任意の名前を付ける
```

### 実施時の注意

- 各関節の可動域を十分に動かす
- 途中で引っかかりがないか確認する
- キャリブレーション後の状態を記録する
- `wrist_roll` は除外されている

### 8.3 `wrist_roll` がキャリブレーション対象に出てこない場合

標準のキャリブレーションを実行した際、`wrist_roll` の欄が表示されない場合がある。  
この状態では、他の関節はキャリブレーションできても、リーダーアームとフォロワーアームの `wrist_roll` の中心位置がずれたままになることがある。

この問題に対して、本環境では `wrist_roll` 専用のキャリブレーション用 Python スクリプトを作成した。  
このスクリプトでは、リーダーアームとフォロワーアームの `wrist_roll` をそれぞれ安全な最小位置・最大位置へ動かし、その raw position を読み取る。  
その後、両者の中心位置の差から follower 側の `homing_offset` を補正する。

```{warning}
このスクリプトは follower 側のキャリブレーション JSON を直接更新する。  
実行前に、`LEADER_PORT`、`FOLLOWER_PORT`、`LEADER_JSON`、`FOLLOWER_JSON` が自分の環境に合っているか必ず確認すること。  
更新前には `.bak` ファイルが作成されるが、作業前に元の JSON を別途控えておくと安全である。
```

スクリプトは本記事と同じディレクトリの `scripts/calibrate_wrist_roll.py` として保存している。

```python
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.motors.feetech.feetech import FeetechMotorsBus, Motor
from lerobot.motors.motors_bus import MotorNormMode

LEADER_PORT = "/dev/ttyACM0"
FOLLOWER_PORT = "/dev/ttyACM1"

LEADER_JSON = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader_arm.json"
FOLLOWER_JSON = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json"

MOTOR_NAME = "wrist_roll"
MOTOR_ID = 5
MODEL = "sts3215"

SAMPLE_DELAY = 0.05
AVERAGE_SAMPLES = 10
STEP_LIMIT = 800


def wrap(x: int) -> int:
    return ((x + 2048) % 4096) - 2048


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def read_json(p):
    with open(p, "r") as f:
        return json.load(f)


def write_json(p, d):
    with open(p, "w") as f:
        json.dump(d, f, indent=4)
        f.write("\n")


def backup(p):
    b = p.with_suffix(".bak")
    b.write_text(p.read_text())
    return b


def make_bus(port):
    return FeetechMotorsBus(
        port=port,
        motors={MOTOR_NAME: Motor(MOTOR_ID, MODEL, MotorNormMode.DEGREES)},
    )


def read_raw(bus):
    return int(bus.read("Present_Position", MOTOR_NAME, normalize=False))


def avg(bus):
    vals = []
    for _ in range(AVERAGE_SAMPLES):
        vals.append(read_raw(bus))
        time.sleep(SAMPLE_DELAY)
    return round(sum(vals) / len(vals))


def measure(bus, label):
    print(f"\n{label} に動かして Enter")
    input("Enter: ")
    v = avg(bus)
    print(f"{label} = {v}")
    return v


def main():
    print("min/maxから中心＋スケール推定します")
    print("leader / follower を安全な最小・最大位置へ動かしてください")

    follower_data = read_json(FOLLOWER_JSON)
    old_offset = int(follower_data[MOTOR_NAME]["homing_offset"])

    lb = make_bus(LEADER_PORT)
    fb = make_bus(FOLLOWER_PORT)

    lb.connect()
    fb.connect()

    try:
        print("\n=== leader ===")
        Lmin = measure(lb, "leader 最小")
        Lmax = measure(lb, "leader 最大")

        print("\n=== follower ===")
        Fmin = measure(fb, "follower 最小")
        Fmax = measure(fb, "follower 最大")

    finally:
        lb.disconnect()
        fb.disconnect()

    Lc = (Lmin + Lmax) / 2
    Fc = (Fmin + Fmax) / 2

    Lspan = Lmax - Lmin
    Fspan = Fmax - Fmin

    if Lspan == 0 or Fspan == 0:
        raise ValueError("spanが0")

    scale = Fspan / Lspan

    delta_center = wrap(round(Lc - Fc))
    step = clamp(delta_center, -STEP_LIMIT, STEP_LIMIT)
    new_offset = wrap(old_offset + step)

    print("\n=== 結果 ===")
    print(f"Lmin, Lmax = {Lmin}, {Lmax}")
    print(f"Fmin, Fmax = {Fmin}, {Fmax}")
    print(f"Lcenter    = {Lc:.2f}")
    print(f"Fcenter    = {Fc:.2f}")
    print(f"Lspan      = {Lspan}")
    print(f"Fspan      = {Fspan}")
    print(f"scale(F/L) = {scale:.4f}")

    print("\n--- offset補正 ---")
    print(f"delta_center = {delta_center}")
    print(f"old_offset   = {old_offset}")
    print(f"new_offset   = {new_offset}")

    print("\n※ scaleが1からズレている場合")
    print("→ 機械誤差 or サーボ差")
    print("→ offsetだけでは完全一致しない")

    ans = input("\n更新する？ [y/N]: ").strip().lower()
    if ans != "y":
        print("中止")
        return

    b = backup(FOLLOWER_JSON)
    follower_data[MOTOR_NAME]["homing_offset"] = new_offset
    write_json(FOLLOWER_JSON, follower_data)

    print("\n更新完了")
    print("backup:", b)


if __name__ == "__main__":
    main()
```

実行例は以下の通りである。

```bash
python scripts/calibrate_wrist_roll.py
```

実行すると、`leader 最小`、`leader 最大`、`follower 最小`、`follower 最大` の順に手動で位置を合わせるよう求められる。  
各位置で Enter キーを押すと、現在位置を複数回読み取って平均し、中心位置とスケールを表示する。

このスクリプトで更新するのは follower 側の `wrist_roll` の `homing_offset` である。  
`scale(F/L)` が 1 から大きくずれている場合、中心位置の補正だけでは完全には一致しない。  
その場合は、機械的な組み付け、サーボの個体差、可動域の取り方も確認する。

## 9. テレオペレーション確認

最後に、リーダーアームの動きがフォロワーアームに追従するか確認する。

```bash
python -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port={follower_port} \
  --robot.id={follower_id} \
  --teleop.type=so101_leader \
  --teleop.port={leader_port} \
  --teleop.id={leader_id}
```

`{follower_id}` と `{leader_id}` には、キャリブレーション時に指定した名前を入れる。

### 確認項目

- [ ] 各関節が追従する
- [ ] グリッパーが追従する
- [ ] 動きが逆転していない
- [ ] 異音がない
- [ ] 通信が途中で切れない

### 動作例

以下は、リーダーアームの動きにフォロワーアームが追従している様子である。

<video controls muted playsinline style="width: 100%; max-width: 760px;">
  <source src="../../teleoperate_noaudio.mp4" type="video/mp4">
  お使いのブラウザは video タグに対応していません。
</video>

## 10. まとめ

本記事では、SO-ARM101 の概要から、LeRobot の環境構築、USB ポート確認、モーター ID 設定、アーム組み立て、キャリブレーション、テレオペレーション確認までの流れを整理した。

SO-ARM101 は、LeRobot を用いた模倣学習やロボット制御実験を比較的低コストで再現しやすい構成になっている一方で、初回構築時にはモーター ID 設定や USB ポート確認、キャリブレーションなど、詰まりやすいポイントも多い。

特に、モーターは見た目が似ていても関節ごとにギア比が異なるため、配置を誤ると大きな手戻りにつながる。  
また、モーター ID 設定時は、対象モーターのみを接続して作業することが重要である。

本記事が、SO-ARM101 を用いた環境構築や LeRobot の導入を行う際の参考になれば幸いである。
