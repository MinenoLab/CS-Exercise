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
