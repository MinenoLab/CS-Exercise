from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

try:
    from experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, sanitize_name, write_json
except ModuleNotFoundError:
    from scripts.experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, sanitize_name, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = PROJECT_ROOT / "datasets" / "custom_yolo" / "data.yaml"
DEFAULT_EXPERIMENT_NAME = "custom_object_ep100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultralytics YOLOで物体検出モデルをファインチューニングします。")
    parser.add_argument("--model", default="yolo11n.pt", help="初期重み。デフォルト: yolo11n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="data.yamlのパス。")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50, help="early stoppingの待機epoch数。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU番号。CPUの場合はcpu。")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR), help="実験単位の保存先ルート。")
    parser.add_argument("--experiment", default="", help="実験名。未指定の場合は--nameを使います。")
    parser.add_argument("--name", default=DEFAULT_EXPERIMENT_NAME, help="実験名。")
    parser.add_argument("--train-dir-name", default="train", help="実験ディレクトリ内の学習結果ディレクトリ名。")
    parser.add_argument("--exist-ok", action="store_true", help="同じtrainディレクトリがあっても再利用します。")
    parser.add_argument("--keep-val-confusion", action="store_true", help="学習時に出るval混同行列を残します。")
    return parser.parse_args()


def print_cuda_info() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"[INFO] torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"[INFO] CUDA GPU数: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"[INFO] GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("[WARN] CUDA GPUが利用できません。device=cpuでの実行を検討してください。")


def remove_val_confusion_matrices(train_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        path = train_dir / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def is_empty_skeleton_dir(directory: Path) -> bool:
    if not directory.exists():
        return True
    return all(path.name == ".gitkeep" for path in directory.iterdir())


def load_yolo_model(model_name: str) -> YOLO:
    candidates = [model_name]
    if model_name == "yolo11n.pt":
        candidates.append("yolov8n.pt")
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            print(f"[INFO] モデルを読み込みます: {candidate}")
            return YOLO(candidate)
        except Exception as exc:
            print(f"[WARN] モデル読み込み失敗: {candidate}: {exc}")
            last_error = exc
    raise RuntimeError(f"YOLOモデルを読み込めませんでした: {last_error}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"[ERROR] data.yamlが見つかりません: {data_path}")

    experiment_name = sanitize_name(args.experiment or args.name)
    exp_dir = experiment_dir(Path(args.experiments_dir), experiment_name)
    ensure_experiment_dirs(exp_dir)
    train_dir = exp_dir / args.train_dir_name
    train_exist_ok = args.exist_ok or is_empty_skeleton_dir(train_dir)

    print_cuda_info()
    model = load_yolo_model(args.model)
    print(f"[INFO] 実験名: {experiment_name}")
    print(f"[INFO] 実験ディレクトリ: {exp_dir}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(exp_dir),
        name=args.train_dir_name,
        exist_ok=train_exist_ok,
    )

    save_dir = Path(getattr(results, "save_dir", train_dir))
    best_path = save_dir / "weights" / "best.pt"
    removed_val_confusion = [] if args.keep_val_confusion else remove_val_confusion_matrices(save_dir)
    write_json(
        exp_dir / "metadata" / "train_metadata.json",
        {
            "experiment": experiment_name,
            "data": str(data_path),
            "initial_model": args.model,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "train_dir": str(save_dir),
            "best_pt": str(best_path),
            "val_confusion_policy": "kept" if args.keep_val_confusion else "removed; use test_evaluation/confusion_matrix.png",
            "removed_val_confusion_files": removed_val_confusion,
        },
    )
    print("[DONE] 学習が完了しました。")
    print(f"[DONE] best.pt: {best_path}")


if __name__ == "__main__":
    main()
