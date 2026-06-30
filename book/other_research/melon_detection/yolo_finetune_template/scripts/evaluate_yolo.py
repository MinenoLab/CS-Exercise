from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

try:
    from experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, resolve_experiment_name, sanitize_name, write_json
except ModuleNotFoundError:
    from scripts.experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, resolve_experiment_name, sanitize_name, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = PROJECT_ROOT / "datasets" / "custom_yolo" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOモデルを指定splitで評価し、評価指標と混同行列を保存します。")
    parser.add_argument("--model", default=str(find_default_model()), help="評価するbest.ptのパス。")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="data.yamlのパス。")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU閾値。")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--experiment", default="", help="実験名。未指定ならモデルパスから推定します。")
    parser.add_argument("--name", default="", help="評価結果の出力名。未指定なら<split>_evaluation。")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def metric_value(metrics: object, name: str) -> float | None:
    value = getattr(metrics, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except TypeError:
        return None


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] モデルが見つかりません: {model_path}")
    if not data_path.exists():
        raise SystemExit(f"[ERROR] data.yamlが見つかりません: {data_path}")

    experiment_name = resolve_experiment_name(args.experiment, model_path)
    exp_dir = experiment_dir(Path(args.experiments_dir), experiment_name)
    ensure_experiment_dirs(exp_dir)
    output_name = sanitize_name(args.name or f"{args.split}_evaluation")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(exp_dir),
        name=output_name,
        exist_ok=args.exist_ok,
        plots=True,
    )

    save_dir = Path(getattr(metrics, "save_dir", exp_dir / output_name))
    box = getattr(metrics, "box", None)
    write_json(
        save_dir / "evaluation_metadata.json",
        {
            "experiment": experiment_name,
            "model": str(model_path),
            "data": str(data_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "save_dir": str(save_dir),
            "precision": metric_value(box, "mp"),
            "recall": metric_value(box, "mr"),
            "map50": metric_value(box, "map50"),
            "map50_95": metric_value(box, "map"),
            "confusion_matrix": str(save_dir / "confusion_matrix.png"),
            "confusion_matrix_normalized": str(save_dir / "confusion_matrix_normalized.png"),
        },
    )
    print(f"[DONE] 評価結果を保存しました: {save_dir}")


if __name__ == "__main__":
    main()
