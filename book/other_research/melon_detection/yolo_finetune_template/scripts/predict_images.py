from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

try:
    from experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, format_float_for_name, resolve_experiment_name, sanitize_name, write_json
except ModuleNotFoundError:
    from scripts.experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, format_float_for_name, resolve_experiment_name, sanitize_name, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "custom_yolo" / "images" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="画像に対してYOLO推論を実行します。")
    parser.add_argument("--model", default=str(find_default_model()))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--experiment", default="", help="実験名。未指定ならモデルパスから推定します。")
    parser.add_argument("--name", default="", help="predictions/images内の出力名。")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def build_output_name(source_path: Path, conf: float, requested_name: str) -> str:
    if requested_name:
        return sanitize_name(requested_name)
    return sanitize_name(f"{source_path.stem}_conf{format_float_for_name(conf)}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] モデルが見つかりません: {model_path}")
    if not source_path.exists():
        raise SystemExit(f"[ERROR] 推論対象が見つかりません: {source_path}")

    experiment_name = resolve_experiment_name(args.experiment, model_path)
    exp_dir = experiment_dir(Path(args.experiments_dir), experiment_name)
    ensure_experiment_dirs(exp_dir)
    output_parent = exp_dir / "predictions" / "images"
    output_name = build_output_name(source_path, args.conf, args.name)

    model = YOLO(str(model_path))
    results = model.predict(source=str(source_path), conf=args.conf, device=args.device, save=True, project=str(output_parent), name=output_name, exist_ok=args.exist_ok)
    save_dir = Path(results[0].save_dir) if results else output_parent / output_name
    write_json(save_dir / "predict_metadata.json", {"experiment": experiment_name, "type": "images", "model": str(model_path), "source": str(source_path), "conf": args.conf, "device": args.device, "output_dir": str(save_dir)})
    print(f"[DONE] 検出結果画像を保存しました: {save_dir}")


if __name__ == "__main__":
    main()
