from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, format_float_for_name, resolve_experiment_name, sanitize_name, write_json
except ModuleNotFoundError:
    from scripts.experiment_paths import DEFAULT_EXPERIMENTS_DIR, ensure_experiment_dirs, experiment_dir, find_default_model, format_float_for_name, resolve_experiment_name, sanitize_name, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "videos" / "input" / "sample.mp4"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動画に対してYOLO推論を実行します。")
    parser.add_argument("--model", default=str(find_default_model()))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="入力動画パス。")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--experiment", default="", help="実験名。未指定ならモデルパスから推定します。")
    parser.add_argument("--name", default="", help="predictions/videos内の出力名。")
    parser.add_argument("--output-format", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--keep-avi", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def build_output_name(source_path: Path, conf: float, requested_name: str) -> str:
    if requested_name:
        return sanitize_name(requested_name)
    return sanitize_name(f"{source_path.stem}_conf{format_float_for_name(conf)}")


def find_video_outputs(save_dir: Path, source_stem: str) -> list[Path]:
    return [path for path in sorted(save_dir.glob(f"{source_stem}.*")) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]


def convert_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [ffmpeg, "-y", "-i", str(input_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"[WARN] ffmpegでmp4変換できませんでした: {completed.stderr.strip()}")
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def convert_with_opencv(input_path: Path, output_path: Path) -> bool:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def ensure_mp4(video_path: Path, keep_avi: bool) -> Path:
    if video_path.suffix.lower() == ".mp4":
        return video_path
    if video_path.suffix.lower() != ".avi":
        return video_path
    mp4_path = video_path.with_suffix(".mp4")
    converted = convert_with_ffmpeg(video_path, mp4_path) or convert_with_opencv(video_path, mp4_path)
    if not converted:
        raise RuntimeError(f"mp4変換に失敗しました: {video_path}")
    if not keep_avi:
        video_path.unlink()
    return mp4_path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] モデルが見つかりません: {model_path}")
    if not source_path.exists():
        raise SystemExit(f"[ERROR] 入力動画が見つかりません: {source_path}")

    experiment_name = resolve_experiment_name(args.experiment, model_path)
    exp_dir = experiment_dir(Path(args.experiments_dir), experiment_name)
    ensure_experiment_dirs(exp_dir)
    output_parent = exp_dir / "predictions" / "videos"
    output_name = build_output_name(source_path, args.conf, args.name)

    model = YOLO(str(model_path))
    results = model.predict(source=str(source_path), conf=args.conf, device=args.device, save=True, project=str(output_parent), name=output_name, exist_ok=args.exist_ok)
    save_dir = Path(results[0].save_dir) if results else output_parent / output_name
    video_outputs = find_video_outputs(save_dir, source_path.stem)
    if args.output_format == "mp4":
        video_outputs = [ensure_mp4(path, keep_avi=args.keep_avi) for path in video_outputs]
    write_json(save_dir / "predict_metadata.json", {"experiment": experiment_name, "type": "video", "model": str(model_path), "source": str(source_path), "conf": args.conf, "device": args.device, "output_format": args.output_format, "keep_avi": args.keep_avi, "output_dir": str(save_dir), "output_files": [str(path) for path in video_outputs]})
    print(f"[DONE] 検出結果動画を保存しました: {save_dir}")
    for output_path in video_outputs:
        print(f"[DONE] 動画: {output_path}")


if __name__ == "__main__":
    main()
