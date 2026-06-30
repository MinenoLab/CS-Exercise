from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "videos" / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "frames" / "raw"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動画から一定間隔で画像を切り出します。")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR, help="入力動画ディレクトリ。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="切り出した画像の保存先。")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="画像を保存する間隔（秒）。")
    parser.add_argument("--patterns", nargs="*", default=["*.mp4"], help="対象動画のglobパターン。例: *.mp4 *.mov")
    parser.add_argument("--recursive", action="store_true", help="入力動画ディレクトリを再帰的に探索します。")
    parser.add_argument("--overwrite", action="store_true", help="既存画像を上書きします。")
    return parser.parse_args()


def list_videos(video_dir: Path, patterns: list[str], recursive: bool) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"入力動画ディレクトリがありません: {video_dir}")

    videos: set[Path] = set()
    for pattern in patterns:
        iterator = video_dir.rglob(pattern) if recursive else video_dir.glob(pattern)
        for path in iterator:
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.add(path)
    return sorted(videos)


def open_video(video_path: Path) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 動画を開けませんでした。スキップします: {video_path}")
        return None
    return cap


def save_frame(frame, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        print(f"[WARN] 画像保存に失敗しました: {output_path}")
    return bool(success)


def extract_frames_from_video(video_path: Path, output_dir: Path, interval_sec: float, overwrite: bool) -> int:
    cap = open_video(video_path)
    if cap is None:
        return 0

    saved_count = 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            print(f"[WARN] FPSまたは総フレーム数を取得できません。スキップします: {video_path}")
            return 0

        frame_step = max(1, int(round(fps * interval_sec)))
        output_index = 1
        print(f"[INFO] {video_path.name}: fps={fps:.3f}, frames={frame_count}, step={frame_step}")

        for frame_index in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(f"[WARN] フレーム読み込み失敗: {video_path} frame={frame_index}")
                continue

            output_path = output_dir / f"{video_path.stem}_{output_index:06d}.jpg"
            if save_frame(frame, output_path, overwrite):
                saved_count += 1
            output_index += 1
    finally:
        cap.release()

    print(f"[INFO] {video_path.name}: 保存枚数 {saved_count} 枚")
    return saved_count


def main() -> None:
    args = parse_args()
    if args.interval_sec <= 0:
        raise SystemExit("[ERROR] --interval-sec は 0 より大きい値にしてください。")

    try:
        videos = list_videos(args.video_dir, args.patterns, args.recursive)
    except Exception as exc:
        raise SystemExit(f"[ERROR] 動画探索に失敗しました: {exc}") from exc

    if not videos:
        print(f"[WARN] 対象動画が見つかりません: {args.video_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    for video_path in videos:
        total_saved += extract_frames_from_video(video_path, args.output_dir, args.interval_sec, args.overwrite)

    print(f"[DONE] フレーム切り出し完了。合計保存枚数: {total_saved} 枚")


if __name__ == "__main__":
    main()
