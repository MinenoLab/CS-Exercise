from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "custom_yolo"
DEFAULT_RAW_IMAGES_DIR = PROJECT_ROOT / "frames" / "raw"
DEFAULT_RAW_LABELS_DIR = PROJECT_ROOT / "annotations" / "yolo_raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetReport:
    split_name: str
    image_count: int = 0
    label_count: int = 0
    missing_labels: list[Path] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)
    invalid_lines: list[str] = field(default_factory=list)

    @property
    def has_problem(self) -> bool:
        return bool(self.missing_labels or self.orphan_labels or self.invalid_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOデータセットの画像・ラベル対応とtxt形式を確認します。")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--class-count", type=int, default=None, help="有効なクラス数。未指定ならdata.yamlから読みます。")
    parser.add_argument("--check-raw", action="store_true", help="frames/raw と annotations/yolo_raw を確認します。")
    return parser.parse_args()


def load_class_count(dataset_dir: Path, explicit_count: int | None) -> int | None:
    if explicit_count is not None:
        return explicit_count
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return None
    data: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)
    return None


def load_data_yaml(dataset_dir: Path) -> dict[str, Any]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return {}
    return yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}


def infer_label_dir(dataset_dir: Path, image_dir_value: str) -> Path:
    parts = Path(image_dir_value).parts
    if parts and parts[0] == "images":
        return dataset_dir.joinpath("labels", *parts[1:])
    return dataset_dir / image_dir_value


def split_dirs_from_data_yaml(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    data = load_data_yaml(dataset_dir)
    if not data:
        return [
            ("train", dataset_dir / "images" / "train", dataset_dir / "labels" / "train"),
            ("val", dataset_dir / "images" / "val", dataset_dir / "labels" / "val"),
            ("test", dataset_dir / "images" / "test", dataset_dir / "labels" / "test"),
        ]

    split_dirs: list[tuple[str, Path, Path]] = []
    for split_name in ["train", "val", "test"]:
        image_dir_value = data.get(split_name)
        if not image_dir_value:
            continue
        image_dir = dataset_dir / str(image_dir_value)
        label_dir = infer_label_dir(dataset_dir, str(image_dir_value))
        split_dirs.append((split_name, image_dir, label_dir))
    return split_dirs


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        print(f"[WARN] 画像ディレクトリがありません: {images_dir}")
        return []
    return [path for path in sorted(images_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def list_labels(labels_dir: Path) -> list[Path]:
    if not labels_dir.exists():
        print(f"[WARN] ラベルディレクトリがありません: {labels_dir}")
        return []
    return [path for path in sorted(labels_dir.iterdir()) if path.is_file() and path.suffix == ".txt"]


def validate_yolo_line(label_path: Path, line_number: int, line: str, class_count: int | None) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    parts = stripped.split()
    if len(parts) != 5:
        return [f"{label_path}:{line_number}: 列数が不正です。5列必要ですが{len(parts)}列です。"]

    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return [f"{label_path}:{line_number}: 数値に変換できない値があります。"]

    errors: list[str] = []
    if class_id < 0:
        errors.append(f"{label_path}:{line_number}: class_id は0以上にしてください。値={class_id}")
    if class_count is not None and class_id >= class_count:
        errors.append(f"{label_path}:{line_number}: class_id がクラス数の範囲外です。値={class_id}, class_count={class_count}")

    for name, value in [("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)]:
        if value < 0 or value > 1:
            errors.append(f"{label_path}:{line_number}: {name} が0〜1の範囲外です。値={value}")
    if width <= 0:
        errors.append(f"{label_path}:{line_number}: width は0より大きい必要があります。値={width}")
    if height <= 0:
        errors.append(f"{label_path}:{line_number}: height は0より大きい必要があります。値={height}")
    return errors


def validate_label_file(label_path: Path, class_count: int | None) -> list[str]:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [f"{label_path}:1: UTF-8として読めません。"]
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        errors.extend(validate_yolo_line(label_path, line_number, line, class_count))
    return errors


def check_pair(images_dir: Path, labels_dir: Path, split_name: str, class_count: int | None) -> DatasetReport:
    images = list_images(images_dir)
    labels = list_labels(labels_dir)
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    report = DatasetReport(split_name=split_name, image_count=len(images), label_count=len(labels))
    report.missing_labels = [path for path in images if path.stem not in label_stems]
    report.orphan_labels = [path for path in labels if path.stem not in image_stems]
    for label_path in labels:
        report.invalid_lines.extend(validate_label_file(label_path, class_count))
    return report


def print_report(report: DatasetReport) -> None:
    print(f"\n[{report.split_name}]")
    print(f"画像数: {report.image_count}")
    print(f"ラベル数: {report.label_count}")
    print("ラベルがない画像: なし" if not report.missing_labels else "ラベルがない画像:")
    for path in report.missing_labels:
        print(f"  {path}")
    print("画像がないラベル: なし" if not report.orphan_labels else "画像がないラベル:")
    for path in report.orphan_labels:
        print(f"  {path}")
    print("YOLO形式として不正な行: なし" if not report.invalid_lines else "YOLO形式として不正な行:")
    for message in report.invalid_lines:
        print(f"  {message}")


def main() -> None:
    args = parse_args()
    class_count = load_class_count(args.dataset_dir, args.class_count)
    if class_count is not None:
        print(f"[INFO] クラス数: {class_count}")

    if args.check_raw:
        reports = [check_pair(DEFAULT_RAW_IMAGES_DIR, DEFAULT_RAW_LABELS_DIR, "raw", class_count)]
    else:
        reports = [check_pair(images_dir, labels_dir, split_name, class_count) for split_name, images_dir, labels_dir in split_dirs_from_data_yaml(args.dataset_dir)]

    has_problem = False
    for report in reports:
        print_report(report)
        has_problem = has_problem or report.has_problem
    if has_problem:
        print("\n[WARN] データセットに確認が必要な項目があります。")
    else:
        print("\n[DONE] データセット確認が完了しました。問題は見つかりませんでした。")


if __name__ == "__main__":
    main()
