from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "frames" / "raw"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "annotations" / "yolo_raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "custom_yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="画像とYOLOラベルを train/val/test に分割します。")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-empty-labels", action="store_true", help="空ラベルtxtの画像も含めます。")
    parser.add_argument("--split-by-group", action="store_true", help="ファイル名の接頭辞単位で分割します。")
    parser.add_argument("--group-delimiter", default="_", help="グループID抽出に使う区切り文字。")
    parser.add_argument("--val-groups", nargs="*", default=None, help="valに入れるグループID。")
    parser.add_argument("--test-groups", nargs="*", default=None, help="testに入れるグループID。")
    return parser.parse_args()


def collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"画像ディレクトリがありません: {images_dir}")
    return [path for path in sorted(images_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def is_empty_label_file(label_path: Path) -> bool:
    return not label_path.read_text(encoding="utf-8").strip()


def collect_labeled_samples(images_dir: Path, labels_dir: Path, include_empty_labels: bool) -> list[tuple[Path, Path]]:
    if not labels_dir.exists():
        raise FileNotFoundError(f"ラベルディレクトリがありません: {labels_dir}")

    samples: list[tuple[Path, Path]] = []
    skipped_missing = 0
    skipped_empty = 0
    for image_path in collect_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] ラベルがないためスキップします: {image_path.name}")
            skipped_missing += 1
            continue
        if not include_empty_labels and is_empty_label_file(label_path):
            print(f"[WARN] 空ラベルのためスキップします: {image_path.name}")
            skipped_empty += 1
            continue
        samples.append((image_path, label_path))

    print(f"[INFO] 分割対象画像: {len(samples)} 枚")
    print(f"[INFO] ラベルなしでスキップ: {skipped_missing} 枚")
    print(f"[INFO] 空ラベルでスキップ: {skipped_empty} 枚")
    return samples


def recreate_output_dirs(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    split_dirs: dict[str, tuple[Path, Path]] = {}
    for split_name in ["train", "val", "test"]:
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        for path in [image_dir, label_dir]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = (image_dir, label_dir)
    return split_dirs


def get_group_id(image_path: Path, delimiter: str) -> str:
    if delimiter and delimiter in image_path.stem:
        return image_path.stem.split(delimiter)[0]
    return image_path.stem


def split_by_explicit_groups(
    samples: list[tuple[Path, Path]],
    val_groups: list[str] | None,
    test_groups: list[str] | None,
    delimiter: str,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    val_set = set(val_groups or [])
    test_set = set(test_groups or [])
    overlap = sorted(val_set & test_set)
    if overlap:
        raise ValueError(f"同じグループをvalとtestの両方には指定できません: {overlap}")

    all_groups = {get_group_id(image_path, delimiter) for image_path, _ in samples}
    unknown = sorted((val_set | test_set) - all_groups)
    if unknown:
        raise ValueError(f"指定されたグループが見つかりません: {unknown}")

    train_samples = []
    val_samples = []
    test_samples = []
    for sample in samples:
        image_path, _ = sample
        group_id = get_group_id(image_path, delimiter)
        if group_id in test_set:
            test_samples.append(sample)
        elif group_id in val_set:
            val_samples.append(sample)
        else:
            train_samples.append(sample)
    return train_samples, val_samples, test_samples


def split_random(
    samples: list[tuple[Path, Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if test_ratio > 0:
        train_val, test_samples = train_test_split(samples, test_size=test_ratio, random_state=seed, shuffle=True)
    else:
        train_val, test_samples = list(samples), []

    if not train_val:
        raise ValueError("train/val候補が0枚です。")
    if len(train_val) == 1 or val_ratio <= 0:
        return list(train_val), [], list(test_samples)

    train_samples, val_samples = train_test_split(train_val, test_size=val_ratio, random_state=seed, shuffle=True)
    return list(train_samples), list(val_samples), list(test_samples)


def split_groups_random(
    samples: list[tuple[Path, Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    delimiter: str,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    groups = sorted({get_group_id(image_path, delimiter) for image_path, _ in samples})
    if len(groups) < 2:
        raise ValueError("グループ単位分割には2つ以上のグループが必要です。")

    if test_ratio > 0:
        train_val_groups, test_groups = train_test_split(groups, test_size=test_ratio, random_state=seed, shuffle=True)
    else:
        train_val_groups, test_groups = groups, []

    if len(train_val_groups) < 2 or val_ratio <= 0:
        train_groups, val_groups = train_val_groups, []
    else:
        train_groups, val_groups = train_test_split(train_val_groups, test_size=val_ratio, random_state=seed, shuffle=True)

    return split_by_explicit_groups(samples, list(val_groups), list(test_groups), delimiter)


def split_samples(args: argparse.Namespace, samples: list[tuple[Path, Path]]) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not samples:
        raise ValueError("分割対象が0枚です。")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio は0以上1未満にしてください。")
    if args.test_ratio < 0 or args.test_ratio >= 1:
        raise ValueError("--test-ratio は0以上1未満にしてください。")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio と --test-ratio の合計は1未満にしてください。")

    if args.val_groups or args.test_groups:
        return split_by_explicit_groups(samples, args.val_groups, args.test_groups, args.group_delimiter)
    if args.split_by_group:
        return split_groups_random(samples, args.val_ratio, args.test_ratio, args.seed, args.group_delimiter)
    return split_random(samples, args.val_ratio, args.test_ratio, args.seed)


def copy_samples(samples: list[tuple[Path, Path]], image_output_dir: Path, label_output_dir: Path) -> None:
    for image_path, label_path in samples:
        shutil.copy2(image_path, image_output_dir / image_path.name)
        shutil.copy2(label_path, label_output_dir / label_path.name)


def print_group_counts(samples: list[tuple[Path, Path]], title: str, delimiter: str) -> None:
    counts: dict[str, int] = {}
    for image_path, _ in samples:
        group_id = get_group_id(image_path, delimiter)
        counts[group_id] = counts.get(group_id, 0) + 1
    print(f"[INFO] {title} グループ別枚数:")
    if not counts:
        print("  なし")
        return
    for group_id in sorted(counts):
        print(f"  {group_id}: {counts[group_id]} 枚")


def main() -> None:
    args = parse_args()
    try:
        samples = collect_labeled_samples(args.images_dir, args.labels_dir, args.include_empty_labels)
        train_samples, val_samples, test_samples = split_samples(args, samples)
        split_dirs = recreate_output_dirs(args.output_dir)
        copy_samples(train_samples, *split_dirs["train"])
        copy_samples(val_samples, *split_dirs["val"])
        copy_samples(test_samples, *split_dirs["test"])
    except Exception as exc:
        raise SystemExit(f"[ERROR] データセット分割に失敗しました: {exc}") from exc

    print("[DONE] train/val/test 分割が完了しました。")
    print(f"[DONE] train: {len(train_samples)} 枚")
    print(f"[DONE] val  : {len(val_samples)} 枚")
    print(f"[DONE] test : {len(test_samples)} 枚")
    print_group_counts(train_samples, "train", args.group_delimiter)
    print_group_counts(val_samples, "val", args.group_delimiter)
    print_group_counts(test_samples, "test", args.group_delimiter)
    print(f"[DONE] 出力先: {args.output_dir}")


if __name__ == "__main__":
    main()
