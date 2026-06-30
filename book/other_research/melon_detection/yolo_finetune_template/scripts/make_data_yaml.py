from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "custom_yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO用のdata.yamlを生成します。")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train", default="images/train", help="dataset-dirから見たtrain画像ディレクトリ。")
    parser.add_argument("--val", default="images/valid", help="dataset-dirから見たval画像ディレクトリ。")
    parser.add_argument("--test", default="images/test", help="dataset-dirから見たtest画像ディレクトリ。空文字ならtestを書きません。")
    parser.add_argument("--class-names", nargs="*", default=None, help="クラス名。例: --class-names class_a class_b")
    parser.add_argument("--classes-file", type=Path, default=None, help="1行1クラス名のテキストファイル。")
    return parser.parse_args()


def load_class_names(class_names: list[str] | None, classes_file: Path | None) -> list[str]:
    if classes_file:
        if not classes_file.exists():
            raise FileNotFoundError(f"クラス名ファイルが見つかりません: {classes_file}")
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = [name.strip() for name in (class_names or []) if name.strip()]
    return names or ["target_object"]


def path_for_yaml(dataset_dir: Path) -> str:
    try:
        return dataset_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return dataset_dir.as_posix()


def has_files(directory: Path) -> bool:
    return directory.exists() and any(path.is_file() for path in directory.iterdir())


def make_data_yaml(dataset_dir: Path, train: str, val: str, test: str, names: list[str]) -> dict:
    data = {
        "path": path_for_yaml(dataset_dir),
        "train": train,
        "val": val,
    }
    if test and has_files(dataset_dir / test):
        data["test"] = test
    data["names"] = {index: name for index, name in enumerate(names)}
    return data


def main() -> None:
    args = parse_args()
    names = load_class_names(args.class_names, args.classes_file)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.dataset_dir / "data.yaml"
    output_path.write_text(
        yaml.safe_dump(make_data_yaml(args.dataset_dir, args.train, args.val, args.test, names), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[DONE] data.yaml を生成しました: {output_path}")
    print(f"[DONE] クラス数: {len(names)}")


if __name__ == "__main__":
    main()
