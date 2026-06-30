from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "runs" / "experiments"
DEFAULT_EXPERIMENT_NAME = "yolo_experiment"


def sanitize_name(name: str) -> str:
    """ディレクトリ名として扱いやすい実験名に整えます。"""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return cleaned.strip("._-") or DEFAULT_EXPERIMENT_NAME


def format_float_for_name(value: float) -> str:
    return str(value).replace(".", "p")


def infer_experiment_name_from_model(model_path: Path) -> str:
    """モデルパスから実験名を推定します。"""
    parts = model_path.parts
    if "experiments" in parts:
        index = parts.index("experiments")
        if index + 1 < len(parts):
            return sanitize_name(parts[index + 1])

    if model_path.parent.name == "weights" and len(model_path.parents) >= 2:
        return sanitize_name(model_path.parents[1].name)

    return sanitize_name(model_path.stem)


def resolve_experiment_name(
    experiment: str,
    model_path: Path | None = None,
    fallback: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    if experiment:
        return sanitize_name(experiment)
    if model_path is not None:
        return infer_experiment_name_from_model(model_path)
    return sanitize_name(fallback)


def experiment_dir(experiments_dir: Path, experiment_name: str) -> Path:
    return experiments_dir / sanitize_name(experiment_name)


def ensure_experiment_dirs(base_dir: Path) -> None:
    """実験単位で使う標準ディレクトリを作ります。"""
    for relative in [
        "train",
        "test_evaluation",
        "predictions/images",
        "predictions/videos",
        "error_analysis",
        "metadata",
    ]:
        directory = base_dir / relative
        directory.mkdir(parents=True, exist_ok=True)
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), **data}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_default_model() -> Path:
    candidates = [
        DEFAULT_EXPERIMENTS_DIR / DEFAULT_EXPERIMENT_NAME / "train" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / DEFAULT_EXPERIMENT_NAME / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
