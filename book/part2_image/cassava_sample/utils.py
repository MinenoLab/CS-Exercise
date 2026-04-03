# utils.py
import os
import random
import numpy as np
import torch
import pandas as pd

CANONICAL_CLASSES = [
    {
        "aliases": ["Cassva__bacterial_blight", "Cassava__bacterial_blight"],
        "display_name": "細菌性胴枯れ病",
    },
    {
        "aliases": ["Cassva__brown_streak_disease", "Cassava__brown_streak_disease"],
        "display_name": "条斑ウイルス",
    },
    {
        "aliases": ["Cassva__green_mottle", "Cassava__green_mottle"],
        "display_name": "緑斑ウイルス",
    },
    {
        "aliases": ["Cassva__mosaic_disease", "Cassava__mosaic_disease"],
        "display_name": "モザイク病",
    },
    {
        "aliases": ["Cassva__healthy", "Cassava__healthy"],
        "display_name": "健康",
    },
]

def set_seed(seed: int = 42):
    """シード値固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_train_log_csv(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def _get_class_root(data_dir: str) -> str:
    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir):
        return train_dir
    return data_dir

def get_class_info(data_dir: str):
    """
    戻り値:
      class_root: クラスフォルダが並んでいるディレクトリ
      class_info: [(folder_name, label_id, display_name), ...]
    """
    class_root = _get_class_root(data_dir)
    if not os.path.isdir(class_root):
        raise RuntimeError(f"Dataset directory not found: {class_root}")

    subdirs = [d for d in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, d))]
    if not subdirs:
        raise RuntimeError(f"No class directories found in: {class_root}")

    if all(d.isdigit() for d in subdirs):
        subdirs_sorted = sorted(subdirs, key=int)
        class_info = [(d, int(d), d) for d in subdirs_sorted]
        return class_root, class_info

    lower_to_original = {d.lower(): d for d in subdirs}
    class_info = []
    matched_all = True

    for label_id, cls in enumerate(CANONICAL_CLASSES):
        found = None
        for alias in cls["aliases"]:
            if alias.lower() in lower_to_original:
                found = lower_to_original[alias.lower()]
                break
        if found is None:
            matched_all = False
            break
        class_info.append((found, label_id, cls["display_name"]))

    if matched_all:
        return class_root, class_info

    subdirs_sorted = sorted(subdirs)
    class_info = [(d, idx, d) for idx, d in enumerate(subdirs_sorted)]
    return class_root, class_info

def load_class_names(labels_file: str = "", data_dir: str = None):
    """
    labels_file が存在すれば読み込み、
    なければ data_dir 配下のクラスフォルダからクラス名を決定する
    """
    if labels_file and os.path.isfile(labels_file):
        names = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    _, label = line.split(":", 1)
                    names.append(label.strip())
                else:
                    names.append(line)
        return names

    if data_dir:
        _, class_info = get_class_info(data_dir)
        return [display_name for _, _, display_name in class_info]

    return []