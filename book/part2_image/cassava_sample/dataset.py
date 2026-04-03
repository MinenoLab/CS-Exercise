# dataset.py
import os
import glob
import cv2
from PIL import Image
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import Counter
from utils import ensure_dir, get_class_info

class CassavaDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Can't read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # numpy -> PIL に変換して transforms に渡す（ToPILImage を使う）
        img_pil = Image.fromarray(img)
        if self.transform:
            img = self.transform(img_pil)
        else:
            img = T.ToTensor()(img_pil)
        label = int(self.labels[idx])
        return img, label

def make_transforms(train: bool = True, image_size: int = 224):
    """
    ImageNet 準拠の正規化を使用。データ拡張は train=True の場合のみ有効。
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

def gather_image_paths_and_labels(data_dir: str):
    """
    以下の両方に対応:
      1) data_dir/train/<class>/*
      2) data_dir/<class>/*
    """
    class_root, class_info = get_class_info(data_dir)

    paths = []
    labels = []

    for folder_name, label_id, _ in class_info:
        cls_dir = os.path.join(class_root, folder_name)
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(glob.glob(os.path.join(cls_dir, ext)))

        for f in sorted(files):
            paths.append(f)
            labels.append(label_id)

    return paths, labels

def make_dataloaders(data_dir: str,
                     batch_size: int = 32,
                     val_size: float = 0.1,
                     test_size: float = 0.1,
                     seed: int = 42,
                     image_size: int = 224,
                     num_workers: int = 4):
    """
    データを stratified split で train / val / test に分割し、DataLoader を返す
    """
    paths, labels = gather_image_paths_and_labels(data_dir)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {data_dir}/train")

    # train vs temp (val+test)
    stratify = labels
    paths_train, paths_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(val_size + test_size), stratify=stratify, random_state=seed)
    # temp を val / test に分割 (比率を正規化)
    val_ratio = val_size / (val_size + test_size)
    paths_val, paths_test, y_val, y_test = train_test_split(
        paths_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=seed)

    # transforms
    train_trans = make_transforms(train=True, image_size=image_size)
    val_trans = make_transforms(train=False, image_size=image_size)

    train_dataset = CassavaDataset(paths_train, y_train, transform=train_trans)
    val_dataset = CassavaDataset(paths_val, y_val, transform=val_trans)
    test_dataset = CassavaDataset(paths_test, y_test, transform=val_trans)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def visualize_samples(dataset: Dataset, class_names: List[str], n: int = 8, save_path: str = None):
    """データセットからサンプルをプロットする（正規化を戻して表示）"""
    # 先頭から n サンプルを取得
    n = min(n, len(dataset))
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    for i in range(n):
        img, label = dataset[i]
        # img は正規化されたテンソル: C,H,W
        img_np = img.cpu().numpy()
        # Denormalize (ImageNet)
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1,2,0))
        axes[i].imshow(img_np)
        axes[i].set_title(f"{label}: {class_names[label] if class_names else label}")
        axes[i].axis("off")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path)
    plt.show()

def plot_label_distribution(labels: List[int], class_names: List[str], save_path: str = None):
    """ラベル分布の棒グラフを表示"""
    counter = Counter(labels)
    classes = sorted(list(set(labels)))
    counts = [counter[c] for c in classes]
    names = [class_names[c] if class_names else str(c) for c in classes]
    plt.figure(figsize=(8,4))
    plt.bar(names, counts)
    plt.xlabel("class")
    plt.ylabel("count")
    plt.title("Label distribution")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path)
    plt.show()