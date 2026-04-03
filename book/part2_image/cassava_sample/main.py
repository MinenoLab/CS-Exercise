# main.py
"""
使い方（例）:
python main.py --data_dir ./dataset --epochs 10 --batch_size 32 --lr 1e-3
"""
import argparse
import os
from utils import set_seed, get_device, load_class_names, ensure_dir
from dataset import make_dataloaders, visualize_samples, plot_label_distribution, gather_image_paths_and_labels
from model import SimpleCNN
from train_eval import train
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset", help="dataset ディレクトリ (train/<class> または <class> 直下を想定)")
    parser.add_argument("--labels_file", type=str, default="./dataset/labels.txt", help="labels.txt があれば指定")
    parser.add_argument("--results_dir", type=str, default="./results", help="結果保存ディレクトリ")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # クラス名
    class_names = load_class_names(args.labels_file, data_dir=args.data_dir)
    print("Class names:", class_names)

    # dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_size=0.1,
        test_size=0.1,
        seed=args.seed,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # 可視化: ラベル分布
    paths, labels = gather_image_paths_and_labels(args.data_dir)
    ensure_dir(args.results_dir)
    from dataset import plot_label_distribution, visualize_samples
    plot_label_distribution(labels, class_names, save_path=os.path.join(args.results_dir, "label_distribution.png"))

    # 訓練に使う画像の可視化
    visualize_samples(train_loader.dataset, class_names, n=8, save_path=os.path.join(args.results_dir, "samples.png"))

    # モデル作成
    model = SimpleCNN(num_classes=len(class_names) if class_names else 5)

    # 学習開始
    model, logs = train(model, train_loader, val_loader, test_loader, device,
                       epochs=args.epochs, lr=args.lr, results_dir=args.results_dir,
                       scheduler_step=max(1, args.epochs // 2), scheduler_gamma=0.1, class_names=class_names)

    print("Training finished. Results saved in:", args.results_dir)

if __name__ == "__main__":
    main()