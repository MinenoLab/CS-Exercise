# visualize.py
"""
可視化: 学習曲線、混同行列、誤分類例
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import torch
from utils import ensure_dir
import japanize_matplotlib

def plot_learning_curve(train_losses, val_losses, train_accs, val_accs, save_path=None):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='train_acc')
    plt.plot(epochs, val_accs, label='val_acc')
    plt.xlabel('epoch'); plt.ylabel('acc'); plt.legend()
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, cmap=plt.cm.Blues, save_path=None):
    """
    sklearn.metrics.confusion_matrix を用いたヒートマップ表示
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path)
    plt.show()

def show_misclassified_examples(model, dataloader, device, class_names, n_examples=8, save_path=None):
    """
    データローダを評価して誤分類例を数枚表示
    """
    model.eval()
    device = device
    mis_imgs = []
    mis_trues = []
    mis_preds = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            mask = preds != labels
            if mask.any():
                mis = imgs[mask].cpu()
                mtr = labels[mask].cpu().tolist()
                mpred = preds[mask].cpu().tolist()
                for i in range(mis.size(0)):
                    mis_imgs.append(mis[i])
                    mis_trues.append(mtr[i])
                    mis_preds.append(mpred[i])
            if len(mis_imgs) >= n_examples:
                break
    n = min(n_examples, len(mis_imgs))
    if n == 0:
        print("誤分類が見つかりませんでした。")
        return
    plt.figure(figsize=(3*n,3))
    for i in range(n):
        img = mis_imgs[i]
        # Denormalize
        img_np = img.numpy()
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1,2,0))
        plt.subplot(1,n,i+1)
        plt.imshow(img_np)
        plt.title(f"T:{mis_trues[i]} {class_names[mis_trues[i]]}\nP:{mis_preds[i]} {class_names[mis_preds[i]]}")
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path)
    plt.show()