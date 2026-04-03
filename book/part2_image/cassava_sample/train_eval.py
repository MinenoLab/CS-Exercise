# train_eval.py
import os
import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from utils import ensure_dir, save_train_log_csv
from visualize import plot_learning_curve, plot_confusion_matrix, show_misclassified_examples

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Eval", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

def train(model, train_loader, val_loader, test_loader, device,
          epochs=10, lr=1e-3, results_dir="results", scheduler_step=5, scheduler_gamma=0.1, class_names=None):
    ensure_dir(results_dir)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    logs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        t1 = time.time()
        epoch_time = t1 - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        logs.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                     'val_loss': val_loss, 'val_acc': val_acc, 'time_sec': epoch_time})

        print(f"Epoch {epoch}/{epochs} - train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} | val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}  ({epoch_time:.1f}s)")

        # best model 保存（val_acc が基準）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_path = os.path.join(results_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved to {best_path} (val_acc={best_val_acc:.4f})")

    # 学習ログ CSV 保存
    csv_path = os.path.join(results_dir, "train_log.csv")
    save_train_log_csv(logs, csv_path)
    print(f"Training log saved to {csv_path}")

    # 学習曲線保存
    lc_path = os.path.join(results_dir, "learning_curve.png")
    plot_learning_curve(train_losses, val_losses, train_accs, val_accs, save_path=lc_path)
    print(f"Learning curve saved to {lc_path}")

        # 最良モデルの重みをロードしてテストセットで評価
    model.load_state_dict(best_model_wts)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # 混同行列・分類レポート
    if class_names is None or len(class_names) == 0:
        eval_class_names = [str(i) for i in sorted(set(y_true + y_pred))]
    else:
        eval_class_names = class_names

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names=eval_class_names, normalize=False, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    report = classification_report(y_true, y_pred, target_names=eval_class_names, digits=4)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    print(report)

    mis_path = os.path.join(results_dir, "misclassified.png")
    show_misclassified_examples(model, test_loader, device, eval_class_names, n_examples=8, save_path=mis_path)
    print(f"Misclassified examples saved to {mis_path}")