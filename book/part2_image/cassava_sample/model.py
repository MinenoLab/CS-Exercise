# model.py
"""
シンプルなCNNモデル定義
- Conv2d -> ReLU -> MaxPool を複数層
- 最後に全結合層で5クラス出力
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(SimpleCNN, self).__init__()
        # 入力 RGB (3チャネル)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224 -> 224x224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 14x14
        )
        # 全結合層: AdaptivePoolで固定長ベクトルにしてから使用
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # CrossEntropyLoss と組み合わせるので softmax は不要