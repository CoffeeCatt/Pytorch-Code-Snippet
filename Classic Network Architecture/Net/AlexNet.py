import torch
import torch.nn as nn
from typing import Any


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 3 * 224 * 224 --> 64 * 55 * 55
            # np.floor((224 + 2*2-1*(11-1)-1)/4+1) = 55
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # inplace means to inplace operation
            nn.ReLU(inplace=True),
            # 64 * 55 * 55 --> 64 * 27 * 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 64 * 27 * 27 --> 192 * 27 * 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 192 * 27 * 27 --> 192 * 13 * 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 192 * 13 * 13 --> 384 * 13 * 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 384 * 13 * 13 --> 256 * 13 * 13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 256 * 12 * 12 --> 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 256 * 13 * 13 --> 256 * 6 * 6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

