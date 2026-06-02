import torch
import torch.nn as nn
import torch.nn.functional as F


class LCN(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        out_dim = 4 * (N - 2)

        self.conv1 = nn.Conv2d(2, 128, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2_dw = nn.Conv2d(128, 128, 2, groups=128)
        self.conv2_pw = nn.Conv2d(128, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_dw = nn.Conv2d(128, 128, 2, groups=128)
        self.conv3_pw = nn.Conv2d(128, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = F.leaky_relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
