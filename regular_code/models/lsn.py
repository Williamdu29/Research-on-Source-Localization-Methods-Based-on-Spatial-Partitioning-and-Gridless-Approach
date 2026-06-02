import torch
import torch.nn as nn
import torch.nn.functional as F


class LSN(nn.Module):
    def __init__(self, num_sources: int):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_dw = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, groups=64)
        self.conv2_pw = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3_dw = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, groups=64)
        self.conv3_pw = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(32, num_sources)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        return self.fc_out(x)
