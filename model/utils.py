import torch
import torch.nn as nn

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

# Inception for 1D
class Inception1D(nn.Module):
    def __init__(self, in_channels, out_channels, out_channels_b1, channels_b2_0, out_channels_b2, channels_b3_0, out_channels_b3, out_channels_b4, dim, inception_dropout_prob=0.1):
        super(Inception1D, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_b1, kernel_size=1),
            nn.BatchNorm1d(out_channels_b1),
            # nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.b2 = nn.Sequential(
            nn.Conv1d(in_channels, channels_b2_0, kernel_size=1),
            nn.BatchNorm1d(channels_b2_0),
            # nn.LayerNorm(dim),
            nn.GELU(),
            nn.Conv1d(channels_b2_0, out_channels_b2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels_b2),
            # nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.b3 = nn.Sequential(
            nn.Conv1d(in_channels, channels_b3_0, kernel_size=1),
            nn.BatchNorm1d(channels_b3_0),
            # nn.LayerNorm(dim),
            nn.GELU(),
            nn.Conv1d(channels_b3_0, out_channels_b3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels_b3),
            # nn.LayerNorm(dim),
            nn.GELU(),
            nn.Conv1d(out_channels_b3, out_channels_b3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels_b3),
            # nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels_b4, kernel_size=1),
            nn.BatchNorm1d(out_channels_b4),
            # nn.LayerNorm(dim),
            nn.GELU(),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(out_channels_b1 + out_channels_b2 + out_channels_b3 + out_channels_b4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            # nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(inception_dropout_prob),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y  = torch.cat([y1, y2, y3, y4], 1)
        out= self.conv(y)
        return out

# flatten 2D to 1D
class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.reshape(N, -1)

# Bottleneck For ResNet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1),
            nn.BatchNorm2d(self.expansion * out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.shortcut(x)
        out = x1 + x2
        out = self.relu(out)
        return out

# BasicBlock For ResNet18
class BasicBlock(nn.Module):
    # 相比于输入维度，输出维度的增大倍数
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 定义两个 3*3 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # 用于引入残差块的输入
        self.shortcut = nn.Sequential()
        # 如果处理后的维度维度大小不足，需要进行升维；如果 stride 大于 1，需要进行缩小
        # 这里的方式是使用 1*1 卷积，也可以使用全 0 填充
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x2 = self.shortcut(x)
        out = x1 + x2
        out = self.relu(out)
        return out

# VGG16
class VGG16(nn.Module):
    def __init__(self, in_channels=1, num_features=10, img_size=64, dropout=0.25):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear((img_size // 32) ** 2 * 512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_features),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        output = self.fc(x)
        return output
