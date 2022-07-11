import torch
import torch.nn as nn

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

from model.utils import Flatten, BasicBlock, Bottleneck

class ImageModelResNet(nn.Module):
    def __init__(self, opt, config, out_features, device="cpu", model:str="resnet18"):
        super(ImageModelResNet, self).__init__()
        self.device = device
        self.opt = opt 
        self.config = config

        # image
        self.in_channels = 64

        if model == "resnet50":
            # ResNet50
            num_blocks = [3,4,6,3]
            block = Bottleneck
        elif model == "resnet18":
            # ResNet18
            block = BasicBlock
            num_blocks = [2,2,2,2]
        else:
            raise "invalid resnet model: " + model

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(512, out_features, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_features),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(512 * block.expansion, out_features),
            nn.ReLU()
        )
        
    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, img):
        x = self.conv1(img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0],-1)
        out = self.fc(x)
        # out = self.conv2(x)
        return out



