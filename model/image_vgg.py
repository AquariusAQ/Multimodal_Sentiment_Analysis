import torch
import torch.nn as nn

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

from model.utils import Flatten, BasicBlock, Bottleneck, VGG16

class ImageModelVGG(nn.Module):
    def __init__(self, opt, config, out_features, device="cpu"):
        super(ImageModelVGG, self).__init__()
        self.device = device
        self.opt = opt 
        self.config = config

        # image
        self.vgg = VGG16(3, out_features, opt.size, dropout=opt.dropout)


    def forward(self, img):
        out = self.vgg(img)
        return out



