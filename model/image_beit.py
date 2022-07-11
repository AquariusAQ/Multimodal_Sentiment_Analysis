# BEiT: https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k

import torch
import torch.nn as nn

from transformers import BeitFeatureExtractor, BeitForImageClassification, BeitModel, BeitConfig
from transformers import logging
 
logging.set_verbosity_warning()

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

from model.utils import Flatten, BasicBlock, Bottleneck

class ImageModelBEiT(nn.Module):
    def __init__(self, opt, config, out_features, device="cpu"):
        super(ImageModelBEiT, self).__init__()
        
        self.device = device
        self.opt = opt 
        self.config = config

        # image
        self.model = BeitModel.from_pretrained(opt.image_model_path)
        self.model_config = BeitConfig.from_pretrained(opt.image_model_path)


        self.fc = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, out_features),
            nn.Dropout(opt.dropout),
            nn.ReLU(),
        )
        

    def forward(self, img):
        # img = img.permute(0,2,3,1)
        # inputs = self.feature_extractor(images=img[0], return_tensors="pt").to(self.device)
        # print(img)
        # outputs = self.model(**img)
        outputs = self.model(img['pixel_values'].squeeze(dim=1))['last_hidden_state']
        # print(outputs.shape)
        # exit(0)
        out = self.fc(outputs)
        return out



