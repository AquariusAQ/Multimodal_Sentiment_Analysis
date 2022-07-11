import torch
import torch.nn as nn

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

from model.text_bert import TextModelBert
from model.image_resnet import ImageModelResNet
from model.image_vit import ImageModelViT
from model.image_beit import ImageModelBEiT
from model.image_vgg import ImageModelVGG
from model.utils import Flatten

class Model(nn.Module):
    # Multimodal Sentiment Analysis
    def __init__(self, opt, config, tokenizer, encoder, img_features=256, txt_features=256, rnn_dim=128, inception_dim=256, device="cpu"):
        super(Model, self).__init__()
        self.device = device
        self.opt = opt 
        self.config = config
        self.img_features = img_features
        self.txt_features = txt_features

        emb_size = rnn_dim * 2 + inception_dim

        if opt.image_model in ["resnet18", "resnet50"]:
            self.imgmodel = ImageModelResNet(opt, config, img_features, device="cpu", model=opt.image_model)
        elif opt.image_model == "vit":
            self.imgmodel = ImageModelViT(opt, config, img_features, device=device)
        elif opt.image_model == "beit":
            self.imgmodel = ImageModelBEiT(opt, config, img_features, device=device)
        elif opt.image_model == "vgg":
            self.imgmodel = ImageModelVGG(opt, config, img_features, device=device)
        else:
            raise "Invalid image model: " + opt.image_model

        self.img_scale = nn.Linear(img_features, txt_features, bias=False)

        self.txtmodel = TextModelBert(opt, config, txt_features, tokenizer, encoder, rnn_dim=rnn_dim, inception_dim=inception_dim, device=device)

        self.attn = nn.MultiheadAttention(txt_features, num_heads=8, batch_first=True)

        # bs * 197 * txt_features
        self.conv_beit = ConvBEit(self.txt_features, self.txt_features)

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.txt_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )


    def forward(self, img:torch.Tensor, txt:torch.Tensor):
        img_feature = self.imgmodel(img)
        # batch_size * channel * img_features

        txt_feature = self.txtmodel(txt)
        # batch_size * sent_length * embedding_size
        # embedding_size = rnn_dim * 2 + inception_dim


        # img_feature = img_feature.reshape(img_feature.shape[0], img_feature.shape[1], -1).transpose(1, 2)
        if self.img_features != self.txt_features:
            img_feature = self.img_scale(img_feature)
        if len(img_feature.shape) == 2:
            img_feature = img_feature.unsqueeze(1)
        
        attn_output, _ = self.attn(img_feature, txt_feature, txt_feature)

        if self.opt.image_model == "beit": 
            attn_output = self.conv_beit(attn_output.transpose(-1,-2))

        output = self.fc(attn_output)
        return output

class ConvBEit(nn.Module):
    def __init__(self, in_channels, out_features):
        super(ConvBEit, self).__init__()
        self.conv = nn.Sequential(
                # 197
                nn.Conv1d(in_channels, 256, kernel_size=4, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),
                # 98
            )

        self.fc = nn.Sequential(
            nn.Linear(98, 3),
            nn.ReLU(),
            nn.Dropout(0.25),
            Flatten(),
            nn.Linear(3*256, out_features)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# @inproceedings{truong2019vistanet,
#   title={VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis},
#   author={Truong, Quoc-Tuan and Lauw, Hady W},
#   publisher={AAAI Press},
#   year={2019},
# }






        