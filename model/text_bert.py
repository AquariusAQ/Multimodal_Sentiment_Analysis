import torch
import torch.nn as nn

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

from model.utils import Inception1D

class TextModelBert(nn.Module):
    def __init__(self, opt, config, out_features, tokenizer, encoder, rnn_dim=128, inception_dim=256 ,device="cpu"):
        super(TextModelBert, self).__init__()
        self.device = device
        self.opt = opt 
        self.config = config


        # text
        self.tokenizer = tokenizer
        self.encoder = encoder

        self.rnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=6, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(opt.dropout)

        # self.inception = Inception(config.hidden_size, inception_dim, 256, 384, 512, 64, 128, 128)
        self.inception = Inception1D(config.hidden_size, inception_dim, 128, 192, 256, 32, 64, 64, dim=opt.max_length)

        channel_list = [rnn_dim*2+inception_dim, 512, out_features]
        conv = []
        for i in range(len(channel_list)-1):
            conv.append(nn.Conv1d(channel_list[i], channel_list[i+1], kernel_size=3, padding=1))
            conv.append(nn.GELU()),
            conv.append(nn.BatchNorm1d(channel_list[i+1]))
        self.conv = nn.Sequential(*conv)
        

    def forward(self, txt):
        input_ids = txt['input_ids']
        token_type_ids = txt['token_type_ids']
        attn_mask = txt['attention_mask']
        sequence_output = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attn_mask)['last_hidden_state']
        # batch_size max_length hidden_size


        sequence_output = self.dropout(sequence_output)

        main, _ = self.rnn(sequence_output)
        side = self.inception(sequence_output.transpose(1, 2))

        out = torch.cat((main, side.transpose(1, 2)), dim=2)

        out = self.conv(out.transpose(1, 2)).transpose(1, 2)
        # out = self.fc(out)

        return out
