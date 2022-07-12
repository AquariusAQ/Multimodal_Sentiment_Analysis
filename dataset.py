import os
import torch
from torch.utils.data import Dataset
from transformers import BeitFeatureExtractor, BeitForImageClassification, BeitModel

from PIL import Image

import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

test_label_path = "./datasets/test_without_label.txt"
train_label_path = "./datasets/train.txt"

NULL = -1
POSITIVE = 0
NEUTRAL = 1
NEGATIVE = 2


class MyDataset(Dataset):
    def __init__(self, opt, index:list, label:list, train:bool, tokenizer, transform=None):
        self.opt = opt
        self.index = index
        self.label = label
        self.train = train
        self.transform = transform
        self.tokenizer = tokenizer
        self.mask_image = opt.mask_image
        self.mask_text = opt.mask_text
        
        if opt.image_model ==  "beit" :
            self.feature_extractor = BeitFeatureExtractor.from_pretrained(opt.image_model_path)

    def __getitem__(self, index):
        idx = self.index[index]
        
        if self.mask_image:
            img = Image.new(mode = "RGB", size = (128, 128))
        else:
            img = Image.open(r"./datasets/data/{}.jpg".format(idx))
        if self.transform:
            img = self.transform(img)

        if self.opt.image_model ==  "beit" :
            img = self.feature_extractor(images=img, return_tensors="pt")
            
        txt = ""
        if not self.mask_text:
            with open(r'./datasets/data/{}.txt'.format(idx), 'rb') as f:
                for line in f:
                    txt += line.decode("utf-8", "ignore")
                f.close()
        txt = self.tokenizer(txt, padding=True, truncation=True, max_length=self.opt.max_length)

        if len(txt['input_ids']) < self.opt.max_length:
            txt['input_ids'] = txt['input_ids'] + [0] * (self.opt.max_length - len(txt['input_ids']))
            txt['attention_mask'] = [1] * len(txt['attention_mask']) + [0] * (self.opt.max_length - len(txt['attention_mask']))
            txt['token_type_ids'] = txt['token_type_ids'] + ([0] * (self.opt.max_length - len(txt['token_type_ids'])))

        txt['input_ids'] = np.array(txt['input_ids'])
        txt['attention_mask'] = np.array(txt['attention_mask'])
        txt['token_type_ids'] = np.array(txt['token_type_ids'])

        # print(len(txt['input_ids']), len(txt['attention_mask']), len(txt['token_type_ids']))

        # new_txt = {}
        # new_txt['input_ids'] = np.array(txt['input_ids'])
        # new_txt['attention_mask'] = np.array(txt['attention_mask'])
        # txt = new_txt

        if self.train:
            return img, txt, self.label[index]
        else:
            return img, txt

    def __len__(self):
        return len(self.index)


def get_test_dataset(opt, tokenizer, transform=None):
    with open(test_label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        index = []
        label = []
        for line in lines:
            group = line.rstrip('\n').split(",")
            index.append(int(group[0]))
            label.append(get_label(group[1]))
        dataset = MyDataset(opt, index, label, False, tokenizer, transform)
        f.close()
        return dataset


def get_train_dataset(opt, tokenizer, transform=None):
    with open(train_label_path, 'r', encoding='utf-8') as f:
        s = [0,0,0]
        lines = f.readlines()[1:]
        index = []
        label = []
        for line in lines:
            group = line.rstrip('\n').split(",")
            # if group[1] == "positive" and np.random.rand(1)[0] > 0.5:
            #     continue
            index.append(int(group[0]))
            label.append(get_label(group[1]))
            s[get_label(group[1])] += 1
        print(s)
        dataset = MyDataset(opt, index, label, True, tokenizer, transform)
        f.close()
        return dataset

    
def get_label(label:str):
    if label == "null":
        return NULL
    elif label == "positive":
        return POSITIVE
    elif label == "neutral":
        return NEUTRAL
    elif label == "negative":
        return NEGATIVE
    else:
        raise "Invalid label: {}".format(label)

def get_label_name(label:int):
    if label == NULL:
        return "null"
    elif label == POSITIVE:
        return "positive"
    elif label == NEUTRAL:
        return "neutral"
    elif label == NEGATIVE:
        return "negative"
    else:
        raise "Invalid label: {}".format(label)

def save_label_to_file(labels, path):
    with open(test_label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        index = []
        label = []
        for line in lines:
            group = line.rstrip('\n').split(",")
            index.append(int(group[0]))
        f.close()

    with open(path, 'w') as f:
        f.write("guid,tag\n")
        for i, idx in enumerate(index):
            f.write("{},{}\n".format(idx, get_label_name(labels[i])))    
