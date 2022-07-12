import os
import time
import json
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T

import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoConfig, BeitFeatureExtractor
from transformers import logging
 
logging.set_verbosity_warning()
from PIL import Image
import datetime 
import argparse
from dataset import get_train_dataset, get_test_dataset, save_label_to_file, get_label_name
from utils import check_accuracy, model_structure
from model.MSA import Model

# args
parser = argparse.ArgumentParser()

parser.add_argument('--save_model_path',  type=str, help='MSA model')

parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')

parser.add_argument('--cuda', action='store_true', help='use GPU computation')

parser.add_argument('--pipeline', action='store_true', help='using certain image and text')
parser.add_argument('--image_path', type=str, default="", help='path of image for pipeline')
parser.add_argument('--text_path', type=str, default="", help='path of text for pipeline')

opt_test = parser.parse_args()

opt = argparse.ArgumentParser()
opt_dict = vars(opt)

with open(os.path.join(opt_test.save_model_path, "opt.json"), 'rt') as f:
    opt_dict.update(json.load(f))
opt_dict.update(vars(opt_test))
if opt.pipeline:
    opt_dict.update({"batch_size": 1})

# print(opt_dict)

# chech mask
assert not (opt.mask_image and  opt.mask_text), "Text and image cannot be masked at the same time!"

# check image model:
valid_image_models = ["resnet18", "resnet50", "vit", "beit", "vgg"]
assert opt.image_model in valid_image_models, "image model must be in " + str(valid_image_models)

# use gpu
if opt.cuda and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# model
model_config = AutoConfig.from_pretrained(opt.model_path)
tokenizer = AutoTokenizer.from_pretrained(opt.model_path, add_special_tokens=False, do_lower_case=True)
model_encoder = AutoModel.from_pretrained(opt.model_path)


model = Model(opt, model_config, tokenizer=tokenizer, encoder=model_encoder, device=device)
model.load_state_dict(torch.load(os.path.join(opt.save_model_path, "Best-model.pth")))
model.to(device)

# dataset
transform = T.Compose([
                T.Resize((opt.size, opt.size)),
                T.ToTensor(),
            ])
test_dataset = get_test_dataset(opt, tokenizer=tokenizer, transform=transform)
NUM_TEST = test_dataset.__len__()

loader_test = DataLoader(test_dataset, batch_size=opt.batch_size,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TEST)))

if opt.pipeline:
    model.eval()
    with torch.no_grad():

        img = Image.open(opt.image_path)
        img = transform(img)

        if opt.image_model ==  "beit" :
            feature_extractor = BeitFeatureExtractor.from_pretrained(opt.image_model_path)
            img = feature_extractor(images=img, return_tensors="pt")
            
        txt = ""
        with open(opt.text_path, 'rb') as f:
            for line in f:
                txt += line.decode("utf-8", "ignore")
            f.close()
        txt_raw = txt
        txt = tokenizer(txt, padding=True, truncation=True, max_length=opt.max_length)

        if len(txt['input_ids']) < opt.max_length:
            txt['input_ids'] = txt['input_ids'] + [0] * (opt.max_length - len(txt['input_ids']))
            txt['attention_mask'] = [1] * len(txt['attention_mask']) + [0] * (opt.max_length - len(txt['attention_mask']))
            txt['token_type_ids'] = txt['token_type_ids'] + ([0] * (opt.max_length - len(txt['token_type_ids'])))

        txt['input_ids'] = torch.from_numpy(np.array([txt['input_ids']]))
        txt['attention_mask'] = torch.from_numpy(np.array([txt['attention_mask']]))
        txt['token_type_ids'] = torch.from_numpy(np.array([txt['token_type_ids']]))

        # transform_txt = T.Compose([T.ToTensor()])
        # txt = transform_txt(txt)

        # image = img.unsqueeze(1)
        # text = txt.unsqueeze(1)
        image = img.to(device)
        text = txt.to(device)
        scores = model(image, text).detach().cpu()
        _, preds = scores.max(1)
    print("-> Image: {}, text: {}".format(opt.image_path, txt_raw))
    print("-> Predicted label:", get_label_name(preds[0]))
else:
    print("-> Testing model.")
    print("-> Using", device)

    pred_label = []
    model.eval()
    with torch.no_grad():
        for t, (image, text) in enumerate(loader_test):

            # print(text['input_ids'])
            image = image.to(device) 
            text = text.to(device)
            scores = model(image, text)
            # print(scores)
            _, preds = scores.max(1)
            for y in preds:
                pred_label.append(y.item())

    # print(pred_label)
    test_result_path = os.path.join(opt.save_model_path, "test_with_pred_label.txt")
    save_label_to_file(pred_label, test_result_path)

    print("-> Test finished!")
    print("-> Predicted label saved to {}".format(test_result_path))