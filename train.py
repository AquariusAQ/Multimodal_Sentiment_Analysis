import os
import time
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import ElectraConfig, ElectraTokenizer, ElectraModel
from transformers import BeitFeatureExtractor, BeitForImageClassification, BeitModel
from transformers import logging
 
logging.set_verbosity_warning()

import datetime 
import argparse
from dataset import get_train_dataset, get_test_dataset
from utils import check_accuracy, model_structure
from model.MSA import Model

# args
parser = argparse.ArgumentParser()

parser.add_argument('--valid_ratio', type=float, default=0.2, help='proportion of validation set to full train set')

parser.add_argument('--mask_image',  action='store_true', help='mask image input')
parser.add_argument('--mask_text',  action='store_true', help='mask text input')

parser.add_argument('--model_path',  type=str, default="./pre-trained-model/bert-medium", help='pre-trained NLP model')
parser.add_argument('--image_model',  type=str, default="resnet18", help='valid model: vgg, resnet18, resnet50, vit, beit')
parser.add_argument('--image_model_path',  type=str, default="microsoft/beit-base-patch16-224-pt22k-ft22k", help='pre-trained CV model')

parser.add_argument('--epoch', type=int, default=1, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--l2', type=float, default=0.0, help='dropout rate')
parser.add_argument('--scheduler',  action='store_true', help='use lr scheduler')
parser.add_argument('--lr_step',  type=int, default=1, help='lr scheduler step size')
parser.add_argument('--lr_gamma',  type=float, default=0.1, help='lr scheduler decline gamma')

parser.add_argument('--save_model',  action='store_true', help='save model parameters')

parser.add_argument('--size', type=int, default=128, help='size of the image')
parser.add_argument('--max_length', type=int, default=144, help='max length of the text')

parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

# save opt
if not os.path.exists("./output"):
    os.makedirs("./output")
time_str = datetime.datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')
if opt.save_model:
    output_path = os.path.join("./output/", time_str)
    os.makedirs(output_path)
    with open(os.path.join(output_path, "opt.json"), 'wt') as f:
        json.dump(vars(opt), f, indent=4)
        f.close()

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


# ./pre-trained-model/bert-base-uncased
# bert-base-uncased


model = Model(opt, model_config, tokenizer=tokenizer, encoder=model_encoder, device=device)
model.to(device)


model_structure(model)



# loss function
loss_function = torch.nn.CrossEntropyLoss()


# optimizer
no_decay = ['bias', 'LayerNorm.weight'] # 不需要 L2 正则化的参数
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.l2, eps = 1e-8)

# scheduler
if opt.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma, last_epoch=-1, verbose=False)


# dataset
transform = T.Compose([
                T.Resize((opt.size, opt.size)),
                T.ToTensor(),
            ])
train_dataset = get_train_dataset(opt, tokenizer=tokenizer, transform=transform)
NUM_ALL = train_dataset.__len__()
NUM_VALID = int(NUM_ALL * opt.valid_ratio)
NUM_TRAIN = NUM_ALL - NUM_VALID

if opt.image_model == "vit":
    NUM_VALID = NUM_VALID // opt.batch_size * opt.batch_size
    NUM_TRAIN = NUM_TRAIN // opt.batch_size * opt.batch_size
    # print(NUM_VALID, NUM_TRAIN)

loader_train = DataLoader(train_dataset, batch_size=opt.batch_size,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
if NUM_VALID != 0:
    loader_val = DataLoader(train_dataset, batch_size=opt.batch_size,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN+NUM_VALID)))


print("-> Training model with {} epochs.".format(opt.epoch))
print("-> Using", device)

loss_list = []
acc_train_list = []
acc_list = []
acc_max = 0
for e in range(opt.epoch):
    learning_rate_temp = optimizer.state_dict()['param_groups'][0]['lr']
    print('\tEpochs {}, lr = {} '.format(e+1, learning_rate_temp), end="")
    start_time = time.time()

    num_correct = 0
    num_samples = 0
    for t, (image, text, y) in enumerate(loader_train):

        model.train()
        image = image.to(device)#, dtype=torch.float32)
        text = text.to(device)
        y = y.to(device, dtype=torch.long)
        scores = model(image, text)
        # print(scores, y)
        loss = loss_function(scores, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    
    acc = float(num_correct) / num_samples

    loss_list.append(loss.item())
    print('loss = {:.4f}'.format(loss.item()))
    acc_train_list.append(acc)
    print('\t\tAcc on data_train: {:.2f}%'.format(acc * 100))

    if NUM_VALID != 0:
        acc_val = check_accuracy(opt, loader_val, model, device)
        acc_list.append(acc_val)
        print('\t\tAcc on data_val: {:.2f}%'.format(acc_val * 100))
    
    # torch.save(model.state_dict(), 'output/{}/Epoch-{}.pth'.format(time_str, e))
        if acc_val > acc_max:
            acc_max = acc_val
            if opt.save_model:
                torch.save(model.state_dict(), 'output/{}/Best-model.pth'.format(time_str))
    else:
        if opt.save_model:
            torch.save(model.state_dict(), 'output/{}/Best-model.pth'.format(time_str))

    if opt.scheduler:        
        scheduler.step()

    end_time = time.time()
    print("\t\tUse time: {:.2f} sec".format(end_time-start_time))

result = {}
result['acc_train'] = acc_train_list
if NUM_VALID != 0:
    result["acc_valid"] = acc_list
result["loss_train"] = loss_list
with open(os.path.join(output_path, "train_results.json"), 'wt') as f:
    json.dump(result, f, indent=4)
    f.close()

print("-> Train finished!")
print("\tAcc on train set:", acc_train_list)
if NUM_VALID != 0:
    print("\tAcc on validation set:", acc_list)
print("\tLoss on train set:", loss_list)
if NUM_VALID != 0:
    print("-> Best Acc: {:.2f}%".format(acc_max * 100))
if opt.save_model:
    print("-> Model parameters saved to " + './output/{}/Best-model.pth'.format(time_str))