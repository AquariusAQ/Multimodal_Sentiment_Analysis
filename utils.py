import torch
import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

def check_accuracy(opt,loader, model:torch.nn.Module, device:str):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        labels = 3
        matrix = np.zeros((labels, labels), dtype=np.int16)
        
        for t, (image, text, y) in enumerate(loader):
            model.eval()

            image = image.to(device) #, dtype=torch.float32)
            text = text.to(device)
            y = y.to(device, dtype=torch.long)
            scores = model(image, text)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            for i, pred in enumerate(preds):
                matrix[pred][y[i]] += 1
        acc = float(num_correct) / num_samples

        print("POSITIVE NEUTRAL NEGATIVE")
        print(matrix)

        return acc


# https://blog.csdn.net/qq_33757398/article/details/109210240
def model_structure(model):
    blank = ' '
    # print('-'*90)
    # print('|'+' '*11+'weight name'+' '*10+'|' \
    #         +' '*15+'weight shape'+' '*15+'|' \
    #         +' '*3+'number'+' '*3+'|')
    # print('-'*90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4
    
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30: 
            key = key + (30-len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40-len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10-len(str_num)) * blank
    
        # print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-'*90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-'*90)
