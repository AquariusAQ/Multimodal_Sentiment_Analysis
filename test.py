
# import os
# d = {}
# for idx in range(1, 5130):
#     txt = ""
#     if os.path.exists(r'./datasets/data/{}.txt'.format(idx)):
#         with open(r'./datasets/data/{}.txt'.format(idx), 'rb') as f:
#             for line in f:
#                 txt += line.decode("utf-8", "ignore")
#             f.close()
#         d[len(txt)] = d.get(len(txt), 0) + 1
# print(sorted(d.items()))

from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model_path =  "./pre-trained-model/beit-base-patch16-224-pt22k-ft22k"
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = BeitFeatureExtractor.from_pretrained(model_path)
model = BeitForImageClassification.from_pretrained(model_path)
inputs = feature_extractor(images=image, return_tensors="pt")
# print(**inputs)
# outputs = model(**inputs)
outputs = model(inputs['pixel_values'])
logits = outputs.logits
# model predicts one of the 21,841 ImageNet-22k classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
    