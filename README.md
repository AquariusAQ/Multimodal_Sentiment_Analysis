# Multimodal_Sentiment_Analysis
本项目基于配对的文本和图像，预测对应的情感标签（positive, neutral, negative），模型包括：

- BERT + X (VGG/ResNet/Visual Transformers/BEiT) + MultiheadAttention
- ELECTRA-discriminator + X (VGG/ResNet/Visual Transformers/BEiT) + MultiheadAttention

本项目是当代人工智能课程实验五代码仓库。

## Dataset

数据集来源未知，随实验布置发放，上传至仓库的 datasets 文件夹下。

## Model

本项目实现了文本+图像的多模态情感分析模型，主要内容为：

- 文本特征：
  - 使用 BERT 或 ELECTRA-discriminator 提取文本特征
  - 依据使用预训练模型不同，也可以使用其他 NLP 模型提取文本特征
- 图像特征：
  - VGG16
  - ResNet18 / ResNet50
  - Visual Transformers
  - BEiT
- 特征融合：
  - 使用多头注意力机制，将图像特征序列或向量作为 query 在文本特征上进行搜索，然后使用全连接层输出到标签。

## Requirements

本实现基于 Python3. 运行代码需要的依赖库如下：

- torch==1.11.0
- torchvision==0.12.0
- transformers==4.18.0
- numpy==1.21.5
- einops==0.4.1

你可以运行以下指令安装所有环境：

```python
pip install -r requirements.txt
```

## Repository structure
部分重要文件描述

```python
|-- datasets # 完整数据集
    |-- data # 包含互相对应的图片和文本数据
    |-- train.txt # 训练集样本的 ID 和标签、
    |-- test_without_label.txt # 测试集样本的 ID
|-- model # 实验模型文件夹
    |-- MSA.py # 多模态情感分析模型
    |-- text_bert.py # 提取文本特征的子模型
    |-- image_vgg.py # 使用 VGG 提取图像特征的子模型
    |-- image_resnet.py # 使用 ResNet 提取图像特征的子模型
    |-- image_vit.py # 使用 Visual Transformers 提取图像特征的子模型
    |-- image_beit.py # 使用 BEiT 提取图像特征的子模型
    |-- utils.py # 一些辅助性的子模型
|-- pre-trained-model # 预训练模型文件夹，如 BERT 等
|-- output # 输出文件夹，包括训练好的模型参数、预测结果等
|-- train.py # 模型训练脚本
|-- test.py # 模型测试脚本
|-- dataset.py # 读取数据集的脚本
|-- utils.py # 包含一些常用函数的脚本
```

## Pretrained Model Required

需要从 🤗Huggingface 下载模型需要的预训练模型，并保存到 pre-trained-model 文件夹下，如：

- [bert-medium](https://huggingface.co/prajjwal1/bert-medium)
- [bert-base-uncased](https://huggingface.co/bert-base-uncased)
- [electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator)
- [beit-base-patch16-224-pt22k-ft22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)

## Results

各个模型在数据验证集上的结果（准确率）如下表所示：

> 所有模型的特征融合模型均为多头注意力机制

| Model | Accuracy |
| ----- | -------- |
| BERT-medium + VGG16 | 70.8%
| BERT-medium + ResNet18 | 71.0%
| BERT-medium + ViT | 69.9%
| BERT-medium + BEiT-base | 71.2%
| ELECTRA-small-discriminator + VGG16 | 64.7%
| ELECTRA-small-discriminator + ResNet18 | 69.0%
| ELECTRA-small-discriminator + ViT | 65.2%
| ELECTRA-small-discriminator + BEiT-base | 67.2%
| BERT-base-uncased + ViT | 71.3%
| BERT-base-uncased + BEiT-base | **72.1%**

- 消融实验
分别 mask 文本或图像（输入空字符串或空白图像），使用 BERT-base-uncased + BEiT-base 模型分别仅对图像或文本进行训练，验证集准确率如下：

| Model | Accuracy |
| ----- | -------- |
| BERT-base-uncased + BEiT-base (mask text) | 57.6%
| BERT-base-uncased + BEiT-base (mask image) | 70.6%


## Quick Start

可以通过以下命令对模型进行训练（BERT-base-uncased + BEiT-base），并保存结果

```shell
python .\train.py 
  --epoch 10 --lr 2e-5 --batch_size 8 --l2 1e-6 
  --scheduler --lr_step 3 --lr_gamma 0.1 
  --model_path .\pre-trained-model\bert-base-uncased\ 
  --image_model beit 
  --image_model_path .\pre-trained-model\beit-base-patch16-224-pt22k-ft22k\ 
  --save_model --cuda

```

## Parameter Setting

训练主要有以下参数：

- valid_ratio：在训练集中划分验证集的比例
- mask_image：是否 mask 图片（即输入空白图片）
- mask_text：是否 mask 文本（即输入空白文本）
- model_path：NLP 预训练模型的路径
- image_model：图像特征提取模型类型，可选的包括 vgg, resnet18, resnet50, vit, beit 等
- image_model_path：如果使用 beit，其预训练模型的路径
- epoch：训练轮数
- batch_size：批大小
- lr：学习率
- scheduler：是否使用学习率计划（StepLR）
- save_model：是否保存模型参数等训练结果
- size：图像 reshape 后的大小
- max_length：文本长度
- cuda：是否使用 GPU 训练

## Run pipeline
1. 可以通过以下命令加载训练好的模型，输出测试集的预测结果

```shell
python .\test.py 
  --save_model_path ./output/2022-07-12-11-43-25 
  --batch_size 8 
  --cuda
```

2. 可以通过以下命令使用某一个图片及文本做预测
  
```shell
python .\test.py 
  --pipeline 
  --save_model_path ./output/2022-07-12-11-43-25 
  --image_path .\datasets\data\1.jpg 
  --text_path .\datasets\data\1.txt 
```

## Attribution

部分代码来源于以下仓库/博客:

- [ViT](https://github.com/tahmid0007/VisualTransformers/blob/main/ResViT.py)

- [模型参数量分析](https://blog.csdn.net/qq_33757398/article/details/109210240)
