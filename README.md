# SentenceBERT

![image-20220105155954331](https://tva1.sinaimg.cn/large/008i3skNgy1gy2urzphpmj30pe0cq75s.jpg)

### Requirement

- pytorch 1.8.2
- 数据文件需要放在 `./data` 目录下，预训练语言模型需要放在 `./{PLM_NAME}` 目录下，这里使用的是 [MacBERT](https://github.com/ymcui/MacBERT)

### Dataset

数据集来自于该博客中整理的中文竞赛数据集 https://zhuanlan.zhihu.com/p/373253456

### Instruction

- `config.py`

    配置项，可以在里面修改数据所在目录和预训练语言模型所在目录位置

- `dataset.py`

    `torch.utils.data.Dataset  `类，包含训练SentenceBERT和原始BERT需要使用的`Dataset`类

- `model.py`

    包含SentenceBERT和原始BERT

- `run.py`

    训练具体模型的代码

- `train_eval.py`

    训练及评估的utils

