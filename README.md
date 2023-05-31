## 项目介绍

本项目使用GPT-2模型分别进行预训练和微调，目的是探究如何让对话系统在不遗忘已有知识的情况下学到新的知识。

## 项目结构

```
.
├── README.md
├── chitchat
├── finetune
├── from_scratch
├── generate.py
├── infer.py
├── my_tokenizer
├── requirements.txt
└── web_demo.py
```

* `generate.py`：包含对话生成相关算法及命令行交互界面
* `infer.py`：包含测试BLEU值的代码
* `web_demo.py`：基于Gradio搭建的可视化界面
* `my_tokenizer`：包含tokenizer相关文件

### chitchat

基于[**GPT2-chitchat**](https://github.com/yangjianxin1/GPT2-chitchat)项目搭建的对话系统，仅用此项目作为参考，后续并未实际使用。

```
.
├── config
├── data
├── data_loader.py
├── model
├── preprocess.py
├── train.py
├── utils.py
└── vocab
```

* `preprocess.py`：预处理对话语料，将其拼接为`[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]`的形式
* `data_loader.py`：将预处理后等数据构建为pytorch的dataloader
* `utils.py`：包含训练需要的工具函数，例如固定随机种子、保存模型、创建日志等
* `train.py`：使用GPT-2模型在数据集上进行自回归训练
* `model`、`data`、`vocab`分别包含GPT-2模型、训练语料和tokenizer词表

### from_scratch

使用39M条单轮对话与3M条多轮对话**预训练**GPT-2模型。

```
.
├── preprocess.py
├── trainer.py
└── trainer_multi.py
```

* `preprocess.py`：预处理对话语料，将其拼接为`[CLS][speaker1]utterance1[SEP][speaker2]utterance2[SEP]`

* `trainer.py`：在单轮语料上使用🤗Transformers库的Trainer训练对话系统
* `trainer_multi.py`：在单轮语料上训练完毕后，在多轮语料上使用🤗Transformers库的Trainer继续训练对话系统

### finetune

使用涵盖5个领域的对话数据**微调**GPT-2模型，并实现多种持续学习算法。

```
.
├── data_loader.py
├── ewc.py
├── preprocess.py
├── train_ewc.py
├── train_mix.py
├── train_replay.py
├── train_upper_bound.py
└── utils.py
```

* `ewc.py`：实现了EWC算法

* `train_ewc.py`：使用EWC算法做持续学习

* `train_replay.py`：使用重放算法做持续学习
* `train_mix.py`：使用组合策略做持续学习
* `train_upper_bound.py`：使用多任务学习作为持续学习的性能参考上限。