import argparse
import torch
import os
from os.path import join, exists
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, BertTokenizerFast, set_seed
from data_loader import get_training_loader
from tqdm import tqdm
from utils import create_logger, save_model
from my_data_loader import get_task_dataloaders
from ewc import EWC
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
set_seed(42)
warnings.filterwarnings("ignore")


def filter_loss(model, dataloader):
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)

            logits = outputs.logits[..., :-1, :].contiguous()
            labels = batch['labels'][..., 1:].contiguous()

            real_loss = outputs.loss.mean()
            loss_matrix = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction='none')

            loss = loss_matrix.sum(dim=1)/torch.sum(labels != -100, dim=1)
            loss_list.extend(loss.tolist())

        print(len(loss_list))


def main():
    tokenizer = BertTokenizerFast.from_pretrained('../my_tokenizer')
    model = GPT2LMHeadModel.from_pretrained('models/ewc_model/task0')

    # 并行训练模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    # 记录参数设置
    train_dataloaders, _ = get_task_dataloaders(tokenizer, 8)

    filter_loss(model, train_dataloaders[0])


if __name__ == '__main__':
    main()
