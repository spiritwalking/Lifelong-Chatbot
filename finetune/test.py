import torch
import os
import torch.nn as nn
import evaluate
import numpy as np
from utils import create_logger
from transformers import GPT2LMHeadModel, BertTokenizerFast, set_seed
from tqdm import tqdm
from datasets import load_from_disk
from data_loader import get_task_dataloaders
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
set_seed(42)
warnings.filterwarnings("ignore")


def test_loss(model, validation_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(validation_loader)):
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss.mean()
            epoch_loss += loss.item()

        # 记录当前epoch的平均loss
        epoch_mean_loss = epoch_loss / len(validation_loader)
        return epoch_mean_loss


def test_bleu(model, valid_set):
    model.eval()
    bleu = evaluate.load('sacrebleu')
    with torch.no_grad():
        for dialog in tqdm(valid_set['dialog']):
            dialog_len = len(dialog) // 2 * 2
            for i in range(0, dialog_len, 2):
                history = dialog[:i + 1]
                reference = dialog[i + 1]



    device = args.device
    model.eval()
    bleu = evaluate.load('sacrebleu')

    with torch.no_grad():
        for history, reference in tqdm(validation_loader):
            responce, _ = chat(model, tokenizer, history, device=device, is_tokenized=True)
            reference = tokenizer.decode(reference.squeeze()).replace(" ", "")
            bleu.add_batch(predictions=[responce], references=[reference])

        bleu_score = bleu.compute(tokenize='zh')
        logger.info(bleu_score)


def main():
    logger = create_logger("logs/evaluate.log")
    checkpoint = "models/ncl_model/task4"
    tokenizer = BertTokenizerFast.from_pretrained('../my_tokenizer')

    _, val_dataloaders = get_task_dataloaders(tokenizer, 8)

    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    model = nn.DataParallel(model).cuda()

    # test loss of the model on each validation dataloader
    loss_list = []
    for val_dataloader in val_dataloaders:
        loss = test_loss(model, val_dataloader)
        loss_list.append(loss)
    print(loss_list)

    dataset = load_from_disk("../from_scratch/tokenized-multi-large")
    test_bleu(model, dataset['test'])


if __name__ == '__main__':
    main()
