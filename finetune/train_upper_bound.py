import argparse
import torch
import os
from os.path import join, exists
import torch.nn as nn
import transformers
from transformers import GPT2LMHeadModel, BertTokenizerFast, set_seed
from data_loader import get_training_loader
from tqdm import tqdm
from utils import create_logger, save_model
from my_data_loader import get_dataloader
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
set_seed(42)
warnings.filterwarnings("ignore")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', default='../my_tokenizer', type=str, help='tokenizer路径')
    parser.add_argument('--model_path', default='../from_scratch/gpt-2-multi-large/checkpoint-470000', type=str,
                        help='预训练的模型的路径')
    parser.add_argument('--save_path', default='upperbound_model', type=str, help='模型保存路径')
    parser.add_argument('--train_folder', default='tokenized-data', type=str, help='训练语料路径')
    parser.add_argument('--log_path', default='logs/finetune.log', type=str, help='训练日志存放位置')
    parser.add_argument('--epochs', default=10, type=int, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=8, type=int, help='训练的batch size')
    parser.add_argument('--lr', default=3e-5, type=float, help='学习率')
    parser.add_argument('--log_step', default=32, type=int, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    args = parser.parse_args()
    return args


def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, args):
    model.train()
    epoch_loss = 0  # 记录下整个epoch的loss的总和

    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        # 捕获cuda out of memory exception
        try:
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss.mean()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            current_lr = scheduler.get_lr()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 反向传播，累积梯度
            loss.backward()
            # 进行一定step的梯度累积之后，梯度下降更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info("batch {} of epoch {}, loss {:.6}, lr {}".format(
                    batch_idx + 1, epoch + 1, batch_loss, current_lr))

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss
    epoch_mean_loss = epoch_loss / len(train_dataloader)
    logger.info("epoch {}: loss {:.6}".format(epoch + 1, epoch_mean_loss))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(args.save_path, 'epoch{}'.format(epoch + 1))
    save_model(model_path, model)
    return epoch_mean_loss


def validate_epoch(model, validation_dataloader, logger, epoch, args):
    logger.info("start validating")
    model.eval()
    epoch_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(validation_dataloader)):
                for key in batch:
                    batch[key] = batch[key].cuda()
                outputs = model(**batch)
                loss = outputs.loss.mean()
                epoch_loss += loss.item()

            # 记录当前epoch的平均loss
            epoch_mean_loss = epoch_loss / len(validation_dataloader)
            logger.info("validate epoch {}: loss {}".format(epoch + 1, epoch_mean_loss))
            return epoch_mean_loss

    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception


def train_model(model, logger, train_dataloader, validation_dataloader, args):
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=total_steps)
    logger.info(f'starting training with total steps: {total_steps}')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer,
                                 scheduler=scheduler, logger=logger, epoch=epoch, args=args)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(model=model, validation_dataloader=validation_dataloader, logger=logger,
                                       epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model of epoch {}'.format(epoch + 1))
            model_path = join(args.save_path, 'min_ppl_model'.format(epoch + 1))
            save_model(model_path, model)

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def main():
    args = set_args()
    logger = create_logger(args.log_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    # 创建模型的输出目录
    if not exists(args.save_path):
        os.mkdir(args.save_path)

    # 并行训练模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    train_dataloader, valid_dataloader = get_dataloader(tokenizer, args)

    train_model(model, logger, train_dataloader, valid_dataloader, args)


if __name__ == '__main__':
    main()
