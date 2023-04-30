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
from my_data_loader import get_task_dataloaders
from ewc import EWC
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
set_seed(42)
warnings.filterwarnings("ignore")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', default='../my_tokenizer', type=str, help='tokenizer路径')
    parser.add_argument('--model_path', default='../from_scratch/gpt-2-multi-large/checkpoint-470000', type=str,
                        help='预训练的模型的路径')
    parser.add_argument('--save_path', default='ewc_model', type=str, help='模型保存路径')
    parser.add_argument('--train_folder', default='tokenized-data', type=str, help='训练语料路径')
    parser.add_argument('--log_path', default='logs/ewc.log', type=str, help='训练日志存放位置')
    parser.add_argument('--epochs', default=5, type=int, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=8, type=int, help='训练的batch size')
    parser.add_argument('--lr', default=2e-5, type=float, help='学习率')
    parser.add_argument('--log_step', default=32, type=int, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--ewc_lambda', type=int, default=0, help="正则项系数")
    args = parser.parse_args()
    return args


def train_epoch(model, train_dataloader, optimizer, logger, epoch, args, ewc_object):
    model.train()
    epoch_loss = 0  # 记录下整个epoch的loss的总和
    epoch_ewc_loss = []  # 记录整个epoch的ewc的loss

    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        # 捕获cuda out of memory exception
        try:
            for key in batch:
                batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss.mean()
            ewc_loss = ewc_object.penalty(model)
            loss += args.ewc_lambda * ewc_loss

            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_ewc_loss.append(ewc_loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 反向传播，累积梯度
            loss.backward()
            # 进行一定step的梯度累积之后，梯度下降更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info("batch {} of epoch {}, loss {:.6}".format(batch_idx + 1, epoch + 1, batch_loss))

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
    epoch_mean_ewc_loss = sum(epoch_ewc_loss) / len(epoch_ewc_loss)
    logger.info("epoch {} training finished: loss {:.6}, ewc loss {:.6}".format(epoch + 1, epoch_mean_loss, epoch_mean_ewc_loss))


def validate_epoch(model, validation_dataloader, logger, args):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dataloader):
            try:
                for key in batch:
                    batch[key] = batch[key].cuda()
                outputs = model(**batch)
                loss = outputs.loss.mean()
                epoch_loss += loss.item()

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logger.info("WARNING: ran out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception

        # 记录当前epoch的平均loss
        epoch_mean_loss = epoch_loss / len(validation_dataloader)
        return epoch_mean_loss


def train_task(model, logger, train_dataloader, val_dataloaders, args, ewc_object):
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr)
    logger.info(f'starting training with total steps: {total_steps}')

    for epoch in range(args.epochs):
        validate_losses = []
        # train
        train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer, logger=logger, epoch=epoch,
                    args=args, ewc_object=ewc_object)

        # validate
        for val_dataloader in val_dataloaders:
            validate_loss = validate_epoch(model=model, validation_dataloader=val_dataloader, logger=logger, args=args)
            validate_losses.append(validate_loss)
        logger.info("epoch {} validation finished: loss {}".format(epoch + 1, validate_losses))

    logger.info('finished training task{}, model saved'.format(args.current_task))
    model_path = join(args.save_path, 'task{}'.format(args.current_task))
    save_model(model_path, model)

    args.current_task += 1
    return model


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
    train_dataloaders, validation_dataloaders = get_task_dataloaders(tokenizer, args)
    args.current_task = 0
    ewc_object = EWC(model=model, dataloaders=[])

    for i in range(5):
        logger.info(f"===================================task{i}===================================")
        logger.info(f"training on task{i}")
        # 加载训练集和验证集
        model = train_task(model, logger, train_dataloaders[i], validation_dataloaders[:i + 1], args, ewc_object)
        ewc_object = EWC(model=model, dataloaders=validation_dataloaders[:i + 1])


if __name__ == '__main__':
    main()
