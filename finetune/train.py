import argparse
import torch
import torch.optim as optim
from datetime import datetime
import os
from os.path import join, exists
import torch.nn as nn
import transformers
import sys
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
from data_loader import get_training_loader

sys.path.append("..")
from utils import EarlyStopping, create_logger, collate_fn, save_model, fix_seed, calculate_acc


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_ids', default='0,1', type=str, help='设置使用哪些显卡')
    parser.add_argument('--vocab_path', default='../vocab/vocab.txt', type=str, help='词表路径')
    parser.add_argument('--train_folder', default='data', type=str, help='训练集路径')
    parser.add_argument('--max_len', default=300, type=int, help='训练时，输入数据的最大长度')
    parser.add_argument('--log_path', default='logs/train.log', type=str, help='训练日志存放位置')
    parser.add_argument('--ignore_index', default=-100, type=int, help='对于ignore_index的label token不计算梯度')
    # parser.add_argument('--input_len', default=200, type=int, help='输入的长度')
    parser.add_argument('--epochs', default=30, type=int, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=8, type=int, help='训练的batch size')
    parser.add_argument('--lr', default=2.6e-5, type=float, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, help='衰减率')
    parser.add_argument('--log_step', default=100, type=int, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='设置最大梯度范数')
    parser.add_argument('--save_model_path', default='model', type=str, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='model/model_epoch40_50w', type=str, help='预训练的模型的路径')
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping")
    args = parser.parse_args()
    return args


def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, args):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, labels=labels)  # TODO: 为什么input和label是这样的？？？
            logits = outputs.logits
            loss = outputs.loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 反向传播，累积梯度
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 防止梯度爆炸

            # 进行一定step的梯度累积之后，梯度下降更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info("batch {} of epoch {}, loss {:.6}, batch_acc {:.6}, lr {}".format(
                    batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps,
                    batch_acc, scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info("epoch {}: loss {:.6}, predict_acc {:.6}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
    save_model(model_path, model)

    epoch_finish_time = datetime.now()
    logger.info('epoch {} finished, spend time: {}'.format(epoch + 1, epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validation_dataloader, logger, epoch, args):
    logger.info("start validating")
    model.eval()
    device = args.device
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validation_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validation_dataloader)
            logger.info("validate epoch {}: loss {}".format(epoch + 1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
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
    early_stopping = EarlyStopping(args.patience, save_path=args.save_model_path)
    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps
    )
    logger.info(f'using linear scheduler with num_warmup_steps: {int(total_steps*0.1)}')
    logger.info('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(model=model, validation_dataloader=validation_dataloader, logger=logger,
                                       epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
            model_path = join(args.save_model_path, 'min_ppl_model'.format(epoch + 1))
            save_model(model_path, model)

        if args.patience == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def main():
    args = set_args()
    logger = create_logger(args.log_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids  # 设置可见GPU
    fix_seed(42)

    # 创建模型的输出目录
    if not exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    # 设置使用哪些显卡进行训练
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置训练主设备
    logger.info('using device: {}'.format(args.device))

    # 加载预训练模型
    logger.info('fine-tune pretrained model')
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size

    # 并行训练模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info("training with DataParallel on cuda " + args.device_ids)
    model = model.to(args.device)

    # 计算模型参数数量
    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    train_dataloader, validate_dataloader = get_training_loader(args, collate_fn, logger)

    train_model(model, logger, train_dataloader, validate_dataloader, args)


if __name__ == '__main__':
    main()
