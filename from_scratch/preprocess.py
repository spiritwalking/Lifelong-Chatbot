from transformers import BertTokenizerFast
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import sys

sys.path.append("..")
from utils import create_logger


def preprocess():
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='../vocab/vocab.txt', type=str, help='词表路径')
    parser.add_argument('--log_path', default='logs/preprocess.log', type=str, help='预处理日志存放位置')
    parser.add_argument('--train_path', default='data/train.txt', type=str, help='训练数据存放位置')
    parser.add_argument('--save_path', default='data/a_train.pkl', type=str, help='训练数据tokenized后的存放位置')
    args = parser.parse_args()

    # 初始化日志对象
    logger = create_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path)
    logger.info("preprocessing data, data path:{}, save path:{}".format(args.train_path, args.save_path))
    # tokenize数据集
    dialogue_list, dialogue_len = process_general_dataset(args, tokenizer, logger)

    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)
    with open(args.save_path, "wb") as f:
        pickle.dump(dialogue_list, f)

    logger.info("finish preprocessing data,the result is stored in {}".format(args.save_path))
    logger.info("mean of dialogue len:{},median of dialogue len:{},max len:{}".format(len_mean, len_median, len_max))


def process_general_dataset(args, tokenizer, logger):
    with open(args.train_path, 'r', encoding='utf-8') as f:
        data = f.read()
    train_data = data.split("\n\n")
    logger.info("there are {} dialogs in general dataset".format(len(train_data)))

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]..."
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    with open(args.save_path, "w", encoding="utf-8") as f:
        for dialogue in tqdm(train_data):
            utterances = dialogue.split("\n")

            input_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances:
                input_ids.extend(
                    tokenizer.encode(utterance, add_special_tokens=False))  # tokenizer不自动添加[CLS], [SEP]等special token
                input_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束

            dialogue_len.append(len(input_ids))
            dialogue_list.append(input_ids)
    return dialogue_list, dialogue_len


if __name__ == '__main__':
    preprocess()
