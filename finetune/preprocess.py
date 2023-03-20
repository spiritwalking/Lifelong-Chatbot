from transformers import BertTokenizerFast
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import json
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
    parser.add_argument('--train_path', default='data/data.json', type=str, help='训练数据存放位置')
    parser.add_argument('--save_folder', default='data', type=str, help='训练数据tokenized后的存放文件夹')
    args = parser.parse_args()

    # 初始化日志对象
    logger = create_logger(args.log_path)
    folder = args.save_folder

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path)
    logger.info("preprocessing data, data path:{}, save folder:{}".format(args.train_path, folder))
    # tokenize数据集
    dialogue_list, dialogue_len = process_cl_dataset(args, tokenizer, logger)

    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)

    topic2id = {'体育': 0, '科技': 1, '教育': 2, '旅行': 3, '电影': 4, '音乐': 5}
    for topic in topic2id:
        train_file = '/task' + str(topic2id[topic]) + "_train.pkl"
        test_file = '/task' + str(topic2id[topic]) + "_test.pkl"
        with open(folder + train_file, 'wb') as f_train, open(folder + test_file, 'wb') as f_test:
            train_len = int(0.8 * len(dialogue_list[topic]))
            logger.info(f"task {topic} with train len:{train_len} and val len:{len(dialogue_list[topic]) - train_len}")
            pickle.dump(dialogue_list[topic][:train_len], f_train)
            pickle.dump(dialogue_list[topic][train_len:], f_test)

    logger.info("finish preprocessing data,the result is stored in folder: {}".format(folder))
    logger.info("mean of dialogue len:{},median of dialogue len:{},max len:{}".format(len_mean, len_median, len_max))


def process_cl_dataset(args, tokenizer, logger):
    dialogue_len = []
    dialogue_list = {'体育': [], '科技': [], '教育': [], '旅行': [], '电影': [], '音乐': []}

    with open(args.train_path, 'r', encoding='utf-8') as f:
        dialogs = json.load(f)
        logger.info("there are {} dialogs in Continual Learning dataset".format(len(dialogs)))
        for dialog in tqdm(dialogs):
            input_ids1 = get_input_ids(dialog['text'][:10], tokenizer)
            input_ids2 = get_input_ids(dialog['text'][10:], tokenizer)

            dialogue_len.extend([len(input_ids1), len(input_ids2)])
            dialogue_list[dialog['topic']].extend([input_ids1, input_ids2])

    return dialogue_list, dialogue_len


def get_input_ids(dialog, tokenizer):
    input_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
    for utterance in dialog:
        input_ids.extend(tokenizer.encode(utterance, add_special_tokens=False))  # 不自动添加[CLS], [SEP]等special token
        input_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
    return input_ids


if __name__ == '__main__':
    preprocess()
