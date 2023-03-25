import evaluate
import torch
import argparse
import sys
from transformers import GPT2LMHeadModel, BertTokenizerFast
from tqdm import tqdm
from data_loader import get_validation_loader

sys.path.append("..")
from utils import fix_seed, create_logger
from interact import chat

dialog = [
    "平克·弗洛伊德这个人你听说过吗？",
    "听说过，很有名气的一位乐队的歌手。",
    "他是法国人吗？",
    "不是的，他是英国人。",
    "他有外文名字吗？",
    "他的外文名叫Pink Floyd，你知道他是哪年出道的吗？",
    "他是1967年出道的。",
    "你听过他哪个歌曲呢？",
    "我听过他的歌The Wall，你听过吗？",
    "没有，很好听吗？",
    "是啊，很不错的一首歌。",
    "他唱的歌销量怎么样？",
    "销量很好，2个亿呢。",
    "这个成绩真不错。他的音乐大都什么风格的呀？",
    "他主打摇滚风。",
    "哦，有时间我一定听听他的歌曲。"
]


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/model_epoch40_50w', help='被测模型路径')
    parser.add_argument('--vocab_path', default='../vocab/vocab.txt', type=str, help='选择词库')
    parser.add_argument('--log_path', default='logs/test.log', type=str, help='测试日志存放位置')
    parser.add_argument('--device', type=str, default='cuda:1', help="设备")
    parser.add_argument('--max_len', default=250, type=int, help='训练时，输入数据的最大长度')
    parser.add_argument('--batch_size', default=1, type=int, help='训练的batch size')
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader加载数据时使用的线程数量")
    return parser.parse_args()


def validate_single_dialog(model, tokenizer, args, logger):
    device = args.device
    model.eval()

    bleu = evaluate.load('sacrebleu')

    with torch.no_grad():
        global dialog
        dialog = dialog[:len(dialog) // 2 * 2]  # 保留前偶数条对话
        history = []
        for i in range(0, len(dialog), 2):
            history.append([dialog[i], None])
            reference = dialog[i + 1]

            responce, _ = chat(model, tokenizer, history, device=device)
            history[-1][1] = reference

            bleu.add_batch(predictions=[responce], references=[reference])

        bleu_score = bleu.compute(tokenize='zh')
        logger.info(bleu_score)


def validate_model(model, tokenizer, validation_loader, args, logger):
    logger.info("testing chatbot")
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
    args = set_args()
    logger = create_logger(args.log_path)
    logger.info(args.model)
    fix_seed(42)

    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)

    for i in range(6):
        data_path = "data/task" + str(i) + "_test.pkl"
        logger.info(f"testing data: {data_path}")
        test_loader = get_validation_loader(data_path, max_len=args.max_len, batch_size=1, logger=logger)
        validate_model(model, tokenizer, test_loader, args, logger)


if __name__ == "__main__":
    main()
