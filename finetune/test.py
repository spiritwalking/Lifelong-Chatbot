import evaluate
import torch
import argparse
import sys
from data_loader import get_dataloader
from transformers import GPT2LMHeadModel, BertTokenizerFast
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sys.path.append("..")
from utils import fix_seed, create_logger, collate_fn


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/model_epoch40_50w', help='被测模型路径')
    parser.add_argument('--task_id', type=int, default=0, help="任务编号")
    parser.add_argument('--vocab_path', default='../vocab/vocab.txt', type=str, help='选择词库')
    parser.add_argument('--log_path', default='logs/test.log', type=str, help='测试日志存放位置')
    parser.add_argument('--device', type=str, default='cuda:0', help="设备")
    parser.add_argument('--train_folder', default='data', type=str, help='训练集路径')
    parser.add_argument('--max_len', default=250, type=int, help='训练时，输入数据的最大长度')
    parser.add_argument('--batch_size', default=1, type=int, help='训练的batch size')
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader加载数据时使用的线程数量")
    return parser.parse_args()


def eval(model, tokenizer, args, logger, validate_dataloader):
    device = args.device
    model.eval()

    bleu = evaluate.load('sacrebleu')

    with torch.no_grad():
        for input_ids, labels in tqdm(validate_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            indices = torch.where(input_ids == 102)
            if len(indices[1])<2:
                continue

            input_ids = input_ids[:, :indices[1][-2]+1]
            labels = labels[:, indices[1][-2]+1:]

            generated_tokens = model.generate(input_ids, max_new_tokens=50, eos_token_id=tokenizer.sep_token_id)
            generated_tokens = generated_tokens[:, len(input_ids[0]):]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            bleu.add_batch(predictions=decoded_preds, references=decoded_labels)

        bleu_score = bleu.compute()
        logger.info(bleu_score)


def main():
    args = set_args()
    logger = create_logger(args.log_path)
    logger.info(args.model)
    fix_seed(42)

    _, validate_dataloader = get_dataloader(args, collate_fn, logger, task_id=args.task_id)

    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)
    eval(model, tokenizer, args, logger, validate_dataloader)


if __name__ == "__main__":
    main()

