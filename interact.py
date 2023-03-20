import torch
import os
import argparse
from datetime import datetime
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
from utils import fix_seed
# from chatbot.model import DialogueGPT2Model
import torch.nn.functional as F
import gradio as gr


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, help='最高k选1')
    parser.add_argument('--topp', default=0.5, type=float, help='最高积累概率')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='选择词库')
    parser.add_argument('--model_path', default='finetune/model/epoch30', type=str, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help="重复惩罚参数，值越大生成的回复的重复性越低")
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()最大的top_k个元素，返回值为values,indices
        top_k_value, _ = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_value[-1]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        token_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(token_probs, dim=-1)  # 计算累积概率

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    args = set_args()
    fix_seed(42)
    device = args.device
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    # 保存聊天记录
    if not os.path.exists(args.save_samples_path):
        os.makedirs(args.save_samples_path)
    samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
    samples_file.write("聊天记录{}:\n".format(datetime.now()))

    # 聊天历史 [[uttr1_ids], [uttr2_ids]...]
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            imput_text = input("user:")
            samples_file.write("user:{}\n".format(imput_text))
            text_ids = tokenizer.encode(imput_text, add_special_tokens=False)
            history.append(text_ids)

            # 组装输入语句
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
            for history_utr in history[-args.max_history_len:]:
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            input_ids = input_ids.unsqueeze(0)  # [[CLS, id0, id1... SEP]]
            response = []

            # 逐个生成tokens
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]

                for id in set(response):  # 对于已生成的结果中的每个token添加一个重复惩罚项，降低其生成概率
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature  # temperature越高，生成的文本的多样性和创造性越高
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')  # 将[UNK]的概率设为无穷小

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                token_probs = F.softmax(filtered_logits, dim=-1)

                # 根据概率从数组中采样，抽取1个元素，返回元素的下标
                next_token = torch.multinomial(token_probs, num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则生成结束
                    break
                else:
                    response.append(next_token.item())
                    input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)  # 把生成的token加入input_ids

            history.append(response)
            response_tokens = tokenizer.convert_ids_to_tokens(response)
            response_text = "".join(response_tokens)

            print("chatbot:" + response_text)
            samples_file.write("chatbot:{}\n".format(response_text))

        except KeyboardInterrupt:
            samples_file.close()
            break


if __name__ == '__main__':
    main()
