import torch
from transformers import GPT2LMHeadModel, BertTokenizerFast, set_seed
import torch.nn.functional as F

multi_bot = "from_scratch/gpt-2-multi-large/checkpoint-470000"
upper_bot = "finetune/models/upperbound_model"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained(upper_bot).to(device)
tokenizer = BertTokenizerFast.from_pretrained("my_tokenizer")

speaker1_id, speaker2_id = tokenizer.additional_special_tokens_ids
cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id


def topp_filtering(logits, top_p, filter_value=-float('Inf')):
    if top_p < 0 or top_p > 1:
        raise ValueError("top_p must between 0 and 1")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    token_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(token_probs, dim=-1)  # 计算累积概率

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right, to prevent the first token from being removed
    sorted_indices_to_remove = sorted_indices_to_remove.roll(shifts=1, dims=0)
    sorted_indices_to_remove[0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value

    return logits


def update_prompt(prompt, next_token_id, next_speaker_id):
    prompt['input_ids'] = torch.cat((prompt['input_ids'], torch.tensor([[next_token_id]], device=device)), dim=1)
    prompt['token_type_ids'] = torch.cat((prompt['token_type_ids'], torch.tensor([[next_speaker_id]], device=device)), dim=1)
    return prompt


def build_input(history, max_history_len):
    prompt = {}
    input_ids = [cls]
    speaker_ids = [cls]

    for prev_query, prev_response in history[-max_history_len:]:
        prev_query = [speaker1_id] + tokenizer.encode(prev_query, add_special_tokens=False) + [sep]
        input_ids.extend(prev_query)
        speaker_ids.extend([speaker1_id] * len(prev_query))
        if prev_response is not None:
            prev_response = [speaker2_id] + tokenizer.encode(prev_response, add_special_tokens=False) + [sep]
            input_ids.extend(prev_response)
            speaker_ids.extend([speaker2_id] * len(prev_response))

    prompt["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [[CLS, s1, id0, id1... SEP]]
    prompt["token_type_ids"] = torch.tensor(speaker_ids, dtype=torch.long, device=device).unsqueeze(0)
    return prompt


def chat(history, chat_model=model, top_p=0.9, temperature=0.7, repetition_penalty=1.0, max_history_len=2, max_new_tokens=50):
    prompt = build_input(history, max_history_len)
    prompt = update_prompt(prompt, speaker2_id, speaker2_id)
    response = []
    for _ in range(max_new_tokens):
        outputs = chat_model(**prompt)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]

        for token in set(response):  # 对于已生成的结果中的每个token添加一个重复惩罚项，降低其生成概率
            next_token_logits[token] /= repetition_penalty

        next_token_logits = next_token_logits / temperature
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')  # 将[UNK]的概率设为无穷小

        filtered_logits = topp_filtering(next_token_logits, top_p=top_p)
        token_probs = F.softmax(filtered_logits, dim=-1)

        # 根据概率从数组中采样，抽取1个元素，返回元素的下标
        next_token = torch.multinomial(token_probs, num_samples=1).item()
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则生成结束
            break
        else:
            response.append(next_token)
            prompt = update_prompt(prompt, next_token, speaker2_id)  # 更新prompt

    response_tokens = tokenizer.convert_ids_to_tokens(response)
    response_text = "".join(response_tokens)
    history[-1][1] = response_text
    return response_text, history


def main():
    history = []
    print('开始和chatbot聊天')

    while True:
        input_text = input("user:")
        history.append([input_text, None])

        response, history = chat(history)
        print("chatbot:" + response)


if __name__ == "__main__":
    main()
