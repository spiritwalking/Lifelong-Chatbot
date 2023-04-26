import torch
from transformers import GPT2LMHeadModel, BertTokenizerFast, set_seed
import torch.nn.functional as F

model = GPT2LMHeadModel.from_pretrained("from_scratch/gpt-2-multi-large/checkpoint-270000", )
tokenizer = BertTokenizerFast.from_pretrained("my_tokenizer", padding_side="left")


def topp_filtering(logits, top_p, filter_value=-float('Inf')):
    if top_p < 0 or top_p > 1:
        raise ValueError("top_p must between 0 and 1")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    token_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(token_probs, dim=-1)  # 计算累积概率
    print(cumulative_probs[:5])

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    print(sorted_indices_to_remove[:5])

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value

    return logits


def build_input(history, max_history_len):
    prompt = {}
    speaker1_id, speaker2_id = tokenizer.additional_special_tokens_ids
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
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

    input_ids += [speaker2_id]
    speaker_ids += [speaker2_id]
    prompt["input_ids"] = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # [[CLS, s1, id0, id1... SEP]]
    prompt["token_type_ids"] = torch.tensor(speaker_ids, dtype=torch.long).unsqueeze(0)
    return prompt


def chat(history, top_p=0.9, temperature=0.7, max_history_len=1):
    prompt = build_input(history, max_history_len)
    prompt_len = prompt["input_ids"].shape[1]
    output_ids = model.generate(**prompt, max_new_tokens=50, do_sample=True, top_p=top_p, temperature=temperature,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.sep_token_id)
    response = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    response = response.replace(" ", "")
    history[-1][1] = response
    return response, history


def my_chat(history, top_p=0.9, temperature=0.7, max_history_len=1):
    prompt = build_input(history, max_history_len)
    response = []
    for _ in range(50):
        outputs = model(**prompt)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')  # 将[UNK]的概率设为无穷小

        filtered_logits = topp_filtering(next_token_logits, top_p=top_p)
        token_probs = F.softmax(filtered_logits, dim=-1)

        # 根据概率从数组中采样，抽取1个元素，返回元素的下标
        next_token = torch.multinomial(token_probs, num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则生成结束
            break
        else:
            response.append(next_token.item())
            prompt['input_ids'] = torch.cat((prompt['input_ids'], next_token.unsqueeze(0)),
                                            dim=1)  # 把生成的token加入input_ids
            prompt['token_type_ids'] = torch.cat((prompt['token_type_ids'], torch.tensor([[2]], dtype=torch.long)),
                                                 dim=1)

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

        response, history = my_chat(history)
        print("chatbot:" + response)


if __name__ == "__main__":
    main()
