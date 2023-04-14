import torch
from transformers import GPT2LMHeadModel, BertTokenizerFast, set_seed

set_seed(42)
model = GPT2LMHeadModel.from_pretrained("from_scratch/gpt-2/checkpoint-260000", )
tokenizer = BertTokenizerFast.from_pretrained("my_tokenizer", padding_side="left")


def chat(history, max_history_len=1, is_tokenized=False):
    prompt = {}
    if is_tokenized:
        input_ids = history  # TODO: to be done
    else:  # 组装输入语句
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

        prompt["input_ids"] = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # [[CLS, s1, id0, id1... SEP]]
        prompt["token_type_ids"] = torch.tensor(speaker_ids, dtype=torch.long).unsqueeze(0)

    prompt_len = prompt["input_ids"].shape[1]
    output_ids = model.generate(**prompt, max_new_tokens=50, top_k=4, penalty_alpha=0.6,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.sep_token_id)
    response = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    response = response.replace(" ", "")
    if not is_tokenized:
        history[-1][1] = response
    return response, history


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
