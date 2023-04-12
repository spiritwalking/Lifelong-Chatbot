from datasets import Dataset
import json
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("../my_tokenizer")


def gen_single():
    with open('data/single.json', 'r', encoding='utf-8') as f:
        dialogs = json.load(f)
        for dialog in dialogs:
            yield {"dialog": dialog}


def map_function(example):
    """
    input_ids: [CLS][speaker1]你好[SEP][speaker2]很高兴认识你[SEP][speaker1]我也是[SEP][speaker2]哈哈[SEP]
    """
    speaker1_id, speaker2_id = tokenizer.additional_special_tokens_ids
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    input_ids = [cls]
    speaker_ids = [cls]
    for i, uttr in enumerate(example['dialog']):
        sid = speaker2_id if i % 2 else speaker1_id
        uttr_ids = [sid] + tokenizer.encode(uttr, add_special_tokens=False) + [sep]

        input_ids.extend(uttr_ids)
        speaker_ids.extend([sid] * len(uttr_ids))
    length = len(input_ids)
    # print(tokenizer.decode(input_ids))

    return {'input_ids': input_ids, 'token_type_ids': speaker_ids, 'length': length}


def get_dataset():
    dataset = Dataset.from_generator(gen_single)
    # dataset = dataset.select(range(2000))
    return dataset.train_test_split(test_size=0.005)


def preprocess():
    uttr_dataset = get_dataset()
    tokenized_dataset = uttr_dataset.map(map_function, num_proc=8)
    tokenized_dataset.save_to_disk("tokenized-single")


if __name__ == "__main__":
    preprocess()
    # uttr_dataset = get_dataset()
    # example = {'dialog': ["你好", "很高兴认识你", "我也是", "哈哈"]}
    # print(map_function(example))
