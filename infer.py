import torch
import evaluate
import numpy as np
from transformers import GPT2LMHeadModel, set_seed
from tqdm import tqdm
from datasets import load_from_disk
from generate import chat
import warnings

set_seed(42)
warnings.filterwarnings("ignore")


def test_bleu(model, valid_set):
    model.eval()
    bleu = evaluate.load('sacrebleu')
    with torch.no_grad():
        for dialog in tqdm(valid_set['dialog']):
            dialog_len = len(dialog) // 2 * 2
            full_history = []
            for i in range(0, dialog_len, 2):
                full_history.append(dialog[i:i + 2])
                reference = full_history[-1][1]
                history = full_history
                history[-1][1] = None
                response, _ = chat(history, chat_model=model, max_history_len=0)
                bleu.add_batch(predictions=[response], references=[reference])

        bleu_score = bleu.compute(tokenize='zh')
        print(bleu_score)


def main():
    # initialize the model
    checkpoint = "from_scratch/gpt-2-multi-large/checkpoint-470000"
    model = GPT2LMHeadModel.from_pretrained(checkpoint).to("cuda")

    # load validation set
    dataset = load_from_disk("from_scratch/tokenized-multi-large")
    test_bleu(model, dataset['test'])


if __name__ == '__main__':
    main()
