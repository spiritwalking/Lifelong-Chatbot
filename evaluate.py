from transformers import (BertTokenizerFast, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)
from datasets import load_from_disk
import numpy as np
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
warnings.filterwarnings("ignore")

checkpoint = "from_scratch/gpt-2-multi-large/checkpoint-470000"
tokenizer = BertTokenizerFast.from_pretrained("my_tokenizer")
model = GPT2LMHeadModel.from_pretrained(checkpoint)


def main():
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    uttr_dataset = load_from_disk("from_scratch/tokenized-single-large")
    training_args = TrainingArguments(
        output_dir='from_scratch/test-gpt2',
        resume_from_checkpoint=checkpoint,
        per_device_eval_batch_size=32
    )

    trainer = Trainer(
        model,
        args=training_args,
        eval_dataset=uttr_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    results = trainer.evaluate()
    print(results)


main()
