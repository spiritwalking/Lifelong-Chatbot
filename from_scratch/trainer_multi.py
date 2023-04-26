from transformers import (BertTokenizerFast, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)
from datasets import load_from_disk
import numpy as np
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
set_seed(42)
warnings.filterwarnings("ignore")

tokenizer = BertTokenizerFast.from_pretrained("../my_tokenizer")
model = GPT2LMHeadModel.from_pretrained("gpt-2-large/checkpoint-620000")


def main():
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = load_from_disk("tokenized-multi-large")
    dataset = dataset.filter(lambda x: [id_len < 500 for id_len in x['length']], batched=True)
    dataset = dataset.map(remove_columns=['dialog', 'length'], batched=True)
    args = TrainingArguments(
        output_dir="gpt-2-multi-large",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_steps=500,
        num_train_epochs=5,
        warmup_steps=1000,
        learning_rate=4e-5,
        save_steps=10000,
        save_total_limit=10,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
