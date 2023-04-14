from transformers import (BertTokenizerFast, GPT2LMHeadModel, GPT2Config, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)
from datasets import load_from_disk
import numpy as np
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
set_seed(42)
warnings.filterwarnings("ignore")

tokenizer = BertTokenizerFast.from_pretrained("../my_tokenizer")
config = GPT2Config.from_pretrained("gpt-2/checkpoint-260000")
model = GPT2LMHeadModel(config)


def main():
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = load_from_disk("tokenized-multi")
    dataset = dataset.filter(lambda x: x["length"] < 600, num_proc=8)
    dataset = dataset.map(remove_columns=['dialog', 'length'], batched=True)
    args = TrainingArguments(
        output_dir="gpt-2-multi",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_steps=500,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=1000,
        learning_rate=3e-5,
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
