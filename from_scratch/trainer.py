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
config = GPT2Config.from_pretrained("gpt2", vocab_size=len(tokenizer))
model = GPT2LMHeadModel(config)


def main():
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = load_from_disk("tokenized-single-large")
    dataset = dataset.map(remove_columns=['dialog', 'length'], batched=True)
    args = TrainingArguments(
        output_dir="gpt-2-large",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_steps=500,
        num_train_epochs=2,
        warmup_steps=1000,
        learning_rate=7e-5,
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
