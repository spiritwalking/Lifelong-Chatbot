from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader


def get_dataloader(tokenizer, args):
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_from_disk("tokenized-data")
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids'])
    train_dataloader = DataLoader(dataset['train'], batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    valid_dataloader = DataLoader(dataset['test'], batch_size=args.batch_size, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader


def get_task_dataloaders(tokenizer, batch_size):
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataset = load_from_disk("tokenized-data")
    train_dataloaders = []
    validation_dataloaders = []

    for i in range(5):
        topic_dataset = dataset.filter(lambda x: x["topic"] == i)
        topic_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids'])
        train_dataloaders.append(DataLoader(topic_dataset['train'], batch_size=batch_size, collate_fn=collate_fn, shuffle=True))
        validation_dataloaders.append(DataLoader(topic_dataset['test'], batch_size=batch_size, collate_fn=collate_fn))

    return train_dataloaders, validation_dataloaders


