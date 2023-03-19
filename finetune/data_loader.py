from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pickle


class ChatDataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def get_dataset(args):
    """
    加载训练集和验证集
    """
    train_path = args.train_path
    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 划分训练集与验证集
    val_num = args.val_num
    all_dataset = ChatDataset(input_list, args.max_len)

    # debug with small dataset
    # lengths = [1000, 100, len(all_dataset)-1100]
    # train_dataset, val_dataset, _ = random_split(all_dataset, lengths)

    lengths = [len(all_dataset) - val_num, val_num]
    train_dataset, val_dataset = random_split(all_dataset, lengths)

    return train_dataset, val_dataset


def get_dataloader(args, collate_fn, logger):
    logger.info("loading training dataset and validating dataset")
    train_dataset, validate_dataset = get_dataset(args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True)

    return train_dataloader, validate_dataloader
