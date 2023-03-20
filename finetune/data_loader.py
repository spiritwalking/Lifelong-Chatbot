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


def get_dataset(args, logger):
    """
    加载训练集和验证集
    """
    train_files = [args.train_folder+'/task'+str(i)+'_train.pkl' for i in range(6)]
    test_files = [args.train_folder + '/task' + str(i) + '_test.pkl' for i in range(6)]

    train_list = []
    test_list = []
    for train_file, test_file in zip(train_files, test_files):
        with open(train_file, "rb") as f1, open(test_file, "rb") as f2:
            train_list.extend(pickle.load(f1))
            test_list.extend(pickle.load(f2))

    # 划分训练集与验证集
    train_dataset = ChatDataset(train_list, args.max_len)
    val_dataset = ChatDataset(test_list, args.max_len)
    logger.info(f"train set length: {len(train_dataset)}, val set length: {len(val_dataset)}")

    return train_dataset, val_dataset


def get_dataloader(args, collate_fn, logger):
    logger.info("loading training dataset and validating dataset")
    train_dataset, validate_dataset = get_dataset(args, logger)
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
