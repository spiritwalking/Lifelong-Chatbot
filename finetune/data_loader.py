from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pickle
import numpy as np


class TrainingDataset(Dataset):
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


class ValidationDataset(Dataset):
    def __init__(self, path, max_len):
        self.data = []
        with open(path, "rb") as f:
            dialog_list = pickle.load(f)
        for dialog in dialog_list:
            dialog = np.array(dialog[:max_len])
            indices = np.where((dialog == 102) | (dialog == 101))[0]
            if len(indices) % 2 == 0:
                indices = indices[:-1]

            for i in range(1, len(indices), 2):
                history = dialog[:indices[i] + 1]  # [[CLS] text_ids [SEP] text_ids [SEP]]
                reference = dialog[indices[i] + 1:indices[i + 1]]  # [test_ids]
                self.data.append((history.tolist(), reference.tolist()))

    def __getitem__(self, index):
        history = torch.tensor(self.data[index][0], dtype=torch.long)
        reference = torch.tensor(self.data[index][1], dtype=torch.long)
        return history, reference

    def __len__(self):
        return len(self.data)


def get_task_dataset(task_id, args, logger):
    train_file = args.train_folder + '/task' + str(task_id) + '_train.pkl'
    test_file = args.train_folder + '/task' + str(task_id) + '_test.pkl'

    with open(train_file, "rb") as f1, open(test_file, "rb") as f2:
        train_list = pickle.load(f1)
        test_list = pickle.load(f2)

    train_dataset = TrainingDataset(train_list, args.max_len)
    val_dataset = TrainingDataset(test_list, args.max_len)
    logger.info(f"Task{task_id} train set length: {len(train_dataset)}, val set length: {len(val_dataset)}")

    return train_dataset, val_dataset


def get_training_dataset(args, logger):
    """
    加载训练集和验证集
    """
    train_files = [args.train_folder + '/task' + str(i) + '_train.pkl' for i in range(6)]
    test_files = [args.train_folder + '/task' + str(i) + '_test.pkl' for i in range(6)]

    train_list = []
    test_list = []
    for train_file, test_file in zip(train_files, test_files):
        with open(train_file, "rb") as f1, open(test_file, "rb") as f2:
            train_list.extend(pickle.load(f1))
            test_list.extend(pickle.load(f2))

    # 划分训练集与验证集
    train_dataset = TrainingDataset(train_list, args.max_len)
    val_dataset = TrainingDataset(test_list, args.max_len)
    logger.info(f"train set length: {len(train_dataset)}, val set length: {len(val_dataset)}")

    return train_dataset, val_dataset


def get_training_loader(args, collate_fn, logger, task_id=None):
    logger.info("loading training dataset and validation dataset")
    if task_id:
        train_dataset, validate_dataset = get_task_dataset(task_id, args, logger)
    else:
        train_dataset, validate_dataset = get_training_dataset(args, logger)
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


def get_validation_loader(path, max_len, batch_size, logger):
    logger.info("loading validation dataset")
    valset = ValidationDataset(path, max_len)
    validation_loader = DataLoader(valset, batch_size=batch_size)
    return validation_loader


if __name__ == "__main__":
    valset = ValidationDataset("data/task5_test.pkl", 300)
    test_loader = DataLoader(valset, batch_size=1)
    print("hello")
