import logging
import os


def create_logger(log_path):
    """将日志输出到日志文件和控制台"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def save_model(path, model):
    if not os.path.exists(path):
        os.mkdir(path)
    model_to_save = model.module if hasattr(model, 'module') else model  # 访问被封装在DataParallel中的model，要使用module属性
    model_to_save.save_pretrained(path)
