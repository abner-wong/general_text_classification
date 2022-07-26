#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2022/7/22 23:07
# @Author  :Abner Wong
# @Software: PyCharm

import re
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class TextTokenize:
    def __init__(self, config):
        self.config = config
        self.vocab = self.load_vocab()

    def load_vocab(self):
        return joblib.load(self.config.vocab_path)

    @staticmethod
    def text_processing(text):
        """
        文本预处理
        :param text:
        :return:
        """
        if not text:
            return ""
        punctuation = ['!', '"', '#', '&', '\(', '\)', '\*', '\+', '\:', ';', '<', '=', '>', '\[', '\\\\', '\]',
                       '^', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', '//', '/', '\.']
        text = re.sub("|".join(punctuation), " ", text.replace('{', ' ')).lower()
        return text

    def __call__(self, text):
        """
        分词
        :param text:
        :param config:
        :return:
        """
        text = [i for i in self.text_processing(text).split(" ") if len(i) > 0]

        if len(text) > self.config.sequence_max_len:
            text = text[:self.config.sequence_max_len]
        else:
            text = text + [self.config.PAD_TAG] * (self.config.sequence_max_len - len(text))  # 填充PAD

        tokens = [self.vocab.get(i, self.config.unk_idx) for i in text]
        return tokens


class MyDataSet(Dataset):
    def __init__(self, df, config) -> None:
        super().__init__()
        self.len = len(df)
        self.label = list(df[config.label_col])
        self.bodys = list(df[config.text_col])

    def __getitem__(self, index):
        return self.label[index], self.bodys[index]

    def __len__(self, ):
        return self.len


class PreDataSet(Dataset):
    def __init__(self, smses):
        self.len = len(smses)
        self.smses = smses

    def __getitem__(self, idx):
        return self.smses[idx]

    def __len__(self):
        return self.len


def get_train_valid_data_loader(config, tokenizer, is_train=False):
    """

    :param config:
    :param tokenizer: func
    :param is_train:
    :return:
    """

    def _collate_fn(batch):
        """
        对batch数据进行处理
        :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
        :return: 元组
        """
        labels = []
        texts = []
        for label, text in batch:
            if label not in config.label_map:
                raise ValueError(f"unknow label {label}")
            labels.append(config.label_map.get(label))
            texts.append(tokenizer(text))
        texts = torch.LongTensor(texts)
        labels = torch.LongTensor(labels)
        return labels, texts
    
    if is_train:
        df = pd.read_csv(config.train_data_path)
    else:
        df = pd.read_csv(config.test_data_path)
    train_iter = MyDataSet(df, config)

    data_loader = DataLoader(
        train_iter,
        batch_size=config.train_batch_size if is_train else config.test_batch_size,
        shuffle=True if is_train else False,
        collate_fn=_collate_fn
    )
    return data_loader


def get_pre_data_loader(config, tokenizer, data):
    """

    :param config:
    :param tokenizer:  func
    :param data:
    :return:
    """

    def _collate_fn(batch):
        """
        对batch数据进行处理
        :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
        :return: 元组
        """
        texts = []
        for text in batch:
            texts.append(tokenizer(text))
        texts = torch.LongTensor(texts)
        return texts

    pre_iter = PreDataSet(data)

    data_loader = DataLoader(
        pre_iter,
        batch_size=config.pre_batch_size,
        shuffle=False,
        collate_fn=_collate_fn
    )
    return data_loader

