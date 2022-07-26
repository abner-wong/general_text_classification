#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2022/7/22 21:41
# @Author  :Abner Wong
# @Software: PyCharm

import torch


class Config(object):
    """配置参数"""

    def __init__(self):
        self.projet_path = ""
        self.UNK_TAG = "<UNK>"  # 表示未知字符
        self.PAD_TAG = "<PAD>"  # 填充符
        self.unk_idx = 1
        self.pad_idx = 0

        self.label_col = "category" # label 字段名字
        self.text_col = "body" # sms 字段名字

        self.model_name = 'TextRNN_Att'  # 'TextRNN_Att'
        self.train_data_path = ""
        self.test_data_path = ""
        self.vocab_path = self.projet_path + "models/ws_dict.pkl"
        self.model_save_path = self.projet_path + 'models/' + self.model_name + '.ckpt'  # 模型训练结果
        self.onnx_save_path = self.projet_path + 'models/' + self.model_name + '.onnx'  # 转onnx
        self.log_path = self.projet_path + f"logs/{self.model_name}.log"

        self.label_map = {
        }
        
        self.map_labe = dict(zip(self.label_map.values(), self.label_map.keys()))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.embedding_pretrained = None # 预训练词向量

        self.num_classes = len(self.label_map)  # 类别数
        self.n_vocab = 34853  # 词表大小，在运行vocab后取到

        self.sequence_max_len = 80

        self.num_epochs = 50  # epoch数
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.train_batch_size = 128  # mini-batch大小
        self.test_batch_size = 512
        self.pre_batch_size = 512

        self.dropout = 0.4  # 随机失活
        self.learning_rate = 1e-3  # 学习率

        self.embed = 200  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.hidden_size2 = 64
        self.num_filters = 250  # 卷积核数量(channels数)
