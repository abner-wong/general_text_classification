#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2022/7/22 23:04
# @Author  :Abner Wong
# @Software: PyCharm

from tqdm import tqdm
from datetime import datetime

import torch

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from config import Config
from data_utils import TextTokenize, get_train_valid_data_loader
from TextRnnAtn import Model


def main():
    config = Config()
    tokenizer = TextTokenize(config=config)
    train_loader = get_train_valid_data_loader(config=config,
                                               tokenizer=tokenizer,
                                               is_train=True)

    test_loader = get_train_valid_data_loader(config=config,
                                               tokenizer=tokenizer,
                                               is_train=False)

    mymodel = Model(config)
    mymodel.to(config.device)
    lossFunc = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(mymodel.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    loss_collect = []
    mymodel.train()
    f1_best = 0.0

    for epoch in range(config.num_epochs):
        with tqdm(iterable=train_loader,
                  bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}', ) as t:
            start_time = datetime.now()

            for i, (label, sms) in enumerate(train_loader):
                t.set_description_str(f"\33[36m【Epoch {epoch + 1:04d}】")
                input_ids = sms.to(config.device)
                labels = label.to(config.device)
                outputs = mymodel(input_ids)
                mymodel.zero_grad()
                loss = lossFunc(outputs, labels)
                loss.backward()
                optimizer.step()
                delta_time = datetime.now() - start_time
                loss_collect.append(loss.tolist())
                pres = torch.max(outputs.data, 1)[1].cpu()
                lab = labels.detach().cpu().numpy()
                f1 = f1_score(y_true=lab.tolist(), y_pred=pres.tolist(), average='macro')
                t.set_postfix_str(
                    f"train_loss={sum(loss_collect) / len(loss_collect):.6f} F1={f1:.6f}， 执行时长：{delta_time}\33[0m")
                t.update()

        mymodel.eval()
        pre_ls = []
        label_ls = []

        with torch.no_grad():
            for (label, sms) in test_loader:
                input_ids = sms.to(config.device)
                labels = label.to(config.device)
                outputs = mymodel(input_ids)
                pres = torch.max(outputs.data, 1)[1].cpu()
                pre_ls.extend(pres.tolist())
                label_ls.extend(labels.detach().cpu().numpy().tolist())

        all_label = set(pre_ls) & set(label_ls)
        target_names = [config.map_labe.get(int(i)) for i in all_label]
        if len(target_names) == config.num_classes:
            report = classification_report(label_ls, pre_ls, target_names=target_names)
            print(report)
            f1 = f1_score(y_true=label_ls, y_pred=pre_ls, average='macro')
            if f1 > f1_best:
                f1_best = f1
                torch.save(mymodel.state_dict(), config.model_save_path)

            with open(config.log_path, "a+") as f:
                f.write(f"epoch: {epoch}\n")
                f.write(f"best: {f1_best} now: {f1}\n")
                f.write(report)
        mymodel.train()

if __name__ == '__main__':
    main()