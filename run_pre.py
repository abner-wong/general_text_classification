#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2022/7/22 23:04
# @Author  :Abner Wong
# @Software: PyCharm

import torch

from config import Config
from data_utils import TextTokenize, get_pre_data_loader
from TextRnnAtn import Model

config = Config()
tokenizer = TextTokenize(config=config)

model = Model(config)
model.load_state_dict(torch.load(config.model_save_path, map_location=torch.device(config.device)))
model.eval()


def get_cat(smses):
    """

    :param smses:
    :return:
    """
    result = []
    if not smses:
        return result
    data_loader = get_pre_data_loader(config=config, tokenizer=tokenizer, data=smses)

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            outputs = model(data)
            pres = torch.max(outputs.data, 1)[1].cpu()
            result.extend(pres.tolist())
        category_result = [config.map_labe.get(int(i)) for i in result]
    return category_result


if __name__ == '__main__':
    import time
    text = ["En su tarjeta BANORTE IXE 40 para no generar intereses debe cubrir $ 13309.32 pero si paga HOY ANTES DE LAS 4PM $ 200 Evita cargos extras 4775006361",
            "En su tarjeta BANORTE IXE 40 para no generar intereses debe cubrir $ 13309.32 pero si paga HOY ANTES DE LAS 4PM $ 200 Evita cargos extras 4775006361",
            "En su tarjeta BANORTE IXE 40 para no generar intereses debe cubrir $ 13309.32 pero si paga HOY ANTES DE LAS 4PM $ 200 Evita cargos extras 4775006361", 
            ]*100
    bg = time.time()
    print(get_cat(text))
    print(time.time() - bg)

