import torch

from config import Config
from data_utils import TextTokenize, get_pre_data_loader
from TextRnnAtn import Model

# convert to onnx model to speed up inference
# in progress

config = Config()

mymodel = Model(config)
mymodel.load_state_dict(torch.load(config.model_save_path, map_location=torch.device(config.device)))
mymodel.eval()



dummy_input = [[1]*80]
dummy_input = torch.LongTensor(dummy_input)

input_names = ["input_ids"]
output_names = ["outputs"]

torch.onnx.export(
    mymodel,
    (dummy_input, ),
    config.onnx_save_path,
    opset_version=12,
    do_constant_folding=True,	# 是否执行常量折叠优化
    input_names=input_names,
    output_names=output_names,)


