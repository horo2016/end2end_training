import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from model import NetworkNvidia,myNetworkFromNvidia, LeNet 

 # cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("==> Use accelerator: ", device)
 
#model = NetworkNvidia().to(device)
model = myNetworkFromNvidia().to(device)

model.load_state_dict(torch.load("./weight_my20-epoch.pth")) # 加载pytorch模型

 
batch_size = 1  #批处理大小
model.eval()


x = torch.randn(batch_size,3,200,200)    # 生成张量 ch=3 CHW
print(x.shape)
x = x.float().to(device)
export_onnx_file = "mynvidia.onnx"        # 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=11,
                    do_constant_folding=True,   # 是否执行常量折叠优化
                    input_names=["input"],  # 输入名
                    output_names=["output"],    # 输出名
                    dynamic_axes={"input":{0:"batch_size"}, # 批处理变量
                                 "output":{0:"batch_size"}})