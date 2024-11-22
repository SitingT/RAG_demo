import torch

# 检查 PyTorch 是否安装成功
print("PyTorch version:", torch.__version__)

# 创建一个随机的 Tensor
x = torch.rand(5, 3)
print("A random tensor:")
print(x)
