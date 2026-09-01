# 从零写一个 Linear 类：初始化时接收输入维度和输出维度，以及可选的随机种子；用「均值为 0 的小随机数乘以 0.1」来设置权重，用「全 0 向量」来设置偏置；再写一个 forward 方法，返回输入经过「矩阵乘权重 + 偏置」后的结果。
# 创建 layer = Linear(2, 3)，打印权重的形状（应 (2,3)）、偏置（3 个 0），并解释为什么权重要乘 0.1。
# x = [[1.0, 2.0]]（注意写成二维），用 layer.forward(x) 算一遍，确认输出形状是 (1, 3)。
import numpy as np
class Linear:
    def __init__(a,w,out,seed=0):
        np.random.seed(seed)
        a.w=np.random.randn(w,out)*0.1
        a.b=np.zeros(out)
    def forward(a,x):
        return x@a.w+a.b
layer=Linear(2,3)
print(layer.w.shape)
print(layer.b)
x=np.array([[1,2]])
print(layer.forward(x))