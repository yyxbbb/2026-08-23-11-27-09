# 在第 9 天的 Linear 类里加一个特殊方法，使「实例本身能像函数一样被调用」，并且调用时等价于 forward。
# 创建 layer = Linear(1, 1)，手动把权重设成二维的 [[2.0]]、偏置设成 [1.0]，分别用 layer.forward(...) 和「直接把实例当函数调用」两种方式验证结果一致（应为 [[7.]]）。
# 坑验证：写这个特殊方法时如果忘记写 return（即只调用了 forward 却没返回），再调用实例会得到什么？说明原因。
import numpy as np
class Linear:
    def __init__(a,w,b,seed=0):
        np.random.seed(seed)
        a.w=np.random.randn(w,b)*0.1
        a.b=np.zeros(b)
    def forward(a,x):
        return x@a.w+a.b
    def __call__(a,x):
        return a.forward(x)
layer=Linear(1,1)
layer.w=np.array([[2.0]])
layer.b=np.array([1.0])
x=np.array([[3.0]])
print(layer.forward(x))
print(layer(x))