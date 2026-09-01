# 写一个函数 make_matrix(rows, cols, seed=0)：在函数内部先固定随机种子，再返回一个随机矩阵。分别调用不传种子和传种子 1，打印说明两者是否相同。
# 写一个函数 total，它能接收任意多个位置参数并返回它们的和，然后调用它传入 1, 2, 3, 4。
import numpy as np
def juzhen(x,y,seed=0):
    np.random.seed(seed)
    return np.random.rand(x,y)
print(juzhen(1,2))
print(juzhen(1,2,1))
def total(*a):
    return sum(a)
print(total(1,2,3,4))
