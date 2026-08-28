# 生成 0~8 的连续整数并改成 3×3，打印它的形状、元素类型、维度数。
# 取出它的「第 2 行」以及「前两行、第 2~3 列」这两个子矩阵。
# 取出所有大于 4 的元素，并做一个「是否大于 4」的 0/1 掩码。
# 写一个 3×3 矩阵，让向量 [1,2,3] 通过广播加到它的每一列（先改成正确的形状）。
# 定义单位阵 [[1,0],[0,1]] 和矩阵 [[2,3],[4,5]]，分别用矩阵乘、逐元素乘两种算法计算，并对后者做转置。
# 固定种子 0 后生成 1000 个正态样本，打印均值；再把种子设回 0 生成一次，证明两次相同
import numpy as np
a=np.arange(9).reshape(3,3)
print(a.shape)
print(a.dtype)
print(a.ndim)
print(a[1,])
print(a[0:2,1:3])
print(a[a>4])
print((a>4).astype(int))
n=np.arange(1,4).reshape(3,1)
print(a+n)
m=np.array([[1,0],[0,1]])
r=np.array([[2,3],[4,5]])
print(m@r,m*r,r.T)
np.random.seed(0)
b=np.random.randn(1000)
print(np.mean(b))