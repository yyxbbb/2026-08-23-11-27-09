# 创建一个 3 行 4 列、值全为 10 的矩阵 M。再造一个 3 行 1 列的列向量 [1,2,3]，把它加到 M 上，描述每一列分别多了什么。
# 再造一个 1 行 4 列的行向量 [1,2,3,4]，加到 M 上，描述每一行分别多了什么。
# 手算并用代码验证：一个 3×1 的列向量 [[1],[2],[3]] 与一个 1×3 的行向量 [[10,20,30]] 相加，结果是什么形状、什么值？
# 报错验证：直接把一个长度为 3 的一维数组加到 M（形状 3×4）上，会报什么错？用异常处理把它捕获并打印错误信息。
import numpy as np
m=np.ones((3,4))*10
n=np.arange(1,4).reshape(3,1)
print(n+m)
a=np.array([1,2,3,4]).reshape(1,4)
print(a+m)
b=np.array([1,2,3]).reshape(3,1)
c=np.array([1,2,3]).reshape(1,3)
print(b+c)
try:
    m+np.array([1,2,3])
except Exception as e:
    print(e)