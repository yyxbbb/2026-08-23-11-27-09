# M = np.ones((3, 4)) * 10。构造 v_col = np.array([1, 2, 3]).reshape(3, 1)，算 M + v_col，描述每一列分别多了什么。
# 构造 v_row = np.array([1, 2, 3, 4]).reshape(1, 4)，算 M + v_row，描述每一行分别多了什么。
# 手算并用代码验证：a = np.array([[1],[2],[3]])（3×1）与 b = np.array([[10, 20, 30]])（1×3）相加，结果是什么形状、什么值？
# 报错验证：直接 np.array([1, 2, 3]) + M（形状 (3,) 和 (3,4)）会报什么错？用 try/except 捕获并打印错误信息。
import numpy as np
m=np.ones((3,4))*10
b=np.array([1,2,3]).reshape(3,1)
print(m+b)
c=np.array([1,2,3]).reshape(1,3)
print(b+c)
try:
    np.array([1,2,3])+m
except ValueError as e:
    print(e)