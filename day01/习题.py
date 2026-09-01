#题目**：用 `np.full` 创建一个 2 行 3 列、元素全是 7 的数组，并打印它的 `shape`；再用 `np.arange(24)` 把它 reshape 成一个 `(2, 3, 4)` 的三维数组，打印 `shape` 和 `ndim`。
import numpy as np
a=np.full((2,3),7)
print(a.shape)
b=np.arange(24).reshape(2,3,4)
print(b.shape)
print(b.dtype)
print(b.ndim)
print(b)