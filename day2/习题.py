# m = np.arange(9).reshape(3, 3)。

# 取出「第 2 行第 3 列」的元素，它的值是多少？
# 用布尔索引取出所有大于 4 的元素，结果是什么？
import numpy as np
m=np.arange(9).reshape(3,3)
print(m[1,2])
print(m[m>4])