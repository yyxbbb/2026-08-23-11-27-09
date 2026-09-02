# 生成一个 3 行 4 列的矩阵，把它存成 m.npy 文件，再读回来，验证读回的数组与原数组完全一致。
# 用异常处理去读一个不存在的文件 no_such.npy，捕获后打印「文件不存在，请检查路径」。
# 验证存的是「复制」：保存之后修改原数组的某个元素，再读回文件，打印该元素看变没变，说明存盘存的是当时的值还是引用。
import numpy as np
m=np.arange(12).reshape(3,4)
np.save("m.npy",m)
a=np.load("m.npy")
print("和原数组一样",np.array_equal(m,a))
try:
    np.load("no_such.npy")
except FileNotFoundError as e:
    print("文件不存在，请检查路径",e)