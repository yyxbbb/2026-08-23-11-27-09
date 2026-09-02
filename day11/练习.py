import numpy as np
a=np.arange(10)
np.save("a.npy",a)
b=np.load("a.npy")
print("读回的数组",b)
print("和原数组一样",np.array_equal(a,b))
try:
    np.load("不存在的文件.npy")
except FileNotFoundError as e:
    print(e)