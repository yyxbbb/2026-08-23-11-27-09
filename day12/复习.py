import numpy as np
a=np.arange(12).reshape(3,4)
b=np.save("a.npy",a)
c=np.load("a.npy")
print(c)
print(np.array_equal(b,c))
try:
    np.load("不存在的文件")
except FileNotFoundError as e:
    print(e)