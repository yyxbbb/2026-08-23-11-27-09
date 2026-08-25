import numpy as np
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.shape)
print(a.dtype)
print(a.ndim)
b= np.arange(12).reshape(3,4)
print(b)
c=np.full((2,3),7)
print(c)
d=np.zeros((4,4))
print(d)
print(np.ones((3,3),dtype=int))

