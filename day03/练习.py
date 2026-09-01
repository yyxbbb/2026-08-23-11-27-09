import numpy as np
a=np.ones((3,4))*10
print(a)
b=np.array([1,2,3]).reshape(3,1)
print(a+b)
c=np.array([1,2,3,4]).reshape(1,4)
print(a+c)
try:
    np.array([1,2,3]).reshape(3,1)+a
except ValueError as e:
    print(e)