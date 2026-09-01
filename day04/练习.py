import numpy as np
n=np.array([1,2,3,4]).reshape(2,2)
m=np.arange(5,9).reshape(2,2)
print(n@m)
print(n.dot(m))
print(n*m)
print(n.T)