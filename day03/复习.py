import numpy as np
np.random.seed(0)
a=np.random.rand(4,4)
print(np.round(a,2))
b=a[a>0.5]
print(np.round(b,2))
print(np.size(b))
c=(a>0.5).astype(int)
print(c)
d=np.where(a>0.5,1,0)
print(d)