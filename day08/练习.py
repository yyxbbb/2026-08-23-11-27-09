import numpy as np
def juz(x,y,seed=0):
    np.random.seed(seed)
    return np.random.rand(x,y)
n=juz(2,3)
m=juz(2,3,seed=1)
print(np.round(m,2))
print(np.round(n,2))
def jiafa(*nums):
    return sum(nums)
print(jiafa(1,2,3,4))