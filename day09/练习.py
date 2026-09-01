import numpy as np
class Linear:
    def __init__(a,w,b,seed=0):
        np.random.seed(seed)
        a.w=np.random.randn(w,b)*0.1
        a.b=np.zeros(b)
    def jiaf(a,x):
        return x@a.w+a.b
a=Linear(2,3)
x=np.array([[1.0,2.0],[2,3]])
print(a.w,a.b)
print(a.jiaf(x))