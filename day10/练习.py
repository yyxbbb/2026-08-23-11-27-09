import numpy as np
class Linear:
    def __init__(a,w,b,seed=0):
        np.random.seed(seed)
        a.w=np.random.randn(w,b)*0.1
        a.b=np.zeros(b)
    def forward(a,x):
        return x@a.w+a.b
    def __call__(a,x):
        return a.forward(x)
a=Linear(2,3)
x=np.array([[1,2]])
print(a(x))