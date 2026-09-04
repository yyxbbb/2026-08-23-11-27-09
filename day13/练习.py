import numpy as np
def chaji(a,b):
    return np.linalg.norm(a-b)
def fangxiang(a,b):
    return a@b/(np.linalg.norm(a)*np.linalg.norm(b))
a=np.array([1,2,3])
b=np.array([4,5,6])
print("欧氏距离：",chaji(a,b))
print("余弦相似度：",fangxiang(a,b))
print("a和自己的余弦值",fangxiang(a,a))