# 写 euclidean(a, b) 返回 a 与 b 的欧氏距离；写 cosine_sim(a, b) 返回 a 与 b 的余弦相似度（定义为「点积 ÷ 两者长度的乘积」）。
# a = [1,0,0]，b = [0,1,0]：两者的欧氏距离和余弦相似度各是多少？
# 把 b 换成 [2,0,0]（与 a 同向但长度不同）：余弦变不变？欧氏距离变不变？
# a 与自身的余弦应等于 1，验证一下。
import numpy as np
def euclidean(a,b):
    return np.linalg.norm(a-b)
def cosine_sim(a,b):
    return a@b/(np.linalg.norm(a)*np.linalg.norm(b))
a=np.array([1,0,0])
b=np.array([0,1,0])
print("欧式距离:",euclidean(a,b))
print("余弦相似度:",cosine_sim(a,b))
b=np.array([2,0,0])
print("欧式距离：",euclidean(a,b))
print("余弦相似度：",cosine_sim(a,b))