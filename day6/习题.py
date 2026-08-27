# 生成 -3 到 3 共 100 个等间距的 x，画出 y = x² 的折线，并给这条线起个名字。
# 在同一张图里，随机取 30 个点（x 在 -3~3 随机取、y 取 x² 加上随机噪声）画成散点，也给它起个名字。
# 给图加上横轴标签、纵轴标签、标题「抛物线」、图例，并把图显示出来。
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-3,3,100)
plt.plot(x,x**2,label="y=x²")
n=np.random.rand(30)*6-3
# plt.scatter(n,n**2,color="red",label="样本点")
plt.scatter(n,n**2+(np.random.rand(30)-0.5)*0.3,label="噪声")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y=x²")
plt.legend()
plt.show()