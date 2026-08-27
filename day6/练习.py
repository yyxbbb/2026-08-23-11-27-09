import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,2*np.pi,100)
plt.plot(x,np.sin(x),label="sin(x)")
n=np.random.rand(30)*2*np.pi
plt.scatter(n,np.sin(n),color="red",label="样本点")
plt.ylabel("y")
plt.xlabel("x")
plt.title("sin曲线和样本")
plt.legend()
plt.show()