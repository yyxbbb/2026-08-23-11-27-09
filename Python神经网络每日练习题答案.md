# Python → 神经网络 · 每日练习题 · 参考答案

> ⚠️ **本文件是答案，练完再看！** 配套 `Python神经网络每日练习题.md`（题目 + 提示，不含答案）。
> 用法：自己先打出代码跑通，再对照这里的参考实现。每个答案都是完整可运行代码。

---

## 第 1 天 答案
```python
import numpy as np






# 1) 从嵌套列表创建 + 三个属性
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a.shape, a.dtype, a.ndim)        # (3, 2) int64 2

# 2) arange + reshape
b = np.arange(12).reshape(3, 4)
print(b, b.shape)                       # 0~11 排成 3 行 4 列

# 3) 全 0 / 全 1 / 全指定值
print(np.zeros((2, 3)))
print(np.ones((2, 2), dtype=int))
print(np.full((2, 3), 7))              # 2 行 3 列全是 7

# 4) 坑：少外层括号会报 TypeError，正确是 np.array([[1,2],[3,4]])
# 5) 坑：arange(10) 只有 10 个元素，reshape(3,4) 要 12 个 → 报"尺寸不匹配"


---






```
## 第 2 天 答案
```python
import numpy as np






m = np.arange(12).reshape(3, 4)
# 1) 单元素
print(m[1, 2])                  # 6  （第1行=[4,5,6,7]，第2列是 6）

# 2) 切片：前2行(0:2)、第1~2列(1:3)
print(m[0:2, 1:3])
# [[1 2]
#  [5 6]]

# 3) 布尔索引 + size
big = m[m > 6]
print(big, big.size)            # [ 7  8  9 10 11] 5

# 4) 掩码
mask = (m > 6).astype(int)
print(mask)
# [[0 0 0 0]
#  [0 0 1 1]
#  [1 1 1 1]]


---






```
## 第 3 天 答案
```python
import numpy as np






M = np.ones((3, 4)) * 10
# 1) 加每列
v_col = np.array([1, 2, 3]).reshape(3, 1)
print(M + v_col)
# 第0列+1、第1列+2、第2列+3（各列依次加 1/2/3）

# 2) 加每行
v_row = np.array([1, 2, 3, 4]).reshape(1, 4)
print(M + v_row)
# 第0行+1、第1行+2、第2行+3、第3行+4

# 3) (3,1)+(1,3) 广播成 3x3
a = np.array([[1], [2], [3]])
b = np.array([[10, 20, 30]])
print(a + b)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]

# 4) 报错捕获
try:
    np.array([1, 2, 3]) + M
except ValueError as e:
    print("报错:", e)          # 形状无法广播


---






```
## 第 4 天 答案
```python
import numpy as np






A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1) 两种乘法
print("A @ B =\n", A @ B)      # [[19 22] [43 50]]
print("A * B =\n", A * B)      # [[5 12] [21 32]]

# 2) 手算验证
print(1*5 + 2*7)               # 19  == A@B[0,0]

# 3) 转置及还原
print("A.T =\n", A.T)          # [[1 3] [2 4]]
print("(A.T).T == A :\n", (A.T).T)


---






```
## 第 5 天 答案
```python
import numpy as np






# 1) 均值/标准差
np.random.seed(0)
x = np.random.randn(10000)
print(round(x.mean(), 3), round(x.std(), 3))   # ≈ 0.0, ≈ 1.0

# 2) 同种子复现
np.random.seed(0)
y = np.random.randn(10000)
print(np.array_equal(x, y))                    # True

# 3) rand（均匀 0~1）vs randn（正态，可负）
print(np.random.rand(5))       # 都落在 0~1
print(np.random.randn(5))      # 可能有负数、绝对值可 >1

# 4) 不设种子，两次不同
print(np.random.randn(3))
print(np.random.randn(3))      # 一般不一样


---






```
## 第 6 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt






x = np.linspace(-3, 3, 100)
plt.plot(x, x**2, label="y=x²")

x_sample = np.random.rand(30) * 6 - 3
y_sample = x_sample**2 + (np.random.rand(30) - 0.5)   # 加均匀随机噪声（约±0.5），用教程教过的 np.random.rand
plt.scatter(x_sample, y_sample, color="red", label="样本点")

plt.xlabel("x"); plt.ylabel("y")
plt.title("抛物线")
plt.legend()
plt.show()


---






```
## 第 7 天 答案（第 1 周综合小测）
```python
import numpy as np






# 1)
m = np.arange(9).reshape(3, 3)
print(m.shape, m.dtype, m.ndim)     # (3,3) int64 2

# 2)
print(m[1])                         # 第2行
print(m[0:2, 1:3])                  # 前两行、第2~3列

# 3)
print(m[m > 4])                     # [5 6 7 8]
print((m > 4).astype(int))

# 4) 广播加每列
M = np.zeros((3, 3))
print(M + np.array([1, 2, 3]).reshape(3, 1))

# 5)
A = np.array([[1, 0], [0, 1]]); B = np.array([[2, 3], [4, 5]])
print(A @ B)                        # [[2 3] [4 5]]
print(A * B)                        # [[2 0] [0 5]]
print(B.T)                          # [[2 4] [3 5]]

# 6)
np.random.seed(0); x = np.random.randn(1000)
np.random.seed(0); y = np.random.randn(1000)
print(round(x.mean(), 3), np.array_equal(x, y))   # ≈0.0, True


---






```
## 第 8 天 答案
```python
import numpy as np






# 1) 带默认参数的函数
def make_matrix(rows, cols, seed=0):
    np.random.seed(seed)
    return np.random.rand(rows, cols)

print(np.round(make_matrix(2, 3), 2))
print(np.round(make_matrix(2, 3, seed=1), 2))   # 与上面不同（种子不同）

# 2) 变长参数
def total(*nums):
    return sum(nums)
print(total(1, 2, 3, 4))                          # 10

# 3) 可变默认参数陷阱
def f(x, lst=[]):
    lst.append(x)
    return lst
print(f(1))     # [1]
print(f(2))     # [1, 2]  ← 不是 [2]！默认列表被共享了


---






```
## 第 9 天 答案
```python
import numpy as np






class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(in_dim, out_dim) * 0.1   # 乘 0.1 防初始值过大
        self.b = np.zeros(out_dim)
    def forward(self, x):
        return x @ self.W + self.b

layer = Linear(2, 3)
print(layer.W.shape)        # (2, 3)
print(layer.b)              # [0. 0. 0.]
x = np.array([[1.0, 2.0]])
out = layer.forward(x)
print(out, out.shape)       # 形状 (1, 3)


---






```
## 第 10 天 答案
```python
import numpy as np






class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros(out_dim)
    def forward(self, x):
        return x @ self.W + self.b
    def __call__(self, x):          # 加了这一行
        return self.forward(x)

layer = Linear(1, 1)
layer.W = np.array([[2.0]]); layer.b = np.array([1.0])
print(layer.forward(np.array([[3.0]])))   # [[7.]]
print(layer(np.array([[3.0]])))           # [[7.]]  两者一致

# 3) 漏 return 的版本：
# class Bad:
#     def __call__(self, x): self.forward(x)   # 没 return
# 调用后得到 None（拿不到结果）


---






```
## 第 11 天 答案
```python
import numpy as np






a = np.arange(12).reshape(3, 4)
np.save("m.npy", a)
b = np.load("m.npy")
print("一致:", np.array_equal(a, b))     # True

try:
    np.load("no_such.npy")
except FileNotFoundError as e:
    print("文件不存在，请检查路径:", e)

# 3) save 是复制：改原数组不影响已存文件
a[0, 0] = 999
b = np.load("m.npy")
print(b[0, 0])                          # 仍是 0（存的快照，没被改）


---






```
## 第 12 天 答案（无标准代码，能跑出 sin 图并保存 notebook 即达标）
Code cell 可参考：
```python
import numpy as np
import matplotlib.pyplot as plt






%matplotlib inline
x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x))
plt.title("day12")
plt.show()


---






```
## 第 13 天 答案
```python
import numpy as np






def euclidean(a, b):
    return np.linalg.norm(a - b)
def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
print(euclidean(a, b))     # 1.4142... (=√2)
print(cosine_sim(a, b))    # 0.0

b2 = np.array([2.0, 0.0, 0.0])
print(cosine_sim(a, b2))   # 1.0  （方向相同，长度不影响余弦）
print(euclidean(a, b2))    # 1.0  （欧氏距离变了）

print(cosine_sim(a, a))    # 1.0


---






```
## 第 14 天 答案（第 2 周综合自测）
```python
import numpy as np






# 1) 默认参数 + 封装
def make_matrix(rows, cols, seed=0):
    np.random.seed(seed)
    return np.random.rand(rows, cols)
print(np.round(make_matrix(2, 2), 2))
print(np.round(make_matrix(2, 2, seed=1), 2))   # 不同

# 2) 变长参数
def total(*nums):
    return sum(nums)
print(total(2, 4, 6))                            # 12

# 3) Linear 全功能
class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros(out_dim)
    def forward(self, x):
        return x @ self.W + self.b
    def __call__(self, x):
        return self.forward(x)
layer = Linear(3, 2)
print(layer(np.array([[1.0, 2.0, 3.0]])))       # 形状 (1, 2)

# 4) 文件读写 + 异常处理
a = np.arange(10)
np.save("t.npy", a)
print(np.array_equal(a, np.load("t.npy")))       # True
try:
    np.load("missing.npy")
except FileNotFoundError as e:
    print("文件不存在:", e)

# 5) 距离与相似度
def euclidean(a, b): return np.linalg.norm(a - b)
def cosine_sim(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
a = np.array([1.0, 2.0, 3.0]); b = np.array([4.0, 5.0, 6.0])
print(euclidean(a, b))                           # ≈5.196
print(cosine_sim(a, b))                          # ≈0.974

# 6) seed 让随机结果可复现，方便调试和与他人对答案


---






```
## 第 15 天 答案
```python
import numpy as np






def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1., 2.]); b = np.array([3., 4.])
print(a @ b)                                   # 11.0
print(np.isclose(a @ b, np.linalg.norm(a)*np.linalg.norm(b)*cosine_sim(a,b)))  # True
print(np.isclose(a @ b, b @ a))                # True

c = np.array([-2., 1.]); d = np.array([100., 200.])
print(cosine_sim(a, b), cosine_sim(a, c))      # ≈0.98 同向；≈0.0 正交
print(a @ d, cosine_sim(a, d))                 # 500（很大）但余弦仍是 1.0（同方向）


---






```
## 第 16 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt






M = np.array([[0.,-1.],[1.,0.]])
p = np.array([1.,0.])
print((M @ p.T).T)                            # [0. 1.] 即 (0,1)

corners = np.array([[0.,1.,1.,0.],[0.,0.,1.,1.]])
print(M @ corners)                            # 四角被逆时针转 90°

plt.plot(corners[0], corners[1], 'o-', label="前")
plt.plot((M@corners)[0], (M@corners)[1], 's-', label="后(旋转90°)")
plt.axis('equal'); plt.legend(); plt.show()

# 4) 稳妥写法：明确转置或 p @ M.T
print(p @ M.T)                                # [0. 1.]


---






```
## 第 17 天 答案
```python
def num_deriv(f, x, h=1e-5):
    return (f(x+h) - f(x)) / h
def center_deriv(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

print(num_deriv(lambda x: x**2, 3.0))         # ≈5.99999
print(num_deriv(lambda x: x**3, 2.0))         # ≈12.000...
for h in [1e-1, 1e-3, 1e-5, 1e-7]:
    print(h, num_deriv(lambda x: x**2, 3.0, h))
# 4) 中心差分更接近 6
print("单边:", num_deriv(lambda x: x**2, 3.0, 1e-5))      # 5.99999...
print("中心:", center_deriv(lambda x: x**2, 3.0, 1e-5))   # 6.0000000...




```

---

## 第 18 天 答案
```python
import numpy as np






def f(xy):
    x,y=xy; return x**2 + y**2
def num_grad(f, xy, h=1e-5):
    g = np.zeros_like(xy, dtype=float)
    for i in range(len(xy)):
        xp=xy.copy(); xp[i]+=h; xm=xy.copy(); xm[i]-=h
        g[i]=(f(xp)-f(xm))/(2*h)
    return g
print(num_grad(f, np.array([1.,2.])))          # [2. 4.]
print(num_grad(lambda xy: xy[0]**2+2*xy[1]**2, np.array([1.,1.])))  # [2. 4.]
p=np.array([1.,2.]); g=num_grad(f,p)
print(np.allclose(g, np.array([2.,4.])))       # True
# 4) 沿梯度走 f 变大
p2 = p + 0.1*g
print(f(p2) > f(p))                            # True（向上升方向）


---






```
## 第 19 天 答案
```python
import numpy as np






def center_deriv(f, x, h=1e-5):
    return (f(x+h)-f(x-h))/(2*h)
# 1) g=x^2, f=u+1
g=lambda x: x**2; f=lambda u: u+1; h=lambda x: f(g(x))
print(center_deriv(h, 3.0))                    # ≈6.0
print(1*(2*3))                                 # 6 解析
# 3) g=sin, f=u^3
g2=np.sin; f2=lambda u: u**3; h2=lambda x: f2(g2(x))
print(center_deriv(h2, 0.0))                   # ≈0.0
# 4) 链式：f'(g(x)) * g'(x)，f' 在 g(x) 处
x=0.0
print(3*g2(x)**2 * np.cos(x))                  # 3*0^2 *1 = 0


---






```
## 第 20 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt






def f(x): return x**2 - 4*x
def df(x): return 2*x - 4
def gd(lr, x0, steps=30):
    x=x0; hist=[x]
    for _ in range(steps):
        x = x - lr*df(x); hist.append(x)
    return x, hist
x, hist = gd(0.1, 0.0)
print(x, f(x))                                # ≈2.0, ≈-4.0
xs=np.linspace(-1,3,200); plt.plot(xs,f(xs),label="f"); plt.plot(hist,[f(v) for v in hist],'ro-',label="轨迹"); plt.legend(); plt.show()
print("lr=1.5:", gd(1.5, 0.0)[0])            # 不收敛，在 2 附近震荡/发散
print("x0=10:", gd(0.1, 10.0)[0])            # ≈2.0 仍收敛


---






```
## 第 21 天 答案（第 3 周综合自测）
综合前 6 天（Day 15–20）的代码拼接即可。打卡标准：6 题独立做对，存成 `week03_review.ipynb`。

---

## 第 22 天 答案
```python
import numpy as np







def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.9, 3.2])
print(mse(y_pred, y_true))                 # 0.02
print(mse(y_true, y_true))                 # 0.0
# 4) 长度不一致会报 ValueError（operands could not be broadcast together）


---






```
## 第 23 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt







def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def tanh(x):    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def relu(x):    return np.maximum(0, x)

print(sigmoid(0), sigmoid(10), sigmoid(-10))   # 0.5  接近1  接近0

x = np.linspace(-6, 6, 100)
for i, (name, func) in enumerate([("sigmoid", sigmoid), ("tanh", tanh), ("ReLU", relu)], 1):
    plt.subplot(1, 3, i); plt.plot(x, func(x)); plt.title(name); plt.grid(True)
plt.tight_layout(); plt.show()
# 4) 1/(1+e^x)：x 大正数时 e^x 溢出 inf，结果变 nan —— 故用 1/(1+e^-x)


---






```
## 第 24 天 答案
```python
import numpy as np







def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_prime(x): return sigmoid(x) * (1 - sigmoid(x))
def num_deriv(f, x, h=1e-5): return (f(x+h) - f(x-h)) / (2*h)

for x in [-2, -1, 0, 1, 2]:
    print(x, sigmoid_prime(x), num_deriv(sigmoid, x))
# x=0 时 σ'(0)=0.25


---






```
## 第 25 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt







samples = np.random.randn(10000)
print(np.mean(samples), np.var(samples))        # ≈0, ≈1

plt.hist(samples, bins=50, density=True, alpha=0.6)
xs = np.linspace(-4, 4, 200)
plt.plot(xs, (1/np.sqrt(2*np.pi)) * np.exp(-xs**2/2), 'r-', lw=2)
plt.show()
# 3) 纵轴要设 density=True（概率密度），才能和理论密度曲线对比


---






```
## 第 26 天 答案
```python
import numpy as np







def binary_cross_entropy(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)           # 防止 log(0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

print(binary_cross_entropy(0.9, 1))            # ≈0.105 最小
print(binary_cross_entropy(0.5, 1))            # ≈0.693 中等
print(binary_cross_entropy(0.1, 1))            # ≈2.303 最大
# 3) p=0 或 1 时 log(0)→-inf，loss 变 nan；实际训练用 np.clip 把 p 限制在小区间内


---






```
## 第 27 天 答案（第 4 周综合自测）
```python
import numpy as np







# 1) x^3 在 x=2 导数=12（解析 3x^2）
num = (lambda x: x**3)(2.0 + 1e-5) - (lambda x: x**3)(2.0 - 1e-5)
print(num / (2e-5))                            # ≈12.0

# 2) tanh 数值导数，验证 tanh'(0)=1
def tanh(x): return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
print((tanh(1e-5) - tanh(-1e-5)) / (2e-5))     # ≈1.0

# 3) 学习率太大：步子过大，loss 震荡甚至发散（飞出谷底）；学习率太小：收敛极慢，训练迟迟不下降。

---






```
## 第 28 天 答案（阶段复盘 · 写作题参考）
> 训练一个模型，本质是不断调参数让"损失"变小。需要五个零件：①**数据**——带标签的样本 (x, y)，是模型要学的题与答案；②**模型**——一串可学习参数（如权重 W、偏置 b），把 x 映射成预测 ŷ；③**损失**——衡量 ŷ 与 y 差多远（回归用 MSE，分类用交叉熵）；④**梯度**——损失对每个参数的变化率，指出"该往哪个方向调"；⑤**更新规则**——沿负梯度小步调整参数（梯度下降）。这五步循环往复，就是"训练"。

---

---

## 第 5 周 答案（Day 29–35）

### 第 29 天 答案
```python
from sklearn.datasets import load_iris






iris = load_iris()
X = iris.data
y = iris.target
print("特征矩阵形状:", X.shape)        # (150, 4)
print("前 5 行特征:\n", X[:5])
print("标签前 5 个:", y[:5])            # [0 0 0 0 0]
print("标签含义:", iris.target_names)



```
### 第 30 天 答案




```python
import numpy as np
import matplotlib.pyplot as plt




np.random.seed(0)
n = 100
x = np.random.uniform(0, 10, n)
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + np.random.randn(n) * 1.5
plt.scatter(x, y, s=15, label="样本")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()




```

### 第 31 天 答案
```python
import numpy as np




w, b = 0.0, 0.0
lr = 0.01
n = len(x)
for i in range(100):
    yhat = w * x + b
    err = yhat - y
    dw = (2/n) * np.sum(err * x)
    db = (2/n) * np.sum(err)
    w -= lr * dw
    b -= lr * db
    if i % 20 == 0:
        loss = np.mean(err**2)
        print(f"第{i}轮 w={w:.3f} b={b:.3f} loss={loss:.3f}")
print("学到的 w,b:", round(w,3), round(b,3), " 应接近 2 和 1")




```
文字题：梯度来自链式法则——损失 L 对 w 求偏导时，会乘上"w 通过 ŷ 影响 L 的系数"，而 ŷ = w·x + b 中 w 对 ŷ 的贡献正好是 x，所以 ∂L/∂w 里出现 x；b 对 ŷ 的贡献是 1，所以 ∂L/∂b 里乘的是 1（可省略）。一句话：哪个参数影响输出，它的梯度就乘上它对输出的贡献。

### 第 32 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt




def train(lr, epochs=100):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        yhat = w*x + b
        err = yhat - y
        w -= lr * (2/n)*np.sum(err*x)
        b -= lr * (2/n)*np.sum(err)
        losses.append(np.mean(err**2))
    return losses

for lr in [0.001, 0.01, 1.0]:
    plt.plot(train(lr), label=f"lr={lr}")
plt.yscale("log"); plt.xlabel("轮次"); plt.ylabel("loss"); plt.legend(); plt.show()




```
观察：lr=0.001 缓慢下降（慢）；lr=0.01 平滑降到最低（合适）；lr=1.0 剧烈震荡甚至数值溢出（炸）。

### 第 33 天 答案
```python
import numpy as np




w, b = 0.0, 0.0
lr = 0.01
prev_loss = float("inf")
epochs_used = 0
for i in range(10000):
    yhat = w*x + b
    err = yhat - y
    w -= lr*(2/n)*np.sum(err*x)
    b -= lr*(2/n)*np.sum(err)
    loss = np.mean(err**2)
    epochs_used = i + 1
    if abs(prev_loss - loss) < 1e-6:
        print("提前在第", epochs_used, "轮收敛")
        break
    prev_loss = loss
print("最终 w,b:", round(w,3), round(b,3))




```

### 第 34 天 答案
```python
import numpy as np
import time




# 方式一：逐样本 for 循环
def train_loop():
    w, b = 0.0, 0.0
    for _ in range(100):
        for i in range(n):
            yhat_i = w*x[i] + b
            err_i = yhat_i - y[i]
            w -= lr*(2/n)*err_i*x[i]
            b -= lr*(2/n)*err_i

# 方式二：整批向量化
def train_vec():
    w, b = 0.0, 0.0
    for _ in range(100):
        yhat = w*x + b
        err = yhat - y
        w -= lr*(2/n)*np.sum(err*x)
        b -= lr*(2/n)*np.sum(err)

t = time.time(); train_loop(); print("逐样本耗时:", time.time()-t)
t = time.time(); train_vec();  print("向量化耗时:", time.time()-t)

# 偏置并入矩阵：给 x 加一列全 1，用一次矩阵乘同时得到 wx+b
Xb = np.column_stack([x, np.ones(n)])     # 形状 (n, 2)
W = np.array([0.0, 0.0])                  # [w, b]
for _ in range(100):
    yhat = Xb @ W
    err = yhat - y
    W -= lr * (2/n) * (Xb.T @ err)        # (Xb.T @ err) 同时给出 [对w梯度, 对b梯度]




```

### 第 35 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt




plt.scatter(x, y, s=15, alpha=0.6, label="样本")
xs = np.array([x.min(), x.max()])
plt.plot(xs, w*xs + b, "r-", label=f"拟合: y={w:.2f}x+{b:.2f}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
print("学到的 w,b:", round(w,3), round(b,3), " 应接近 2 和 1")




```

---

---

## 第 6 周 答案（Day 36–42）

### 第 36 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures





np.random.seed(0)
n = 20
x = np.random.uniform(-3, 3, n)
y = 0.5*x + np.random.randn(n)*0.6

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0)
print("训练样本数:", len(x_tr), " 测试样本数:", len(x_te))

poly = PolynomialFeatures(degree=9)
X9 = poly.fit_transform(x_tr.reshape(-1,1))
coef = np.linalg.lstsq(X9, y_tr, rcond=None)[0]
xs = np.linspace(-3,3,200).reshape(-1,1)
ys = poly.transform(xs) @ coef
plt.scatter(x_tr, y_tr, label="训练点")
plt.plot(xs[:,0], ys, "r-", label="9次多项式(过拟合)")
plt.legend(); plt.show()




```

### 第 37 天 答案
```python
import numpy as np




np.random.seed(0)
c1 = np.random.randn(50,2) + np.array([1.5, 1.5])
c0 = np.random.randn(50,2) + np.array([-1.5,-1.5])
X = np.vstack([c0, c1])
y = np.array([0]*50 + [1]*50)

def sigmoid(z): return 1/(1+np.exp(-z))
w = np.array([1.0, 1.0]); b = 0.0
probs = sigmoid(X @ w + b)
preds = (probs > 0.5).astype(int)
print("前 5 个概率:", probs[:5].round(3))
print("前 5 个预测类别:", preds[:5])
print("准确率(随机参数):", (preds == y).mean())




```

### 第 38 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt




w = np.zeros(2); b = 0.0
lr = 0.1
for i in range(2000):
    z = X @ w + b
    yhat = sigmoid(z)
    err = yhat - y
    grad_w = (1/len(X)) * (X.T @ err)
    grad_b = (1/len(X)) * np.sum(err)
    w -= lr * grad_w
    b -= lr * grad_b
    if i % 400 == 0:
        loss = -np.mean(y*np.log(yhat+1e-9) + (1-y)*np.log(1-yhat+1e-9))
        print(f"第{i}轮 loss={loss:.3f}")

xx, yy = np.meshgrid(np.linspace(-4,4,100), np.linspace(-4,4,100))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = sigmoid(grid @ w + b).reshape(xx.shape)
plt.contourf(xx, yy, probs, levels=[0,0.5,1], alpha=0.3)
plt.scatter(c0[:,0], c0[:,1], c="blue", label="类0")
plt.scatter(c1[:,0], c1[:,1], c="red", label="类1")
plt.legend(); plt.show()
print("训练后准确率:", ((sigmoid(X@w+b)>0.5).astype(int)==y).mean())




```

### 第 39 天 答案
```python
preds = ((sigmoid(X @ w + b) > 0.5).astype(int))
TP = FP = FN = TN = 0
for p, t in zip(preds, y):
    if t == 1 and p == 1: TP += 1
    elif t == 0 and p == 1: FP += 1
    elif t == 1 and p == 0: FN += 1
    else: TN += 1
acc = (TP+TN)/(TP+FP+FN+TN)
print(f"混淆矩阵:\n TN={TN} FP={FP}\n FN={FN} TP={TP}")
print("准确率:", acc)




```

### 第 40 天 答案
```python
import numpy as np




np.random.seed(0)
a = np.random.randn(100) * 0.01
b = np.random.randn(100) * 100
Xraw = np.c_[a, b]
y = 3*a + 0.03*b + np.random.randn(100)*0.1

def standardize(mat):
    return (mat - mat.mean(0)) / mat.std(0)

def train_on(Xin, lr=0.05, epochs=2000):
    w = np.zeros(Xin.shape[1]); bb = 0.0
    for _ in range(epochs):
        yhat = Xin @ w + bb
        err = yhat - y
        w -= lr*(2/len(Xin))*Xin.T@err
        bb -= lr*(2/len(Xin))*np.sum(err)
    return np.mean(err**2)

loss_raw = train_on(Xraw)
loss_std = train_on(standardize(Xraw))
print("未标准化最终loss:", loss_raw)
print("标准化后最终loss:", loss_std)




```

### 第 41 天 答案
```python
import numpy as np
import matplotlib.pyplot as plt




def full_pipeline():
    np.random.seed(1)
    c1 = np.random.randn(60,2)+[1.5,1.5]; c0 = np.random.randn(60,2)+[-1.5,-1.5]
    X = np.vstack([c0,c1]); y = np.array([0]*60+[1]*60)
    idx = np.random.permutation(len(X))
    cut = int(0.8*len(X))
    xtr, xte, ytr, yte = X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]
    mu, sd = xtr.mean(0), xtr.std(0)
    xtr_s, xte_s = (xtr-mu)/sd, (xte-mu)/sd
    w = np.zeros(2); b = 0.0; lr = 0.1
    for _ in range(2000):
        ph = sigmoid(xtr_s @ w + b); e = ph - ytr
        w -= lr*(1/len(xtr))*xtr_s.T@e; b -= lr*(1/len(xtr))*np.sum(e)
    preds = (sigmoid(xte_s @ w + b) > 0.5).astype(int)
    acc = (preds == yte).mean()
    xx,yy = np.meshgrid(np.linspace(-4,4,80), np.linspace(-4,4,80))
    pp = sigmoid(np.c_[xx.ravel(),yy.ravel()] @ w + b).reshape(xx.shape)
    plt.contourf(xx,yy,pp, levels=[0,0.5,1], alpha=0.3)
    plt.scatter(xtr_s[:,0], xtr_s[:,1], c=ytr, cmap="bwr", s=15)
    plt.title(f"test acc={acc:.2f}"); plt.show()
    return w, b, acc

w, b, acc = full_pipeline()
print("测试准确率:", acc)




```

### 第 42 天 答案（写作要点，无唯一代码）
- 五个零件：①数据（带标签样本）②模型（可学习参数把 x 映射成 ŷ）③损失（衡量 ŷ 与 y 差距，回归 MSE / 分类交叉熵）④梯度（损失对每个参数的变化率，指出调整方向）⑤更新规则（沿负梯度小步调参，梯度下降）。
- 相同：都用"线性得分 + 梯度下降"框架。不同：模型上回归输出实数、分类经 sigmoid 输出概率；损失上回归用 MSE、分类用交叉熵；梯度上误差项分别来自 (ŷ−y)（回归）和 (ŷ−y)（逻辑回归，但 ŷ 经 sigmoid）。
- 过拟合：模型在训练集误差小、测试集误差大，背下了噪声而非学到规律。标准化：让不同量纲特征均值 0、标准差 1，损失地形更圆，梯度下降更快更稳。

---

---

## 第 7 周 参考答案：神经网络核心·前向传播（Day 43–49）

### Day 43 参考答案：感知机
```python
import numpy as np





def perceptron(x, w, b):
    z = np.dot(x, w) + b
    return 1 if z >= 0 else 0

# AND 门
w, b = np.array([0.5, 0.5]), -0.7
for x in [(0,0),(0,1),(1,0),(1,1)]:
    print("AND", x, perceptron(np.array(x), w, b))   # 0 0 0 1

# OR 门
w, b = np.array([0.5, 0.5]), -0.2
for x in [(0,0),(0,1),(1,0),(1,1)]:
    print("OR", x, perceptron(np.array(x), w, b))    # 0 1 1 1

# NOT 门（单输入）
w, b = np.array([-0.5]), 0.2
for x in [(0,),(1,)]:
    print("NOT", x, perceptron(np.array(x), w, b))   # 1 0




```

### Day 44 参考答案：单层感知机解 XOR 永远失败
```python
import numpy as np





X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)

def sigmoid(z): return 1/(1+np.exp(-z))

w = np.random.randn(2,1)*0.1
b = np.zeros((1,1))
lr = 0.1
for epoch in range(5000):
    p = sigmoid(X @ w + b)
    loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    dw = (1/len(X)) * X.T @ (p - y)
    db = (1/len(X)) * np.sum(p - y)
    w -= lr*dw; b -= lr*db

print("最终预测概率:\n", (X @ w + b).round(3))
print("损失:", round(float(loss), 4))   # 约 0.69，卡住不动




```
> 现象：4 个概率都接近 0.5，损失卡在 ~0.69。换随机种子结论一致——单层结构表达不了 XOR。

### Day 45 参考答案：参数计数
```python
import numpy as np




W1 = np.random.randn(2,2)*0.1; b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1; b2 = np.zeros(1)
print(W1.size + b1.size + W2.size + b2.size)   # 9




```
> 结构图：2 个输入圆 → 2 个隐藏圆 → 1 个输出圆；连线代表 W，每个圆旁标 b。

### Day 46 参考答案：前向传播（单个样本）
```python
import numpy as np




def sigmoid(x): return 1/(1+np.exp(-x))
x  = np.array([0.0, 1.0])
W1 = np.random.randn(2,2)*0.1; b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1; b2 = np.zeros(1)
z1 = x @ W1 + b1; a1 = np.tanh(z1)
z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
print(z1.shape, a1.shape, z2.shape, a2.shape)   # (2,) (2,) (1,) (1,)




```

### Day 47 参考答案：前向传播（批量）
```python
import numpy as np




X  = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
W1 = np.random.randn(2,2)*0.1; b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1; b2 = np.zeros(1)
Z1 = X @ W1 + b1; A1 = np.tanh(Z1)
Z2 = A1 @ W2 + b2; A2 = 1/(1+np.exp(-Z2))
print(Z1.shape, A2.shape)     # (4,2) (4,1)
print(A2)




```

### Day 48 参考答案：封装成类
```python
import numpy as np





class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(2,2)*0.1
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(2,1)*0.1
        self.b2 = np.zeros(1)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return 1/(1+np.exp(-z2))

    def summary(self):
        for name in ("W1","b1","W2","b2"):
            print(name, getattr(self, name).shape)
        print("参数总数:", self.W1.size+self.b1.size+self.W2.size+self.b2.size)

net = NeuralNetwork()
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
print(net.forward(X))
net.summary()     # 应打印 9




```

### Day 49 复盘日 参考答案
> 计算图形状链：
X  (4,2) →[@W1 + b1]→ Z1 (4,2) →[tanh]→ A1 (4,2) →[@W2 + b2]→ Z2 (4,1) →[sigmoid]→ A2 (4,1)
> 描述：4 个样本（行）先经 (2,2) 权重变 (4,2)，激活后仍是 (4,2)，再经 (2,1) 权重变 (4,1)，最后激活得 (4,1)。

---

---

## 第 8 周 参考答案：反向传播（Day 50–56）

### Day 50 参考答案：计算图与链式法则
```python
def f(x): return (x+1)**2
def df_dx(x): return 2*(x+1)
def num_grad(x, h=1e-5): return (f(x+h)-f(x-h))/(2*h)
print(df_dx(3.0), num_grad(3.0))          # 都≈8
# 3x²+2x+1：解析导数 6x+2
def f2(x): return 3*x**2+2*x+1
def df2(x): return 6*x+2
print(df2(3.0), num_grad(3.0) if False else (f2(3.001)-f2(2.999))/0.002)  # 都≈20




```

### Day 51 参考答案：输出层误差信号
```python
import numpy as np




def sigmoid(z): return 1/(1+np.exp(-z))
y_true, z = 1.0, 0.5
y_pred = sigmoid(z)
print(y_pred.round(4), (y_pred - y_true).round(4))   # 0.6225, -0.3775
# y_true=0, z=-1 时：y_pred≈0.2689, δ_L≈0.2689




```

### Day 52 参考答案：误差逐层回传
```python
import numpy as np




W2 = np.array([[0.3],[-0.2]])
delta_L = np.array([[0.1]])
z1 = np.array([0.5, -0.3])
delta_1 = (W2.T @ delta_L) * (1 - np.tanh(z1)**2)
print(delta_1.shape, delta_1)




```

### Day 53 参考答案：算出所有梯度
```python
import numpy as np




class Net:
    def __init__(self):
        self.W1=np.random.randn(2,2)*0.1; self.b1=np.zeros(2)
        self.W2=np.random.randn(2,1)*0.1; self.b2=np.zeros(1)
    def forward(self, x):
        self.z1=x@self.W1+self.b1; self.a1=np.tanh(self.z1)
        self.z2=self.a1@self.W2+self.b2; self.a2=1/(1+np.exp(-self.z2))
        return self.a2
    def backward(self, x, y):
        n=len(x); dL=self.a2-y
        dW2=self.a1.T@dL/n; db2=np.sum(dL,0,keepdims=True)/n
        d1=(dL@self.W2.T)*(1-self.a1**2)
        dW1=x.T@d1/n; db1=np.sum(d1,0,keepdims=True)/n
        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}
X=np.array([[0,0],[0,1],[1,0],[1,1]],float)
y=np.array([0,1,1,0],float).reshape(-1,1)
net=Net(); net.forward(X)
for k,v in net.backward(X,y).items(): print(k, v.shape)




```

### Day 54 参考答案：梯度校验
```python
import numpy as np




def sigmoid(z): return 1/(1+np.exp(-z))
def loss_fn(W1,b1,W2,b2,X,y):
    z1=X@W1+b1; a1=np.tanh(z1); z2=a1@W2+b2; a2=sigmoid(z2)
    return -np.mean(y*np.log(a2+1e-8)+(1-y)*np.log(1-a2+1e-8))
X=np.array([[0,0],[0,1],[1,0],[1,1]],float)
y=np.array([0,1,1,0],float).reshape(-1,1)
W1=np.random.randn(2,2)*0.1; b1=np.zeros(2)
W2=np.random.randn(2,1)*0.1; b2=np.zeros(1)
eps=1e-5
W2_p=W2.copy(); W2_p[0,0]+=eps
W2_m=W2.copy(); W2_m[0,0]-=eps
num=(loss_fn(W1,b1,W2_p,b2,X,y)-loss_fn(W1,b1,W2_m,b2,X,y))/(2*eps)
print("数值梯度 W2[0,0]:", num)   # 与 Day53 的 dW2[0,0] 应接近




```

### Day 55 参考答案：完整训练循环
```python
import numpy as np
import matplotlib.pyplot as plt




def sigmoid(z): return 1/(1+np.exp(-z))
X=np.array([[0,0],[0,1],[1,0],[1,1]],float)
y=np.array([0,1,1,0],float).reshape(-1,1)
W1=np.random.randn(2,2)*0.1; b1=np.zeros(2)
W2=np.random.randn(2,1)*0.1; b2=np.zeros(1)
lr,n=0.1,len(X); losses=[]
for _ in range(5000):
    z1=X@W1+b1; a1=np.tanh(z1); z2=a1@W2+b2; a2=sigmoid(z2)
    losses.append(-np.mean(y*np.log(a2+1e-8)+(1-y)*np.log(1-a2+1e-8)))
    dL=a2-y; dW2=a1.T@dL/n; db2=np.sum(dL,0,keepdims=True)/n
    d1=(dL@W2.T)*(1-a1**2); dW1=X.T@d1/n; db1=np.sum(d1,0,keepdims=True)/n
    W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
print("最终预测:\n", a2.round(3))




```

### Day 56 参考答案：XOR 训练成功
```python
import numpy as np




def sigmoid(z): return 1/(1+np.exp(-z))
X=np.array([[0,0],[0,1],[1,0],[1,1]],float)
y=np.array([0,1,1,0],float).reshape(-1,1)
W1=np.random.randn(2,2)*0.1; b1=np.zeros(2)
W2=np.random.randn(2,1)*0.1; b2=np.zeros(1)
lr,n=0.1,len(X)
for _ in range(10000):
    z1=X@W1+b1; a1=np.tanh(z1); z2=a1@W2+b2; a2=sigmoid(z2)
    dL=a2-y
    W1-=lr*(X.T@((dL@W2.T)*(1-a1**2))/n); b1-=lr*(np.sum((dL@W2.T)*(1-a1**2),0)/n)
    W2-=lr*(a1.T@dL/n); b2-=lr*(np.sum(dL,0)/n)
pred=(a2>0.5).astype(int)
print("预测类别:\n", pred.reshape(-1))
print("真实类别:\n", y.reshape(-1))
print("全部正确:", bool((pred==y).all()))




```

---

> 📌 **Day 1–56 参考答案到此。** 练完 Day 50–56 想核对就看这里。下一阶段（第 9 周）的教程 + 练习题继续「继续写第 9 周」时我会一并出。

---

# 第 9 周参考答案（Day 57–63）

## 第 57 天
```python
import numpy as np






def sigmoid(z): return 1/(1+np.exp(-z))
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)
np.random.seed(0)
W1 = np.random.randn(2, 8) * 0.1; b1 = np.zeros(8)
W2 = np.random.randn(8, 1) * 0.1; b2 = np.zeros(1)
for i in range(5000):
    z1 = X@W1 + b1; a1 = np.tanh(z1)
    z2 = a1@W2 + b2; out = sigmoid(z2)
    delta2 = (out - y) * out * (1 - out)
    delta1 = (delta2@W2.T) * (1 - a1**2)
    W2 -= 0.1 * (a1.T@delta2)/4; b2 -= 0.1 * delta2.sum(0)/4
    W1 -= 0.1 * (X.T@delta1)/4;  b1 -= 0.1 * delta1.sum(0)/4
print("loss:", -np.mean(y*np.log(out)+(1-y)*np.log(1-out)))

结论：隐藏层加宽（2→8）后，XOR 收敛更快更稳。权重不乘 0.1 易梯度爆炸。






```
## 第 58 天
```python
import numpy as np






def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
# 隐藏层前向：a1 = relu(X@W1+b1)
# 隐藏层反向：delta1 = (W2.T@delta2) * relu_grad(z1)
# 输出层仍 sigmoid；分别训 sigmoid 版与 ReLU 版，画两条 loss 曲线，ReLU 降更陡







```
## 第 59 天
```python
import numpy as np






# A 全 0：W1 = np.zeros((2,4))  → loss 卡 ~0.69（对称塌缩）
# B 过大：W1 = np.random.randn(2,4)*5 → 可能 NaN/发散
# C 适中：W1 = np.random.randn(2,4)*0.1 → 正常下降
# 偏置 b1 全 0 是 OK 的







```
## 第 60 天
```python
import numpy as np






n, batch = 4, 2
for epoch in range(2000):
    idx = np.random.permutation(n)          # 每轮打乱
    for s in range(0, n, batch):
        b_idx = idx[s:s+batch]
        xb, yb = X[b_idx], y[b_idx]
        # 前向/反向用 xb, yb；梯度除以 len(b_idx) 平均
# 1 epoch = 全部样本过一遍；结果仍能学会 XOR







```
## 第 61 天
```python
from sklearn.datasets import fetch_openml






mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(X.shape, y.shape)          # (70000, 784) (70000,)
X = X / 255.0                    # 归一化，必须做
# 可视化前 10 张：imshow(X[i].reshape(28,28), cmap='gray')，标题 y[i]
# 划分：X[:60000] 训练，X[60000:] 测试







```
## 第 62 天
```python
import numpy as np






def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True)); return e/e.sum(1, keepdims=True)
def cross_entropy(y_oh, p):
    return -np.mean(np.sum(y_oh*np.log(p + 1e-8), axis=1))
# 前向：a1 = relu(X@W1+b1); out = softmax(a1@W2+b2)
# 反向：delta2 = out - y_oh; delta1 = (W2.T@delta2)*relu_grad(z1)
# 标签 One-Hot：y_oh = np.eye(10)[y]
# 验证 out 每行和≈1；delta2 = out - y_oh（softmax+交叉熵的免费结果）







```
## 第 63 天
```python
import numpy as np






X_train, y_train = X[:10000]/255.0, y[:10000]
X_test,  y_test  = X[60000:61000]/255.0, y[60000:61000]
y_oh = np.eye(10)[y_train]
np.random.seed(0)
W1 = np.random.randn(784,64)*0.1; b1 = np.zeros(64)
W2 = np.random.randn(64,10)*0.1;  b2 = np.zeros(10)
n, batch, lr = len(X_train), 64, 0.1
for epoch in range(20):
    idx = np.random.permutation(n)
    for s in range(0, n, batch):
        b = idx[s:s+batch]
        a1 = relu(X_train[b]@W1+b1); out = softmax(a1@W2+b2)
        d2 = out - y_oh[b]; d1 = (d2@W2.T)*relu_grad(a1)
        W2 -= lr*(a1.T@d2)/len(b); b2 -= lr*d2.sum(0)/len(b)
        W1 -= lr*(X_train[b].T@d1)/len(b); b1 -= lr*d1.sum(0)/len(b)
    # 打印 train loss
a1 = relu(X_test@W1+b1); out = softmax(a1@W2+b2)
acc = (out.argmax(1) == y_test).mean()
print("测试准确率:", acc)     # 通常 0.92~0.95

打卡标准：测试准确率 ≥ 0.90。

---

> 📌 **Day 1–63 参考答案到此。** 练完 Day 57–63 想核对就看这里。下一阶段（第 10 周）的教程 + 练习题继续「继续写第 10 周」时我会一并出。

---

# 第 10 周参考答案（Day 64–70）






```
## 第 64 天
```python
import numpy as np
import torch






t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t, t.shape, t.dtype)
print(torch.zeros(2,3)); print(torch.ones(2,2)); print(torch.rand(2,3)); print(torch.randn(2,3)*0.1)
a = np.array([1.0,2.0,3.0]); t2 = torch.from_numpy(a); print(t2, t2.numpy())
print(t+1, t*2, t@t)







```
## 第 65 天
```python
import torch






x = torch.tensor(2.0, requires_grad=True)
y = x**2; y.backward(); print(x.grad)            # 4.0
x.grad.zero_()
y = 3*x**3 + 2*x; y.backward(); print(x.grad)   # 38.0







```
## 第 66 天
```python
import torch
import torch.nn as nn






model = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
print(model)
X = torch.tensor([[0.,0],[0,1],[1,0],[1,1]])
print(model(X))
# 隐藏层 8 个：nn.Sequential(nn.Linear(2,8), nn.Tanh(), nn.Linear(8,1), nn.Sigmoid())







```
## 第 67 天
```python
import numpy as np
import torch
import torch.nn as nn






x_np = np.linspace(0,10,100,dtype=np.float32).reshape(-1,1)
y_np = (3*x_np + 1 + np.random.randn(100,1)*0.5).astype(np.float32)
X = torch.from_numpy(x_np); y = torch.from_numpy(y_np)
model = nn.Linear(1,1); loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(2000):
    pred = model(X); loss = loss_fn(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
print("w,b =", model.weight.item(), model.bias.item())   # ≈3, 1







```
## 第 68 天
```python
import torch
import torch.nn as nn






X = torch.tensor([[0.,0],[0,1],[1,0],[1,1]]); y = torch.tensor([[0.],[1],[1],[0]])
model = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(5000):
    opt.zero_grad(); pred = model(X); loss = loss_fn(pred, y)
    loss.backward(); opt.step()
    if epoch % 1000 == 0: print(epoch, loss.item())
print("预测:\n", model(X).detach().round())







```
## 第 69 天
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset






dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(3000):
    for xb, yb in loader:
        opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
torch.save(model.state_dict(), "xor_model.pt")
m2 = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
m2.load_state_dict(torch.load("xor_model.pt"))
print("加载后预测:\n", m2(X).detach().round())







```
## 第 70 天
```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn






# 用 5 步模板，2-4-1 + ReLU 解 XOR：
model = nn.Sequential(nn.Linear(2,4), nn.ReLU(), nn.Linear(4,1), nn.Sigmoid())
loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
losses = []
for epoch in range(5000):
    opt.zero_grad(); pred = model(X); loss = loss_fn(pred, y)
    loss.backward(); opt.step(); losses.append(loss.item())
print(model(X).detach().round())
# 画图：plt.plot(losses); plt.show()
# 总结：框架用 loss.backward() 替手写 backward()，用 opt.step() 替手写逐参数更新；前向/损失/循环结构不变。


---

> 📌 **Day 1–70 参考答案到此。** 练完 Day 64–70 想核对就看这里。下一阶段（第 11 周）的教程 + 练习题继续「继续写第 11 周」时我会一并出。

---

# 第 11 周（Day 71–77）参考答案






```
## 第 71 天
```python
import numpy as np







def conv2d(img, kernel):
    H, W = img.shape
    K = kernel.shape[0]
    out_h, out_w = H - K + 1, W - K + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(img[i:i+K, j:j+K] * kernel)
    return out

np.random.seed(0)
img = np.random.rand(8, 8)
edge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
print(conv2d(img, edge).shape)        # (6, 6)
print(conv2d(np.random.rand(8,8), np.full((5,5),1/25)).shape)  # (4, 4)

要点：边缘核在像素突变处输出大、平坦处≈0；5×5 核使 8×8→4×4。






```
## 第 72 天
```python
import torch
import torch.nn as nn






# 手算：(28-3+2*1)/1+1 = 28 ；步长2：(28-3+0)/2+1 = 13.xx -> 13
print((28-3+2)//1+1, (28-3+0)//2+1)   # 28 13
layer = nn.Conv2d(1, 4, 3, padding=1, stride=1)
print(layer(torch.randn(1,1,28,28)).shape)   # [1,4,28,28]







```
## 第 73 天
```python
import torch
import torch.nn as nn






model = nn.Sequential(
    nn.Conv2d(1,4,3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(4,8,3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(8*6*6,64), nn.ReLU(), nn.Linear(64,10),
)
x = torch.randn(1,1,32,32)
for n,l in model.named_children():
    x = l(x); print(n, tuple(x.shape))
# 第1层参数：3*3*1*4 + 4 = 40
print("params:", 3*3*1*4 + 4)







```
## 第 74 天
```python
import torch
import torch.nn as nn






cnn = nn.Sequential(
    nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16*7*7,64), nn.ReLU(), nn.Linear(64,10),
)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for X,y in train_loader:
    X = X.unsqueeze(1)
    opt.zero_grad(); loss_fn(cnn(X), y).backward(); opt.step()
# 测试时同样 X.unsqueeze(1)；准确率应比 Day63 的 MLP 高 1~3%







```
## 第 75 天
```python
import numpy as np
import matplotlib.pyplot as plt






w = cnn[0].weight.detach().numpy()              # [8,1,3,3]
w = (w - w.min()) / (w.max() - w.min())
fig, ax = plt.subplots(1,8, figsize=(12,2))
for i in range(8):
    ax[i].imshow(w[i,0], cmap="gray"); ax[i].axis("off")
plt.show()







```
## 第 76 天（选做）
```python
import torch
from torchvision import transforms
from torchvision import models






net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); net.eval()
prep = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
x = prep(Image.open("本地图片.jpg")).unsqueeze(0)
with torch.no_grad():
    top5 = net(x).softmax(1)[0].topk(5)
# 改 43 类交通标志：把最后一层 Linear(64,1000) 换成 Linear(64,43)，数据集换 GTSRB

要点：预训练模型要求 224×224 且做了特定归一化；改类别数 + 换数据集即可套用到 GTSRB。






```
## 第 77 天
推导链（输入 1×28×28）：
- Conv(8,3,padding1): 1×8×28×28
- MaxPool2: 1×8×14×14
- Conv(16,3,padding1): 1×16×14×14
- MaxPool2: 1×16×7×7
- Flatten: 16*7*7=784
- FC(64)→FC(10)
64×64 输入时：Conv→1×8×64×64→Pool 1×8×32×32→Conv 1×16×32×32→Pool 1×16×16×16→Flatten 16*16*16=4096。

---

> 📌 **Day 1–77 参考答案到此。** 练完 Day 71–77 想核对就看这里。下一阶段（第 12 周）的教程 + 练习题继续「继续写第 12 周」时我会一并出。

---

# 第 12 周（Day 78–84）参考答案

## 第 78 天
```python
import torch
import torch.nn as nn

cnn_drop = nn.Sequential(
    nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16*7*7,128), nn.ReLU(),
    nn.Dropout(0.2), nn.Linear(128,10),
)
# 训练 3 epoch 后对比；评估前务必 model.eval() 关闭 Dropout




```
要点：Dropout 只在训练生效；p 在 0.2~0.5 常用；评估切 eval() 否则结果飘。

## 第 79 天
```python
import torch






results = {}
for lr in [0.01, 0.001]:
    for bs in [32, 128]:
        torch.manual_seed(42)            # 每组固定种子
        results[(lr, bs)] = train_and_eval(lr, bs, epochs=2)
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(k, round(v, 3))

要点：通常 lr=0.001 更稳；小 batch 泛化可能更好但慢；固定种子保证可比。






```
## 第 80 天
```python
import torch
import torch.nn as nn






opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
losses = []
for epoch in range(6):
    losses += train_one_epoch(cnn, opt)   # 记录每步 loss
    scheduler.step()                       # 每个 epoch 后调用
# 画 losses 曲线，对比固定 lr 版本

要点：step() 必须放在 epoch/batch 之后；衰减让后期更平滑更低。






```
## 第 81 天
```python
import matplotlib.pyplot as plt
import os
from collections import Counter






counts = Counter()
for c in os.listdir("data/gtsrb/train"):
    counts[c] = len(os.listdir(f"data/gtsrb/train/{c}"))
print("类别数:", len(counts))   # 43
# 用 plt.bar 画分布；第一次先用子集(前10类)跑通







```
## 第 82 天
```python
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, Dataset






class GTSRB(Dataset):
    def __init__(self, root, tfm):
        self.items, self.tfm = [], tfm
        for c in os.listdir(root):
            for f in os.listdir(os.path.join(root, c)):
                self.items.append((os.path.join(root, c, f), int(c)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tfm(Image.open(p).convert("RGB")), y
tfm = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3,[0.5]*3)])
loader = DataLoader(GTSRB("data/gtsrb/train", tfm), batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)   # [32,3,32,32] [32]







```
## 第 83 天
```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns






model = nn.Sequential(
    nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32*8*8,128), nn.ReLU(),
    nn.Dropout(0.3), nn.Linear(128,43),
)
# 训练几个 epoch；评估：
y_true, y_pred = [], []
with torch.no_grad():
    for X,y in test_loader:
        y_true += y.tolist(); y_pred += model(X).argmax(1).tolist()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, cmap="Blues"); plt.show()

要点：第一层 in_channels=3（彩色）、最后 out=43；类别不均衡时看混淆矩阵而非仅总准确率。






```
## 第 84 天
参考要点（写在你的笔记里即可，无需代码）：
- 3 个成果示例：手写 XOR 成功（Day56）、MNIST 手写 NN 90%（Day63）、CNN+GTSRB 跑通（Day83）。
- 300 字总结：神经网络 = 多层"加权求和+激活"的叠加；前向算预测、算损失，反向传播把损失对每层权重的梯度算出来，梯度下降沿梯度反方向更新权重；重复多轮就学会。CNN 用局部卷积尊重图像结构。
- 下一步方向：目标检测 YOLO / 语义分割（车道线）/ Transformer。

---

> 🎊 **Day 1–84 参考答案全部完成。** 整个 84 天计划的练习核对本到此结束。恭喜走完从"二级 Python"到"能训练交通标志分类模型"的完整路径！
