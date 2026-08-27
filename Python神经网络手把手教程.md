# Python → 神经网络 · 手把手教程（Day 1–14）

> 配套：`Python神经网络学习计划.md`  
> 使用方法：在已激活的 `E:\conda\envs\dl` 环境里，用 VS Code / Jupyter 打开，**每天只做当天那一块**，把代码逐个 cell 跑一遍、改一改、再看输出。  
> 每天结构：🎯目标 → 📖理论(8分钟) → 💻动手(9分钟) → 👀你该看到 → ⚠️常见坑 → ✅打卡  
> 约定：所有代码默认已 `import numpy as np`、`import matplotlib.pyplot as plt`。

---

## 第 1 天（第1周·第1天）ndarray 基础

🎯 **目标**：认识 NumPy 最核心的数据结构 `ndarray`（多维数组），学会创建它、查看它的形状与类型。

📖 **理论（8分钟）**

- 神经网络里一切都是「数字矩阵」，NumPy 的 `ndarray` 就是装这些数字的高效容器，比 Python 列表快得多。
- 关键属性：
  - `.shape`：形状，比如 `(3, 4)` 表示 3 行 4 列；
  - `.dtype`：元素类型，比如 `int64`、`float64`；
  - `.ndim`：维度数（1 维=向量，2 维=矩阵）。
- 常用创建函数：`np.array(列表)`、`np.arange(n)`（类似 range，但返回数组）、`np.zeros`、`np.ones`、`np.full((行,列), 值)`（创建全是指定值的数组）、`np.reshape`。

💻 **动手（9分钟）**

```python
import numpy as np

# 1) 用列表创建 3x3 矩阵
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("a 的形状:", a.shape)
print("a 的类型:", a.dtype)
print("a 的维数:", a.ndim)

# 2) arange 生成 0~11，再 reshape 成 3x4
b = np.arange(12).reshape(3, 4)
print("\nb =\n", b)
print("b 的形状:", b.shape)

# 3) 全 0 / 全 1 / 全指定值
print("\nzeros(2,3) =\n", np.zeros((2, 3)))
print("ones(2,2, dtype=int) =\n", np.ones((2, 2), dtype=int))
print("full((2,3), 7) =\n", np.full((2, 3), 7))   # 2行3列，全部填 7
```

👀 **你该看到**：`a` 是 3×3 整数矩阵，`a.shape=(3, 3)`、`a.dtype=int64`；`b` 是 0~11 排成 3 行 4 列。

⚠️ **常见坑**

- `np.array([1,2],[3,4])` 会报错！必须写成 `np.array([[1,2],[3,4]])`（外层一个中括号包住所有行）。
- `reshape` 的元素总数必须一致：`arange(12)` 能 reshape 成 `(3,4)` 或 `(2,6)`，但 `reshape(3,5)` 会报「尺寸不匹配」。

✅ **打卡**：能不看资料，用 `np.arange` 做出任意形状的矩阵，并说出 `.shape/.dtype` 是什么。

---

## 第 2 天（第1周·第2天）索引、切片与布尔索引

🎯 **目标**：从数组里「取数」——按位置取、按条件取。

📖 **理论（8分钟）**

- 取单个元素：`a[行, 列]`（从 0 开始计数）。
- 切片：`a[0:2, 1:3]` 取前两行、第 2~3 列（左闭右开，和 Python 列表一致）。
- **布尔索引**（今天重点）：`a[a > 5]` 会返回一个「所有大于 5 的元素」组成的一维数组。这是后面做掩码、筛选数据的常用技巧。

💻 **动手（9分钟）**

```python
import numpy as np

m = np.random.rand(5, 5)   # 5x5 随机数，范围 0~1
print("原始矩阵 m =\n", np.round(m, 2))

# 1) 取第 0 行
print("\n第 0 行:", m[0])

# 2) 布尔索引：取出所有大于 0.5 的元素
big = m[m > 0.5]
print("\n大于 0.5 的元素:", np.round(big, 2))
print("大于 0.5 的元素个数:", big.size)

# 3) 把大于 0.5 的位置标成 1，否则 0（后面会常用）
mask = (m > 0.5).astype(int)
print("\n掩码 mask =\n", mask)
```

👀 **你该看到**：`big` 是一维数组，里面都是 0.5~1 之间的数；`big.size` 是它们的总数。

⚠️ **常见坑**

- 布尔索引返回的是**一维**数组，哪怕原始是二维的。想保持二维形状，要用 `np.where` 或掩码赋值：`m[m>0.5] = 1`。
- `m > 0.5` 本身是一个布尔矩阵（True/False），`.astype(int)` 把它变成 1/0。

✅ **打卡**：能用布尔索引筛出「满足条件」的元素，并解释为什么结果是扁平的一维数组。

---

## 第 3 天（第1周·第3天）广播机制 Broadcasting

🎯 **目标**：理解 NumPy 怎么让「形状不同的数组」直接相加，这是后面矩阵运算提速的关键。

📖 **理论（8分钟）**

- 广播规则：**从最后一个维度往前对齐**，维度大小要么相等、要么其中一个是 1，才能广播。
- 例子：矩阵 `(3,4)` 想让一个长度 3 的向量加到「每一列」上 → 向量得是 `(3,1)`（3 行 1 列），这样 `(3,1)` 能沿列方向复制到 `(3,4)`。
- 想加到「每一行」上 → 向量得是 `(1,4)`。
- 如果形状对齐不上（如 `(3,)` 和 `(3,4)` 直接相加），会报 `ValueError`。

💻 **动手（9分钟）**

```python
import numpy as np

M = np.ones((3, 4)) * 10     # 3x4 全 10
print("M =\n", M)

# 加到每一列：向量形状 (3,1)
v_col = np.array([1, 2, 3]).reshape(3, 1)
print("\nM + 每列加 [1,2,3] =\n", M + v_col)

# 加到每一行：向量形状 (1,4)
v_row = np.array([1, 2, 3, 4]).reshape(1, 4)
print("\nM + 每行加 [1,2,3,4] =\n", M + v_row)

# 会报错的例子：形状对不上
try:
    np.array([1, 2, 3]) + M   # (3,) 和 (3,4) 末维 3≠4，无法广播
except ValueError as e:
    print("\n报错信息:", e)
```

👀 **你该看到**：加 `v_col` 后每一列分别多了 1/2/3；加 `v_row` 后每一行分别多了 1/2/3/4；最后一段打印 ValueError 报错。

⚠️ **常见坑**

- 最迷惑的就是「(3,) 加不到 (3,4) 上」。记住：**先 `reshape` 成 (3,1) 或 (1,3) 再运算**。
- 广播不会真的复制数据，只是「假装」复制，所以既省内存又快。

✅ **打卡**：能画出 (3,1) 和 (1,4) 分别是怎么广播到 (3,4) 的，并解释为什么 `(3,)` 会报错。

---

## 第 4 天（第1周·第4天）矩阵乘法 vs 逐元素乘

🎯 **目标**：分清楚 `@`（矩阵乘）和 `*`（逐元素乘），这是以后写神经网络最易混的点。

📖 **理论（8分钟）**

- `A @ B` 或 `A.dot(B)`：矩阵乘法，要求 A 的列数 = B 的行数。结果第 i 行第 j 列 = A 的第 i 行 与 B 的第 j 列 对应相乘再求和。
- `A * B`：逐元素乘，要求形状**完全相同**，对应位置相乘。
- 转置 `A.T`：行列互换。

💻 **动手（9分钟）**

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print("A@B =\n", A @ B)
print("A*B =\n", A * B)
print("A.T =\n", A.T)

# 手算验证第一行第一列: 1*5 + 2*7 = 19
print("\n手算 A@B[0,0] =", 1*5 + 2*7)
```

👀 **你该看到**：`A@B = [[19,22],[43,50]]`（1*5+2*7=19，1*6+2*8=22，3*5+4*7=43，3*6+4*8=50）；`A*B` 是对应位置相乘 `[[5,12],[21,32]]`。

⚠️ **常见坑**

- 神经网络里权重乘输入是 `@`，损失函数里算误差平方是 `*`。混用会得到形状不对的报错或错误结果。
- `A*B` 要求形状一致；不同形状想「对应乘」只能用广播或 `@`。

✅ **打卡**：闭眼能说出 `@` 和 `*` 的三个区别（维度要求、计算方式、结果含义）。

---

## 第 5 天（第1周·第5天）随机数、种子与统计

🎯 **目标**：会用随机数生成数据，并理解「种子」为什么重要（保证结果可复现）。

📖 **理论（8分钟）**

- `np.random.randn(n)`：从**标准正态分布**（均值 0、标准差 1）抽 n 个数。
- `np.random.seed(k)`：**固定随机种子**，之后随机序列就确定不变了 → 你跑和别人跑结果一致，方便调试。
- 统计：`x.mean()` 均值、`x.std()` 标准差。

💻 **动手（9分钟）**

```python
import numpy as np

np.random.seed(0)
x = np.random.randn(1000)
print("seed=0 时: 均值=%.3f, 标准差=%.3f" % (x.mean(), x.std()))

np.random.seed(42)
y = np.random.randn(1000)
print("seed=42 时: 均值=%.3f, 标准差=%.3f" % (y.mean(), y.std()))

# 同样种子，结果一定相同
np.random.seed(7)
a = np.random.randn(3)
np.random.seed(7)
b = np.random.randn(3)
print("同种子两次抽样是否相同:", np.array_equal(a, b), a)
```

👀 **你该看到**：两组均值都接近 0、标准差接近 1（抽 1000 个的近似）；同种子下 `a` 和 `b` 完全一样。

⚠️ **常见坑**

- 不设种子时，每次运行随机数都不同，调bug时很痛苦。**写练习时养成先 `seed` 的习惯**。
- `rand`（0~1 均匀）和 `randn`（标准正态）不一样，别混。

✅ **打卡**：能解释「种子」的作用，并用代码证明「同种子 → 同结果」。

---

## 第 6 天（第1周·第6天）matplotlib 画图入门

🎯 **目标**：学会画折线图和散点图——以后看 loss 下降、看数据分布全靠它。

📖 **理论（8分钟）**

- `plt.plot(x, y)` 画折线；`plt.scatter(x, y)` 画散点。
- `plt.xlabel/ylabel/title/legend()` 加标注；`plt.show()` 显示。
- 在 Jupyter 里常加 `%matplotlib inline` 让图直接嵌在 notebook 中。
- 想让样本点带「噪声」（偏离曲线），给 y 加一点随机量即可，例如 `y_noise = np.sin(pts) + (np.random.rand(30)-0.5)*0.5`。`np.random.rand` 本天已学过，它给 0~1 的均匀随机数，`-0.5` 再乘系数就把噪声控制在 ±0.5 附近。

💻 **动手（9分钟）**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)   # 0 到 2π 等分为 100 个点
plt.plot(x, np.sin(x), label="sin(x)")

# 随机取 30 个点当"样本"
pts = np.random.rand(30) * 2 * np.pi
plt.scatter(pts, np.sin(pts), color="red", label="样本点(无噪声)")

# 想模拟"带测量误差的真实数据"，就给 y 加一点随机噪声：
# np.random.rand(30) 给 0~1 的均匀随机数，减 0.5 得到 -0.5~0.5，再乘 0.3 就把噪声控制在 ≈±0.15
y_noise = np.sin(pts) + (np.random.rand(30) - 0.5) * 0.3
plt.scatter(pts, y_noise, color="green", label="样本点(带噪声)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("sin 曲线与样本")
plt.legend()
plt.show()
```

👀 **你该看到**：一条蓝色 sin 曲线，上面撒了 30 个红色点。

⚠️ **常见坑**

- 在普通 `.py` 脚本里忘记 `plt.show()` 图就不弹出来；在 Jupyter 里不加 `show()` 也能显示。
- `plt` 状态是「累积」的：上张图的设置会带到下张图。新图开头加 `plt.clf()` 或开新 cell 时注意清理。

✅ **打卡**：能独立画出一条函数曲线 + 一组随机散点，并加上坐标轴标签。

---

## 第 7 天（第1周·复盘日）不看资料重写

🎯 **目标**：把第 3、4、6 天的代码**凭记忆重写**，巩固肌肉记忆（复盘日不学新东西）。

📝 **今日任务**

1. 重写「广播」：创建一个 (3,4) 矩阵，让 `[1,2,3]` 加到每一列。
2. 重写「矩阵乘 vs 逐元素乘」：自己定义 A、B，打印 `A@B` 和 `A*B`。
3. 重写「sin 曲线 + 散点图」。
4. 写 3 句话总结本周：①ndarray 是什么 ②广播解决什么问题 ③@ 和 * 的区别。

✅ **打卡**：三份代码都能一次跑通，总结写在笔记里。

---

## 第 8 天（第2周·第1天）函数进阶

🎯 **目标**：把重复代码封装成函数，带默认参数，为后面写「模型」打基础。

📖 **理论（8分钟）**

- 默认参数：`def f(x, seed=0)` 调用时 `seed` 可给可不给。
- `*args` / `**kwargs`：接收任意个位置/关键字参数（先认识，后面才常用）。
- 把上周的随机矩阵代码包成 `make_matrix(rows, cols, seed)`。

> 📌 **二级补充**：`*args` 是二级不教的"变长参数"——`def total(*nums)` 能把 `total(1,2,3,4)` 里的所有数收集成元组 `nums=(1,2,3,4)` 再求和。看不懂先翻 `Python二级衔接补充.md` 第 2 节，不耽误今天练习。

💻 **动手（9分钟）**

```python
import numpy as np

def make_matrix(rows, cols, seed=0):
    np.random.seed(seed)
    return np.random.rand(rows, cols)

m1 = make_matrix(2, 3)
m2 = make_matrix(2, 3, seed=1)
print("默认种子:\n", np.round(m1, 2))
print("\nseed=1:\n", np.round(m2, 2))

# 变长参数示例
def total(*nums):
    return sum(nums)
print("\ntotal(1,2,3,4) =", total(1, 2, 3, 4))
```

👀 **你该看到**：`m1`、`m2` 内容不同（种子不同）；`total(1,2,3,4)=10`。

⚠️ **常见坑**

- 默认参数如果是**可变对象**（如列表 `[]`）会有「共享」陷阱，目前用数字当默认值没问题，先记住别用可变默认参数。

✅ **打卡**：能把一个重复操作封装成带默认参数的函数。

---

## 第 9 天（第2周·第2天）类与对象（上）

🎯 **目标**：用一个 `Linear` 类来装「权重 W 和偏置 b」——这正是神经网络里每一层的样子。

📖 **理论（8分钟）**

- `class` 像一张「图纸」，`__init__` 是「建房子时先做什么」（初始化属性）。
- 实例方法第一个参数永远是 `self`，代表「这个对象自己」。
- 我们要建一个线性层：`y = x @ W + b`，把 W、b 作为属性存起来。

> 📌 **二级补充**：`class` / `self` / `__init__` 是二级几乎不教的"面向对象"，但它是后面所有神经网络代码的地基。别慌——`Python二级衔接补充.md` **第 1 节**直接用今天这个 `Linear` 类逐行拆给你看（含 `self` 的"蛋糕比喻"），先读完再敲代码会顺很多。

💻 **动手（9分钟）**

```python
import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(in_dim, out_dim) * 0.1   # 乘 0.1 让初始值小一点
        self.b = np.zeros(out_dim)

    def forward(self, x):
        return x @ self.W + self.b

# 试一下：输入 2 维，输出 3 维
layer = Linear(in_dim=2, out_dim=3)
x = np.array([[1.0, 2.0]])
print("W =\n", layer.W)
print("b =", layer.b)
print("输出 =", layer.forward(x))
```

👀 **你该看到**：`W` 是 2×3 的小数值矩阵，`b` 是 3 个 0，输出是 1×3 的数组。

⚠️ **常见坑**

- 权重初始化乘 `0.1` 很重要：太大训练时容易「梯度爆炸」。这里先记住这个习惯。
- `x` 要是二维的（哪怕只有 1 个样本也要写成 `[[1,2]]`），否则 `x @ W` 形状会乱。

✅ **打卡**：能写出一个 `Linear` 类，创建实例并成功 `forward` 一次。

---

## 第 10 天（第2周·第3天）类与对象（下）：`__call__`

🎯 **目标**：让层可以像函数一样被调用（`layer(x)`），更贴近 PyTorch 的写法。

📖 **理论（8分钟）**

- 定义 `__call__` 后，实例就能当函数用：`obj(参数)` 实际调用 `__call__`。
- 这样 `layer(x)` 和 `layer.forward(x)` 等价，后面写网络会非常顺手。

💻 **动手（9分钟）**

```python
import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros(out_dim)

    def forward(self, x):
        return x @ self.W + self.b

    def __call__(self, x):      # 加了这一行
        return self.forward(x)

layer = Linear(2, 3)
x = np.array([[1.0, 2.0]])
print("layer.forward(x) =", layer.forward(x))
print("layer(x)        =", layer(x))   # 两种写法结果一样
```

👀 **你该看到**：两行输出完全相同。

⚠️ **常见坑**

- 别忘了 `__call__` 里要写 `return` 把结果返回出去，否则调用它拿不到任何结果。

✅ **打卡**：能给自己写的类加上 `__call__`，并用 `对象(参数)` 的方式调用。

---

## 第 11 天（第2周·第4天）异常处理与文件读写

🎯 **目标**：学会用 `try/except` 兜底错误，并用 NumPy 保存/读取数组。

📖 **理论（8分钟）**

- 程序可能出错（如文件不存在），`try/except` 能「接住」错误不让程序崩。
- `np.save('名.npy', 数组)` 存盘，`np.load('名.npy')` 读回。

💻 **动手（9分钟）**

```python
import numpy as np

a = np.arange(10)
np.save("a.npy", a)
b = np.load("a.npy")
print("读回的数组:", b)
print("和原数组一致:", np.array_equal(a, b))

# 接住"文件不存在"的错误
try:
    np.load("不存在的文件.npy")
except FileNotFoundError as e:
    print("捕获到错误:", e)
```

👀 **你该看到**：读回的数组和 `a` 一致；第二段打印捕获到的 `FileNotFoundError`。

⚠️ **常见坑**

- `np.save` 会自动加 `.npy` 后缀；`np.load` 也要带 `.npy`。
- 别用 `except:` 裸捕获（会连 Ctrl+C 都拦住），指定具体异常类型。

✅ **打卡**：能保存并读回一个数组，并用 `try/except` 处理一个会报错的操作。

---

## 第 12 天（第2周·第5天）Jupyter 使用技巧

🎯 **目标**：会用 Jupyter Notebook 组织「代码 + 讲解笔记」，以后每天练习都建议用它。

📖 **理论（8分钟）**

- 启动：`conda activate E:\conda\envs\dl` 后输入 `jupyter notebook`，浏览器自动打开。
- Cell 分两种：**Code**（写代码）和 **Markdown**（写笔记，支持 `# 标题`、`**加粗**`）。
- 常用快捷键：`Ctrl+Enter` 运行当前 cell，`Shift+Enter` 运行并跳下一个，`A` 上方插入、`B` 下方插入。
- 在 cell 顶部加 `%matplotlib inline` 让图直接显示。

💻 **动手（9分钟）**

```python
# 这是第一个 Code cell，先跑这一行
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt

# 再新建一个 Markdown cell，写： "## 第12天：Jupyter 试水"
# 然后新建 Code cell 跑下面：
x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x))
plt.title("我的第一个 notebook 图")
plt.show()
```

👀 **你该看到**：浏览器里出现一张 sin 图，上方有你写的 Markdown 标题。

⚠️ **常见坑**

- 变量在所有 cell 间共享；改了某个 cell 的变量，要重新按顺序跑相关 cell，不然结果对不上。
- Kernel 卡死可选 `Kernel → Restart & Run All` 重来。

✅ **打卡**：能新建一个 notebook，包含 1 个 Markdown 笔记 + 1 段能跑出图的代码。

---

## 第 13 天（第2周·第6天）综合：欧氏距离与余弦相似度

🎯 **目标**：动手实现两个在机器学习里极常用的「距离/相似度」函数，综合练习前面的知识。

📖 **理论（8分钟）**

- **欧氏距离**：两点直线距离，`||a-b||`（向量差的长度）。
- **余弦相似度**：衡量两个向量「方向」是否一致，范围 [-1,1]，`a·b / (|a|·|b|)`。越接近 1 越同向。
- 用 `np.linalg.norm` 求向量长度。

💻 **动手（9分钟）**

```python
import numpy as np

def euclidean(a, b):
    return np.linalg.norm(a - b)

def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print("欧氏距离:", euclidean(a, b))
print("余弦相似度:", cosine_sim(a, b))

# 验证：相同向量余弦相似度应=1
print("a 与自身余弦:", cosine_sim(a, a))
```

👀 **你该看到**：欧氏距离约 5.196；余弦相似度约 0.974（a、b 方向接近）；`a` 与自身余弦 = 1.0。

⚠️ **常见坑**

- 余弦相似度只关心**方向**不关心长度：把 `b` 乘 10 倍，余弦不变，欧氏距离变大。
- 向量里有 0 时 `norm` 可能为 0，除零会报错（目前数据不会，先知道有这坑）。

✅ **打卡**：能独立写出这两个函数，并解释「余弦相似度为什么不受向量长度影响」。

---

## 第 14 天（第2周·复盘日）自测

🎯 **目标**：检验前两周是否扎实。

📝 **自测（不翻资料，答不出就回看对应天）**

1. 广播规则是什么？画一个 `(3,1)` 广播到 `(3,4)` 的例子。
2. `@` 和 `*` 的区别，各给一个 2×2 实例。
3. 种子 `seed` 有什么用？写两行代码证明。
4. 写一个 `Linear` 类（含 `__init__/forward/__call__`），输入 3 维输出 2 维并跑通。

✅ **打卡**：4 题都能独立做对，并保存成一个 `week02_review.ipynb` 或 `.py`。

---

---

## 第 15 天（第3周·第1天）向量点积：几何意义

🎯 **目标**：理解 a·b 不只是"对应相乘再求和"，它还等于 |a|·|b|·cosθ，反映两向量方向的接近程度——这是后面余弦相似度、注意力机制的地基。

📖 **理论（8分钟）**

- 点积两种等价理解：
  1. 代数：`a·b = Σ aᵢbᵢ`（对应相乘相加），NumPy 里就是 `a @ b`。
  2. 几何：`a·b = |a|·|b|·cosθ`，即 a 的长度 × b 在 a 方向上的投影长度。
- 推论：当 a、b 长度固定，点积越大 → cosθ 越大 → 方向越接近。
- 特殊：θ=90°（正交）→ cosθ=0 → 点积=0；θ=0°（同向）→ 点积=|a||b|（最大）。
- 比较浮点数用 `np.isclose(a, b)`（比 `==` 稳，避免小数精度问题）。

💻 **动手（9分钟）**

```python
import numpy as np

def cosine_sim(a, b):                      # Day 13 已学
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])
c = np.array([-2.0, 1.0])                 # 与 a 正交：a·c = 1*-2 + 2*1 = 0

# 1) 代数点积 == 几何 |a||b|cosθ
dot_ab = a @ b
geom   = np.linalg.norm(a) * np.linalg.norm(b) * cosine_sim(a, b)
print("代数 a@b       =", dot_ab)          # 11.0
print("几何 |a||b|cosθ =", geom)          # 11.0 相等

# 2) 交换律
print("a@b == b@a ?", np.isclose(a @ b, b @ a))   # True

# 3) 比较三个向量方向
print("cos(a,b) =", cosine_sim(a, b))     # ≈0.98 同向
print("cos(a,c) =", cosine_sim(a, c))     # ≈0.0  正交
```

👀 **你该看到**：dot_ab 与 geom 都是 11.0；交换律为 True；cos(a,b)≈0.98（接近同向），cos(a,c)≈0（c 与 a 正交）。

⚠️ **常见坑**

- 点积本身受向量长度影响（|a||b| 大，点积就大）。比较"方向"要用余弦相似度（已除掉长度），别用裸点积。
- 比较浮点数别用 `==`，用 `np.isclose`。

✅ **打卡**：能验证点积交换律，并说清"为什么比较方向用余弦而不是点积"。

---

## 第 16 天（第3周·第2天）矩阵即变换：旋转与缩放

🎯 **目标**：理解一个 2×2 矩阵能把平面上的所有点"整体变换"（旋转/缩放/剪切）。这是理解"神经网络一层 = 一次线性变换"的关键直觉。

📖 **理论（8分钟）**

- 把点写成列向量 `p`，矩阵 `M` 作用得到新点 `M @ p`。
- 旋转矩阵（逆时针 θ）：`[[cosθ, −sinθ],[sinθ, cosθ]]`。θ=90° 时变成 `[[0,−1],[1,0]]`，把 (1,0) 转到 (0,1)。
- 缩放矩阵：`[[s_x,0],[0,s_y]]` 把 x 拉 s_x 倍、y 拉 s_y 倍。
- 矩阵能同时变换一组点：把多个点按列排成 (2, N) 矩阵，一次 `M @ P` 算完。

💻 **动手（9分钟）**

```python
import numpy as np
import matplotlib.pyplot as plt

M = np.array([[0.0, -1.0],
              [1.0,  0.0]])               # 逆时针旋转 90°

# 单个点 (1,0)：写成行向量后，用 (M @ p.T).T 转成列算完再转回
p = np.array([1.0, 0.0])
p_new = (M @ p.T).T
print("(1,0) 旋转后 =", p_new)            # [0. 1.] 即 (0,1)

# 单位正方形四角（每行是 x,y），一次变换全部点
corners = np.array([[0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
transformed = M @ corners                 # (2,4) 新坐标

plt.plot(corners[0], corners[1], 'o-', label="变换前")
plt.plot(transformed[0], transformed[1], 's-', label="变换后(旋转90°)")
plt.axis('equal'); plt.grid(True); plt.legend(); plt.show()
```

👀 **你该看到**：(1,0) 变成 (0,1)；正方形被逆时针旋转 90° 成竖着的正方形。

⚠️ **常见坑**

- NumPy 点常写成行向量 `[x, y]`。若要 `M @ p` 直接算，p 得是列向量（2,1）。教程用 `(M @ p.T).T` 把行向量先转置成列、算完再转回，避免形状错乱。也可统一写 `p @ M.T`（行向量版）。
- 画图务必 `plt.axis('equal')`，否则旋转看起来像被压扁。

✅ **打卡**：能用旋转矩阵作用在一组点上，并画出变换前后对比图。

---

## 第 17 天（第3周·第3天）导数：变化率

🎯 **目标**：用"数值导数"亲手验证"导数就是函数在某点的瞬时变化率"。这是后面梯度、反向传播的地基。

📖 **理论（8分钟）**

- 导数 f'(x) 直觉：x 变化一点点，y 变化多少，即切线斜率。
- 数值近似：`f'(x) ≈ (f(x+h) − f(x)) / h`，h 取很小的值（如 1e-5）。
- 解析导数：f(x)=x² 导数是 2x，x=3 处应为 6。

💻 **动手（9分钟）**

```python
import numpy as np

def f(x): return x**2

def num_deriv(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h          # 单边差分

x = 3.0
num = num_deriv(f, x)
analytic = 2 * x
print("数值导数 ≈", num)                   # 接近 6
print("解析导数 =", analytic)              # 6
print("误差 =", abs(num - analytic))

# 不同 h 对误差的影响
for h in [1e-1, 1e-3, 1e-5, 1e-7]:
    print(f"h={h:.0e} -> {num_deriv(f, x, h):.6f}")
```

👀 **你该看到**：数值导数≈6（误差极小）；h=1e-7 时误差反而变大（浮点精度），h≈1e-5 最稳。

⚠️ **常见坑**

- h 不能太大（近似不准）也不能太小（舍入误差爆炸），1e-5 是常用甜点。
- 数值导数只用于"验证"；真实训练用解析/自动求导（后面 PyTorch 会讲）。

✅ **打卡**：能写数值导数函数，验证 x² 在 x=3 处导数≈6。

---

## 第 18 天（第3周·第4天）梯度：最陡上升方向

🎯 **目标**：把"导数"推广到多变量函数，理解梯度向量指向"函数增长最快的方向"——梯度下降就是反着走。

📖 **理论（8分钟）**

- 多元函数 f(x,y)，对 x 求偏导（y 当常数）得 ∂f/∂x；对 y 得 ∂f/∂y。
- 梯度 `∇f = [∂f/∂x, ∂f/∂y]`，指向"上升最快方向"，长度=上升速率。
- 例：f(x,y)=x²+y²，解析梯度=[2x, 2y]，在 (1,2) 处为 [2,4]。
- 数值梯度：对每个变量单独做微小扰动求偏导。用**中心差分** `(f(x+h)−f(x−h))/(2h)` 比单边更精确。

💻 **动手（9分钟）**

```python
import numpy as np

def f(xy):
    x, y = xy
    return x**2 + y**2

def num_grad(f, xy, h=1e-5):
    grad = np.zeros_like(xy, dtype=float)
    for i in range(len(xy)):
        xy_plus = xy.copy(); xy_plus[i] += h
        xy_minus = xy.copy(); xy_minus[i] -= h
        grad[i] = (f(xy_plus) - f(xy_minus)) / (2 * h)   # 中心差分
    return grad

point = np.array([1.0, 2.0])
num_g = num_grad(f, point)
analytic = np.array([2 * point[0], 2 * point[1]])
print("数值梯度 ≈", num_g)                  # 接近 [2, 4]
print("解析梯度 =", analytic)              # [2, 4]
print("两向量几乎相等:", np.allclose(num_g, analytic))
```

👀 **你该看到**：数值梯度≈[2,4]，与解析一致；`np.allclose` 返回 True。

⚠️ **常见坑**

- 中心差分误差是 O(h²)，比单边 O(h) 更准——但计算量翻倍，验证时用很合适。
- 偏导是"每次只动一个变量"，用 for 循环逐个算；以后会学向量化写法，原理一样。

✅ **打卡**：能写数值梯度函数，验证 x²+y² 在 (1,2) 处梯度为 [2,4]。

---

## 第 19 天（第3周·第5天）链式法则：复合函数求导

🎯 **目标**：理解复合函数求导的链式法则——这是反向传播"误差如何一层层传回去"的数学本质。

📖 **理论（8分钟）**

- 复合函数 h(x)=f(g(x))，导数：`h'(x) = f'(g(x)) · g'(x)`。
- 直觉：先算内层变化率 g'(x)，再乘外层在 g(x) 处的变化率 f'(g(x))。
- 例：g(x)=x²，f(u)=u+1，则 h(x)=x²+1，h'(x)=2x。验证：f'(u)=1，g'(x)=2x，相乘=1·2x=2x ✓

💻 **动手（9分钟）**

```python
import numpy as np

def g(x): return x**2
def f(u): return u + 1
def h(x): return f(g(x))                  # = x^2 + 1

def num_deriv(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)   # 中心差分

x = 3.0
h_prime = num_deriv(h, x)                  # 数值 h'(3)
chain = 1 * (2 * x)                       # f'(g) * g' = 1 * 2x = 2x
print("数值 h'(3) ≈", h_prime)             # 接近 6
print("链式法则解析 2x =", chain)          # 6
```

👀 **你该看到**：数值 h'(3)≈6，链式法则算出 2x=6，两者一致。

⚠️ **常见坑**

- 链式法则里 f' 要在"内层的值 g(x)"处求值，不是直接在 x 处。本例碰巧 f'=1 常数，没体现，记住该要点。
- 后面反向传播就是把这条链从输出往回乘，一层层得到每个权重的梯度。

✅ **打卡**：能写 h=f∘g，用数值导数验证 h'(3)≈6，并口述链式法则公式。

---

## 第 20 天（第3周·第6天）梯度下降：沿负梯度下山 ⭐

🎯 **目标**：动手实现梯度下降找到函数最小值。这是"训练模型"的核心算法——模型训练本质就是不断调参让损失最小。

📖 **理论（8分钟）**

- 目标：找 f(x) 的最小值。
- 思路：梯度 ∇f 指向上升最快方向，那"负梯度"就是下降最快方向。
- 更新规则：`x ← x − lr · ∇f(x)`。lr（学习率）控制每步走多大。
- 例：f(x)=x²−4x，解析梯度=2x−4，最小值在 x=2（f=−4）。

💻 **动手（9分钟）**

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x): return x**2 - 4*x
def df(x): return 2*x - 4                 # 解析梯度

lr = 0.1
x = 0.0                                   # 随便选起点
history = [x]
for _ in range(30):
    x = x - lr * df(x)                    # 沿负梯度走一步
    history.append(x)

print("最终 x =", x)                       # 接近 2
print("最小 f =", f(x))                    # 接近 -4

xs = np.linspace(-1, 3, 200)
plt.plot(xs, f(xs), label="f(x)=x²-4x")
plt.plot(history, [f(v) for v in history], 'ro-', label="下降轨迹")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.legend(); plt.show()
```

👀 **你该看到**：x 从 0 逐步逼近 2，f(x) 逼近 −4；红色轨迹沿曲线"滚"到谷底。

⚠️ **常见坑**

- lr 太大（如 1.5）会"步子太大"在谷底两边来回跳甚至发散；lr 太小则收敛慢。把 lr 改成 1.5 试一次。
- 这是一维梯度下降；神经网络是在百万维空间做同样的事。

✅ **打卡**：能实现梯度下降求 x²−4x 最小值并画轨迹，理解 lr 作用。

---

## 第 21 天（第3周·复盘日）自测：徒手写梯度下降

🎯 **目标**：检验本周数学直觉是否扎实，能不依赖教程写出核心代码。

📝 **自测（合上资料）**

1. 用 NumPy 算两向量点积，并验证与点积几何公式一致；验证交换律。
2. 用旋转矩阵 `[[0,−1],[1,0]]` 把点 (1,0) 变到哪？画出来。
3. 写数值导数函数，验证 f(x)=x³ 在 x=2 处导数≈12。
4. 写数值梯度，验证 f(x,y)=x²+2y² 在 (1,1) 处梯度≈[2,4]。
5. 不用看教程，从零用梯度下降求 f(x)=x²−4x 最小值并画轨迹。

✅ **打卡**：5 题独立做对，保存成 `week03_review.ipynb` 或 `.py`。

---

> 📌 **第 3 周（Day 15–21）到此。** 你已掌握向量点积、线性变换、导数、梯度、链式法则、梯度下降——这些是神经网络的全部数学地基。  
> 下一步是**第 4 周（Day 22–28）：损失函数、激活函数、概率与交叉熵**。下面紧接着写。

---

## 第 22 天（第4周·第1天）损失函数：预测有多准

🎯 **目标**：理解"损失函数"是训练模型的指挥棒——它把"模型预测得有多差"变成一个可以最小化的数字。先手写最基础的均方误差 MSE。

📖 **理论（8分钟）**

- 训练模型 = 不断调参数，让"预测值"尽量贴近"真实值"。
- **损失函数**衡量单次预测的误差：损失越小，预测越准。
- **均方误差 MSE**（回归任务常用）：`MSE = mean((y_pred − y_true)²)`，即"差的平方"取平均。
  - 平方的好处：让大误差受惩罚更重；且处处可导（后面反向传播要用）。
- 完美预测时 MSE = 0。

💻 **动手（9分钟）**

```python
import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.9, 3.2])

print("预测偏差 =", mse(y_pred, y_true))        # ((0.1)²+(0.1)²+(0.2)²)/3 = 0.02
print("完美预测 =", mse(y_true, y_true))        # 0.0
print("差得更离谱 =", mse(np.array([5.0,5.0,5.0]), y_true))  # 更大
```

👀 **你该看到**：预测偏差≈0.02，完美预测=0.0，乱猜的更大——损失确实"越小越准"。

⚠️ **常见坑**

- `y_pred` 和 `y_true` 形状必须一致，否则减法和求平均都会出错。
- MSE 对离群点敏感（平方放大）：一个极端错误会把整体损失拉得很高。

✅ **打卡**：能手写 MSE 函数，说清"为什么平方而不是绝对值"。

---

## 第 23 天（第4周·第2天）激活函数：给网络注入非线性

🎯 **目标**：搞懂"为什么神经网络里必须有激活函数"，并亲手画出 sigmoid / tanh / ReLU 三条曲线。

📖 **理论（8分钟）**

- **关键直觉**：多层"线性层"叠在一起，整体仍是线性变换（y = Wx+b 再套 Wx+b 还是直线）。没有非线性，网络永远学不出曲线。
- **激活函数**插在每层之后，把线性输出"掰弯"，网络才具备拟合任意形状的能力。
- 三个最常用的：
  - **sigmoid**：`σ(x) = 1 / (1 + e⁻ˣ)`，输出压在 0~1，像"开关/概率"。
  - **tanh**：输出压在 −1~1，比 sigmoid 更"零中心"。
  - **ReLU**：`max(0, x)`，正区间不变、负区间归零，简单又缓解梯度消失。

📌 **二级视角**：`e` 是自然常数≈2.718，`np.exp(x)` 算"e 的 x 次方"。指数运算二级不教，但直觉很简单——`eˣ` 就是"以 e 为底的快速增长倍数"。今天你只要会用 `np.exp` 这一个点即可。

💻 **动手（9分钟）**

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def tanh(x):    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def relu(x):    return np.maximum(0, x)

x = np.linspace(-6, 6, 100)
for i, (name, func) in enumerate([("sigmoid", sigmoid), ("tanh", tanh), ("ReLU", relu)], 1):
    plt.subplot(1, 3, i)
    plt.plot(x, func(x))
    plt.title(name); plt.grid(True)
plt.tight_layout(); plt.show()
```

👀 **你该看到**：sigmoid 是 S 形（0~~1）、tanh 也是 S 形但过原点（−1~~1）、ReLU 是折线（负区贴地、正区 45°）。

⚠️ **常见坑**

- sigmoid **务必写成 `1/(1+e⁻ˣ)`**，别写成 `1/(1+eˣ)`：当 x 是正大数时 `eˣ` 会数值溢出爆掉，而 `e⁻ˣ` 此时趋于 0，安全。
- ReLU 用 `np.maximum(0, x)`（逐元素取大），不是乘法。

✅ **打卡**：能写出三个激活函数并画出对比图，说清"没有激活函数为什么网络没用"。

---

## 第 24 天（第4周·第3天）sigmoid 求导：σ′=σ(1−σ)

🎯 **目标**：记住并验证 sigmoid 导数这个极省事的性质——只要算出了 σ，导数就直接是 σ(1−σ)，不用再算指数。反向传播里天天用它。

📖 **理论（8分钟）**

- 推导（了解即可）：σ=1/(1+e⁻ˣ)，求导得 `σ′ = e⁻ˣ / (1+e⁻ˣ)² = σ(1−σ)`。
- 含义：σ 接近 0 或 1 时导数接近 0（"饱和"，梯度小）；σ=0.5 时导数最大=0.25。
- 用数值导数验证这个漂亮公式。

💻 **动手（9分钟）**

```python
import numpy as np

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_prime(x): return sigmoid(x) * (1 - sigmoid(x))

def num_deriv(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

for x in [-2, -1, 0, 1, 2]:
    analytic = sigmoid_prime(x)
    numeric  = num_deriv(sigmoid, x)
    print(f"x={x:>3}  解析={analytic:.6f}  数值={numeric:.6f}  误差={abs(analytic-numeric):.2e}")
```

👀 **你该看到**：五组解析值与数值值几乎完全一致；x=0 时 σ′=0.25 最大。

⚠️ **常见坑**

- 利用 `σ′=σ(1−σ)` 可以在反向传播里"复用"前向已算好的 σ，避免重复 `np.exp`，又快又稳。

✅ **打卡**：能验证 σ′=σ(1−σ)，并说出 x=0 处导数值。

---

## 第 25 天（第4周·第4天）概率基础：期望、方差、正态分布

🎯 **目标**：补一点概率直觉——期望（中心）、方差（胖瘦）、正态分布（机器学习最常见的数据分布）。全部用代码感受，不背公式。

📖 **理论（8分钟）**

- **期望**≈一堆数的"平均位置"；**方差**≈这些数"平均离中心多远"（方差大=散布广）。
- **正态分布 N(μ, σ²)**：钟形曲线。μ 是中心，σ 是胖瘦。标准正态 N(0,1) 的 μ=0、σ=1。
- 大量随机样本画直方图，形状会贴合理论钟形曲线。

📌 **二级视角**：概率听起来吓人，其实就三句话——"期望=平均"、"方差=平均离多远"、"正态=中间多两头少的钟形"。今天用代码把这三句话跑出来就够了。

💻 **动手（9分钟）**

```python
import numpy as np
import matplotlib.pyplot as plt

samples = np.random.randn(10000)        # 生成 10000 个标准正态 N(0,1) 样本
print("样本均值 ≈", np.mean(samples))    # 接近 0
print("样本方差 ≈", np.var(samples))     # 接近 1

plt.hist(samples, bins=50, density=True, alpha=0.6)
xs = np.linspace(-4, 4, 200)
pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-xs**2 / 2)   # 理论曲线
plt.plot(xs, pdf, 'r-', lw=2)
plt.title("标准正态直方图 + 理论曲线"); plt.show()
```

👀 **你该看到**：均值≈0、方差≈1；直方图是钟形，并且贴合红色理论曲线。

⚠️ **常见坑**

- 直方图纵轴要用"概率密度"模式（`density=True`），这样曲线下总面积=1，才能和理论密度曲线叠加对比。默认是"计数"，纵轴量级对不上。

✅ **打卡**：能生成正态分布样本、打印均值/方差、画出贴合理论曲线的直方图。

---

## 第 26 天（第4周·第5天）交叉熵：预测与真实的差距

🎯 **目标**：掌握分类任务的主损失——二元交叉熵（BCE），理解"预测越准、损失越小"，以及为什么必须防止 `log(0)`。

📖 **理论（8分钟）**

- 分类问题预测的是"概率"（如"是猫的概率 0.9"），用**交叉熵**衡量预测概率分布与真实分布的差距。
- **二元交叉熵**：真实标签 y∈{0,1}，预测概率 p∈(0,1)，  
  `BCE = −[y·log(p) + (1−y)·log(1−p)]`
  - y=1 时：p 越接近 1 损失越小；p 接近 0 损失爆炸（log0→−∞）。
  - y=0 时对称。
- **防溢出**：p 不能等于 0 或 1（否则 log 发散成 nan），实际训练会把 p 裁剪到一个很小的正区间（如 [1e-12, 1−1e-12]）。

💻 **动手（9分钟）**

```python
import numpy as np

def binary_cross_entropy(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)     # 裁剪，避免 log(0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

print("p=0.9, y=1 ->", binary_cross_entropy(0.9, 1))   # 小：预测很准
print("p=0.5, y=1 ->", binary_cross_entropy(0.5, 1))   # 中：瞎猜
print("p=0.1, y=1 ->", binary_cross_entropy(0.1, 1))   # 大：预测很错
```

👀 **你该看到**：0.9→≈0.105、0.5→≈0.693、0.1→≈2.303——越准损失越小，符合直觉。

⚠️ **常见坑**

- 不裁剪直接 `np.log(0)` 会得到 `-inf`，进而让 loss 变成 `nan`——这就是训练时"loss 突然变 nan"的常见根源之一。

✅ **打卡**：能手写 BCE，解释"预测越准损失越小"和"为什么要裁剪 p"。

---

## 第 27 天（第4周·第6天）综合复习日

🎯 **目标**：检验本周（损失/激活/概率/交叉熵）是否真懂，合上资料自测 3 题。

📝 **自测（合上资料）**

1. 手推 f(x)=x³ 在 x=2 处的导数（解析 3x²=12），并用数值导数函数验证。
2. 实现 `tanh` 的数值导数（用中心差分），验证 `tanh'(0)=1`。
3. 用一句话解释：学习率太大 / 太小，训练分别会发生什么？

✅ **打卡**：3 题独立做对，保存成 `week04_review.py`。

---

## 第 28 天（第4周·复盘日）阶段总结：训练模型有哪些零件

🎯 **目标**：把前 4 周串成一句话——"训练一个模型到底在干什么"。

📝 **复盘写作（约 200 字）**  
写一段总结，回答：**训练一个模型需要哪些零件？** 提示按这 5 个角色展开：

- **数据**：输入 x 和标签 y（模型要学的"题和答案"）
- **模型**：一串可学习的参数（如 W、b），把 x 映射成预测 ŷ
- **损失**：衡量 ŷ 与 y 差多远（MSE / 交叉熵）
- **梯度**：损失对每个参数的变化率，指出"该往哪调"
- **更新规则**：沿负梯度小步调整参数（梯度下降）

✅ **打卡**：写满 200 字，能对着这 5 个零件讲一遍"训练流程"。

---

---

### 第 5 周：机器学习入门（线性回归）

> 目标：从零手写完整的「训练五步曲」——生成数据 → 定义模型 → 算损失 → 梯度下降 → 可视化。这就是后面手写神经网络的直接前置。

**第 29 天 什么是机器学习：监督/无监督**

🎯 **目标**：理解"训练=在数据中找一组参数让模型预测尽量准"，并会加载一个真实数据集看一眼。

📖 **理论（8分钟）**

- 机器学习分两类：
  - **监督学习**：数据带"答案"（标签 y），如"给定房屋面积 x，预测价格 y"。本计划全程是监督学习。
  - **无监督学习**：只有 x 没有 y，如把相似客户自动分组。
- 训练的本质：给定模型结构（如 ŷ = w·x + b），找一组参数（w, b）让"预测 ŷ"尽量接近"真实 y"。
- 我们用 `sklearn`（scikit-learn，一个机器学习库）自带的**鸢尾花数据集**做"认识数据"练习：150 朵花，4 个特征（花萼长/宽、花瓣长/宽），3 个品种。

💻 **动手（9分钟）**

```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data          # 特征，形状 (150, 4)
y = iris.target        # 标签，0/1/2 三种花
print("特征矩阵形状:", X.shape)
print("前 5 行特征:\n", X[:5])
print("标签前 5 个:", y[:5])
print("标签含义:", iris.target_names)
```

👀 **你该看到的输出**

- `X.shape` = `(150, 4)`
- 打印出 5 行 4 列数字，标签 `[0 0 0 0 0]`，`target_names` 是 3 个花名。

⚠️ **常见坑（二级易踩）**

- `load_iris()` 返回的不是数组，而是一个"对象"，要 `.data` / `.target` 才能拿到数组（和 numpy 数组不一样，别直接 `iris[0]`）。
- 这个数据集暂时只用来"看数据长啥样"，第 6 周才真正拿它分类。

✅ **打卡**：能说出"监督学习"和"无监督学习"的区别，并成功打印出鸢尾花数据的形状。

---

**第 30 天 线性回归模型：ŷ = w·x + b**

🎯 **目标**：用代码生成一组"带噪声的线性数据"，理解模型就是一个含未知参数 w、b 的公式。

📖 **理论（8分钟）**

- 线性回归模型：`ŷ = w·x + b`。w 是斜率、b 是截距。我们要让模型"学"的就是 w 和 b。
- 生成"教材数据"：让真实规律是 `y = 2x + 1`，再加一点随机噪声（模拟真实测量误差），这样训练后学到的 w 应接近 2、b 接近 1。

💻 **动手（9分钟）**

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
n = 100
x = np.random.uniform(0, 10, n)          # 0~10 均匀取 100 个点
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + np.random.randn(n) * 1.5   # 加噪声

plt.scatter(x, y, s=15, label="样本")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
```

👀 **你该看到的输出**：一张散点图，点大致沿斜率为 2、截距为 1 的直线散布，但有上下抖动（噪声）。

⚠️ **常见坑**

- 加噪声用 `randn`（标准正态，均值 0），乘 1.5 控制噪声幅度；别和 `rand`（均匀）搞混。
- `uniform(0, 10, n)` 第三参数是"个数"，不是"形状元组"——和 `np.zeros((2,3))` 不同。

✅ **打卡**：能解释"噪声"在训练数据里的作用（让模型学真实规律而非死记硬背）。

---

**第 31 天 损失对参数的梯度：手写梯度下降训练 ⭐**

🎯 **目标**：全计划最重要的一天之一——亲手写出"损失对 w、b 的梯度"，并用梯度下降迭代更新。

📖 **理论（8分钟）**

- 损失用 MSE：`L = mean((ŷ − y)²) = mean((w·x + b − y)²)`。
- 对 w、b 求偏导（梯度）：
  - `∂L/∂w = (2/n)·Σ(ŷ−y)·x`
  - `∂L/∂b = (2/n)·Σ(ŷ−y)`
- 更新：`w ← w − lr·∂L/∂w`，`b ← b − lr·∂L/∂b`。lr 是学习率（步长）。

💻 **动手（9分钟）**

```python
w, b = 0.0, 0.0        # 初始随便给
lr = 0.01
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
print("学到的 w,b:", round(w,3), round(b,3), " 真实:", true_w, true_b)
```

👀 **你该看到的输出**：loss 从几十一路下降，最后 w≈2.0、b≈1.0（接近真实值）。

⚠️ **常见坑**

- `err * x` 是逐元素乘（不是矩阵乘 `@`），因为 x、err 都是一维向量。
- 梯度公式前的 `2/n` 只是常数缩放，不影响下降方向；有时省写成 `(1/n)·Σ(ŷ−y)·x`（lr 自动吸收），结果一致，别纠结系数。

✅ **打卡**：能默写出"w 的梯度里多了个 x 相乘"——这正是"哪个参数影响输出，梯度就乘上它对输出的贡献"。

---

**第 32 天 学习率：步子大小的 trade-off**

🎯 **目标**：直观感受学习率 lr 太大/太小会怎样，学会选一个合适的 lr。

📖 **理论（8分钟）**

- lr 太小：每步走一点点，要很多轮才收敛（慢）。
- lr 太大：一步跨过头，loss 在最小值附近来回震荡甚至发散（爆炸）。
- 经验：线性回归常试 `0.001 ~ 0.1`；太大（如 1.0）容易直接 NaN。

💻 **动手（9分钟）**

```python
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

👀 **你该看到的输出**：三条曲线——lr=0.001 慢慢降、lr=0.01 平滑降到最低、lr=1.0 剧烈震荡甚至飞掉。

⚠️ **常见坑**

- lr=1.0 可能让 w、b 数值溢出成 `inf`/`nan`，图都画不出。遇到就把 1.0 改成 0.5 再试。
- `plt.yscale("log")` 是对数纵轴，能同时看清"慢降"和"爆炸"两种曲线。

✅ **打卡**：能口述"lr 太小慢、太大炸"，并记住常用区间。

---

**第 33 天 收敛判断与早停**

🎯 **目标**：给训练加"自动停止"条件，避免无意义的空转。

📖 **理论（8分钟）**

- 训练时 loss 越降越慢，最后几乎不动——再练也是浪费。
- **早停**：当相邻两次 loss 差小于阈值（如 1e-6）就 `break` 停止，并记录实际用了多少轮。

💻 **动手（9分钟）**

```python
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

👀 **你该看到的输出**：不会跑满 10000 轮，几百轮左右打印"提前收敛"，最终 w≈2、b≈1。

⚠️ **常见坑**

- 阈值 1e-6 太小可能永远触发不了（噪声大时 loss 下不去），可放宽到 1e-4。
- 第一次循环 `prev_loss` 用 `float("inf")` 初值，保证第一次 `abs(inf - loss)` 一定大于阈值，不会一上来就停。

✅ **打卡**：能解释"为什么需要早停"（省时间 + 防过拟合雏形）。

---

**第 34 天 向量化：用矩阵一次算全部样本**

🎯 **目标**：理解"向量化"——把 for 循环换成矩阵运算，速度能快几十倍。这正是 NumPy/PyTorch 的精髓。

📖 **理论（8分钟）**

- 之前每轮 `yhat = w*x + b`：numpy 已自动对整条向量广播，本身就是向量化的。
- 真正的"非向量化"是：对 i 个样本写 for 循环逐个算。对比两者速度。
- 更进一步的向量化：把偏置 b 并入矩阵（给 x 加一列全 1），参数变成 `W = [w, b]`，一次 `X @ W` 同时算 `w·x` 和 `+b`。

💻 **动手（9分钟）**

```python
# 非向量化（逐样本）
def train_loop():
    w, b = 0.0, 0.0
    for _ in range(100):
        for i in range(n):
            yhat_i = w*x[i] + b
            err_i = yhat_i - y[i]
            w -= lr*(2/n)*err_i*x[i]
            b -= lr*(2/n)*err_i
# 向量化（整批）
def train_vec():
    w, b = 0.0, 0.0
    for _ in range(100):
        yhat = w*x + b
        err = yhat - y
        w -= lr*(2/n)*np.sum(err*x)
        b -= lr*(2/n)*np.sum(err)

# %timeit 只在 Jupyter 有效；脚本里改用 time 模块：
import time
t = time.time(); train_loop(); print("逐样本耗时:", time.time()-t)
t = time.time(); train_vec();  print("向量化耗时:", time.time()-t)
```

👀 **你该看到的输出**：向量化版本快很多（可能几十到上百倍）。

⚠️ **常见坑**

- `%timeit` 是 Jupyter 魔法命令，普通 `.py` 脚本里会报错；脚本里改用 `time` 模块手动计时。
- 两种写法更新公式略有不同（逐样本 vs 整批平均），结果会有细微差异，正常。

✅ **打卡**：能说出"向量化 = 用矩阵一次算完，避免 Python 层 for 循环"，以及为什么它快。

---

**第 35 天 复盘日：画出拟合直线**

🎯 **目标**：把训练结果画出来，直观确认"它真的学到了直线"。

📖 **理论（8分钟）**

- 复盘训练五步曲：数据 → 模型(ŷ=wx+b) → 损失(MSE) → 梯度 → 更新。
- 今天不学新东西，把第 31 天的训练结果画成"散点 + 拟合直线"。

💻 **动手（9分钟）**

```python
plt.scatter(x, y, s=15, alpha=0.6, label="样本")
xs = np.array([x.min(), x.max()])
plt.plot(xs, w*xs + b, "r-", label=f"拟合: y={w:.2f}x+{b:.2f}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
print("学到的 w,b:", round(w,3), round(b,3), "（应接近 2 和 1）")
```

👀 **你该看到的输出**：红色直线稳稳穿过蓝色散点中间，w≈2、b≈1。

⚠️ **常见坑**

- 直线歪得离谱，多半是第 31 天 lr 太大训练炸了——回去把 lr 调小重训。
- `xs` 用 `x.min()/x.max()` 控制直线只画在数据范围内。

✅ **打卡**：合上教程，能口述"训练五步曲"，并看到自己的直线盖在散点上。

---

---

### 第 6 周：机器学习入门（逻辑回归与训练五步曲闭环）

> 目标：把"训练五步曲"用在第 37 天起第一次遇到的新任务——**分类**。这一周你会从零写出逻辑回归：分训练/测试集、写 sigmoid 前向、用交叉熵做梯度下降、画决策边界、算混淆矩阵、做标准化。五步曲彻底闭环。

**第 36 天 过拟合与训练/测试集**

🎯 **目标**：理解"训练集 / 测试集"划分，并亲眼看到过拟合（模型死记训练数据、在新数据上很差）。

📖 **理论（8分钟）**

- 把数据分成两份：训练集（学参数用）、测试集（考模型用，训练时**绝对不能看**）。
- 过拟合：模型在训练集上误差很小，但测试集上误差很大——它"背下了"训练样本而非学到规律。
- 直观实验：用极少样本 + 高次多项式去拟合，曲线会剧烈扭曲去穿过每个点（过拟合的典型样子）。
- 机器学习库里有现成工具把数据按比例切分，也有现成的"多项式特征扩展"工具把一次特征变成高次。

💻 **动手（9分钟）**

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

👀 **你该看到的输出**：16 个训练点被一条剧烈弯折的 9 次曲线穿过（训练点之间疯狂振荡）——这就是过拟合。

⚠️ **常见坑（二级易踩）**

- `train_test_split` 返回四个值是**平铺**的 `(x_train, x_test, y_train, y_test)`，别记错顺序。
- `PolynomialFeatures` 默认自带一列"0 次项"（全 1），特征矩阵第一列是常数，不用自己加。
- `np.linalg.lstsq` 是"解线性方程组"的最小二乘解，这里只是偷懒画过拟合曲线；真正训练我们一直手写梯度下降。

✅ **打卡**：能说出"为什么要分训练集和测试集"（测试集模拟真实未知数据，检验模型真本事）。

---

**第 37 天 分类问题与逻辑回归前向**

🎯 **目标**：从"预测一个数"（回归）转向"预测类别"（分类），写出逻辑回归前向 ŷ = sigmoid(w·x + b)。

📖 **理论（8分钟）**

- 回归输出连续值；分类输出"属于某类的概率"（0~1）。
- 逻辑回归：先算线性得分 z = w·x + b，再经 sigmoid 把任意实数压到 (0,1) 当概率：ŷ = σ(z)。
- 预测类别：ŷ > 0.5 判为正类（1），否则负类（0）。
- 生成两类二维点（如"红类"中心在右上、"蓝类"中心在左下），目标是学一条分界线把两类分开。

💻 **动手（9分钟）**

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

👀 **你该看到的输出**：随机参数下准确率约 50%（瞎猜）；概率有大有小，经阈值 0.5 变成 0/1。

⚠️ **常见坑**

- `sigmoid` 用 `np.exp(-z)`，z 是向量也能一次性算（自动逐元素）。
- `(probs > 0.5)` 得布尔数组，`.astype(int)` 才变成 0/1 整数，方便和 y 比。
- 这是第一次见二维特征：X 是 (100,2)，`X @ w` 是 (100,2)·(2,) → (100,)。

✅ **打卡**：能默写出逻辑回归前向三步：线性得分 → sigmoid → 阈值分类。

---

**第 38 天 用交叉熵训练逻辑回归 ⭐**

🎯 **目标**：用梯度下降 + 交叉熵损失，把随机的 w、b 训练成能分开两类的参数，并画出决策边界。

📖 **理论（8分钟）**

- 损失用二元交叉熵：`L = −mean(y·log(ŷ) + (1−y)·log(1−ŷ))`。
- 对 w、b 的梯度（记住结论即可）：
  - `∂L/∂w = (1/n)·Xᵀ·(ŷ − y)`
  - `∂L/∂b = (1/n)·Σ(ŷ − y)`
- 和线性回归梯度很像，区别只是误差项来自 sigmoid 后的 ŷ。

💻 **动手（9分钟）**

```python
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

👀 **你该看到的输出**：loss 从约 0.69 降到很低，散点被一条平滑决策边界（contour 0.5 那条）清晰分开，准确率接近 100%。

⚠️ **常见坑**

- `log(0)` 得 `-inf`，所以加 `1e-9` 裁剪（第 26 天讲过）。
- 梯度里 `X.T @ err`：X 是 (100,2)，err 是 (100,)，乘出来 (2,)，对应 w 的两个分量。
- lr=0.1 配 2000 轮才够；若 loss 不降，检查 err 符号（应是 ŷ−y 不是 y−ŷ）。

✅ **打卡**：能口述"逻辑回归和线性回归梯度只差在误差项来源"。

---

**第 39 天 评估：准确率与混淆矩阵**

🎯 **目标**：训练完不只看"准不准确"，还能拆出"哪类容易错"——用混淆矩阵。

📖 **理论（8分钟）**

- 准确率 = 预测对的数量 / 总数。
- 混淆矩阵（二分类）四个格子：
  - **TP**（真阳性）：实际 1 预测 1
  - **FP**（假阳性）：实际 0 预测 1（误报）
  - **FN**（假阴性）：实际 1 预测 0（漏报）
  - **TN**（真阴性）：实际 0 预测 0
- 用预测类别和真实标签逐个比对，统计四个数。

💻 **动手（9分钟）**

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

👀 **你该看到的输出**：4 个格子数字，准确率接近 1.0（第 38 天已训好）。

⚠️ **常见坑**

- 混淆矩阵行列顺序容易搞混。记死：第一个字母是"预测对不对"（T/F），第二个字母是"实际类别"（P=正类1 / N=负类0）。所以 FP = 预测正、实际负。
- 类别不平衡时准确率会骗人（全猜多数类也能高），所以才需要混淆矩阵。

✅ **打卡**：看到混淆矩阵能立刻说出"FP 是哪种错误"。

---

**第 40 天 特征标准化**

🎯 **目标**：理解为什么要做 z-score 标准化，并对比"标准化前/后"训练收敛速度。

📖 **理论（8分钟）**

- 不同特征量纲不同（如"身高 cm"和"体重 kg"），会让损失地形又扁又斜，梯度下降走得很慢。
- z-score 标准化：`x' = (x − mean) / std`，让每列特征均值 0、标准差 1。
- 实验：造两个量纲差很大的特征，分别训练"标准化前/后"，看同样轮数下 loss 降到多低。

💻 **动手（9分钟）**

```python
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

👀 **你该看到的输出**：标准化后的 loss 通常明显更低（或同样轮数下降得更稳）。

⚠️ **常见坑**

- `mean(0)` / `std(0)` 的 `0` 表示"按列（每个特征）算"，得到和 X 列数相同的一维数组，才能逐列广播。
- 测试集标准化要用**训练集的** mean/std（不能自己另算）；这里演示阶段先忽略，第 41 天实战会注意。

✅ **打卡**：能说出"标准化让不同量纲特征站在同一起跑线，损失地形更圆、下降更快"。

---

**第 41 天 综合实战：串成完整 pipeline**

🎯 **目标**：把前面所有零件装进一个函数——生成数据 → 划分 → 标准化 → 训练 → 评估 → 画图。

📖 **理论（8分钟）**

- 今天不学新东西，把 Day 36–40 串成一条流水线，体会"真实项目"的样子。
- 固定一个函数，输入数据、输出训练好的参数和评估指标。

💻 **动手（9分钟）**

```python
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

👀 **你该看到的输出**：一张决策边界图 + 打印的测试准确率（通常 0.9+）。

⚠️ **常见坑**

- 标准化必须用**训练集**的 mu/sd 去转换测试集（不能各算各的），否则数据泄露。
- 打乱用 `np.random.permutation` 生成索引再切，比 `train_test_split` 更透明，适合理解流程。

✅ **打卡**：合上教程，能口述完整 pipeline 的 6 个步骤。

---

**第 42 天 阶段复盘（写作 + 默写）**

🎯 **目标**：合上资料，默写"训练五步曲"并总结阶段收获。

📝 **复盘写作（200 字）**  
回答三个问题：

1. 训练一个模型需要的五个零件是什么？各起什么作用？
2. 线性回归和逻辑回归，在"模型、损失、梯度"上有什么相同与不同？
3. 过拟合是什么？标准化为什么能帮训练更快？

✅ **打卡**：写满 200 字，且能不看任何资料默写出五步曲。

---

---

## 第 7 周：神经网络核心·前向传播（Day 43–49）

> ⚠️ 本周是计划分水岭：你从"训练别人写好的模型"正式进入"自己搭网络"。基础零件（NumPy / 梯度 / 梯度下降 / 损失 / MSE / 交叉熵 / sigmoid / tanh）前面都备齐了，本周只是把它们按新结构组装。卡住就回 Day 23–24（激活）、Day 20（梯度下降）、Day 9（类）。  
> 📌 二级友好提示：本周最难的只有 **Day 48 用类封装**（回顾 `Python二级衔接补充.md` 第 2 节）。其余都是把前面学过的 `np.dot/@`、`tanh/sigmoid`、广播拼起来。

### Day 43 感知机：加权求和 + 阈值激活

🎯 目标：理解神经网络最古老的"祖先"——感知机，它就是一层"加权求和 + 阈值"。

📖 理论（8分钟）

- 感知机的计算：先算 `z = w·x + b`（w 和 x 都是向量，做点积；b 是偏置），再按阈值决定输出：`z ≥ 0 输出 1，否则 0`。
- 这其实就是一个"线性模型 + 阶跃激活"，是 Day 37 逻辑回归里 sigmoid 的"硬化版"（sigmoid 是平滑的 0~1，阶跃是非 0 即 1）。
- 经典例子 **AND 门**：取 w=[0.5, 0.5], b=−0.7。
  - 输入 (1,1)：z = 0.5+0.5−0.7 = 0.3 ≥ 0 → 输出 1 ✓
  - 输入 (1,0) / (0,1)：z = 0.5−0.7 = −0.2 < 0 → 输出 0 ✓
  - 输入 (0,0)：z = −0.7 < 0 → 输出 0 ✓
  - 完全符合 AND 真值表，说明一个感知机就能学 AND。

💻 动手（9分钟）

```python
import numpy as np

def perceptron(x, w, b):
    z = np.dot(x, w) + b          # 加权求和
    return 1 if z >= 0 else 0     # 阈值激活

w = np.array([0.5, 0.5])
b = -0.7
for x in [(0,0),(0,1),(1,0),(1,1)]:
    print(x, "->", perceptron(np.array(x), w, b))
# 应输出 0 0 0 1
```

👀 你该看到的输出：

```
(0, 0) -> 0
(0, 1) -> 0
(1, 0) -> 0
(1, 1) -> 1
```

⚠️ 常见坑

- 阈值是 `>= 0` 还是 `> 0`？AND 门里 z 从不等于 0，两种都行；以后遇到 z 恰好为 0，约定 `>= 0 → 1` 更常见。
- `np.dot(x, w)` 要求 x、w 同长度；x 是 (2,)、w 是 (2,) 时得到标量——正是我们要的。

✅ 打卡：把同一套逻辑改成 OR 门（w=[0.5,0.5], b=−0.2，验证 0,1,1,1）、再改成 NOT 门（单输入 w=[-0.5], b=0.2，验证 0→1、1→0）。

### Day 44 单层感知机的局限：XOR 不可分

🎯 目标：亲手见证"单层感知机永远学不会 XOR"，理解为什么必须加隐藏层。

📖 理论（8分钟）

- XOR 真值表：(0,0)→0、(1,1)→0、(0,1)→1、(1,0)→1。
- 在二维平面上，(0,0) 和 (1,1) 是一类、(0,1) 和 (1,0) 是一类——**没有任何一条直线能把它们分开**（画一下就明白），这种问题叫"线性不可分"。
- 单层感知机的决策边界永远是一条直线，所以 XOR 学不会——这不是调参问题，是结构性的。

💻 动手（9分钟）——复用 Day 38 的训练流程，但换成 XOR 数据

```python
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)

def sigmoid(z): return 1/(1+np.exp(-z))

w = np.random.randn(2,1)*0.1
b = np.zeros((1,1))
lr = 0.1
for epoch in range(5000):
    z = X @ w + b
    p = sigmoid(z)
    loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    dw = (1/len(X)) * X.T @ (p - y)
    db = (1/len(X)) * np.sum(p - y)
    w -= lr*dw; b -= lr*db

print("最终预测概率:\n", (X @ w + b).round(3))
print("损失:", round(float(loss), 4))
```

👀 你该看到的：4 个概率大约都在 0.5 附近，损失卡在约 0.69 不再下降——模型"左右为难"，谁也分不对。

⚠️ 常见坑

- 别以为是学习率或随机种子问题！**换 100 次随机种子结果都一样烂**，因为单层结构根本表达不了 XOR。这就是要加隐藏层的原因。

✅ 打卡：把损失曲线画出来（每 500 步记录一次 loss，折线图），确认它平台期卡在 0.69。

### Day 45 多层感知机结构：隐藏层解决 XOR

🎯 目标：搞清"加一个隐藏层就能学 XOR"的网络长什么样，并会数参数个数。

📖 理论（8分钟）

- 结构 **2−2−1**：输入 2 维 → 隐藏层 2 个神经元 → 输出 1 维。隐藏层用 tanh/sigmoid 做非线性变换，输出层用 sigmoid 给概率。
- **为什么能学 XOR**：隐藏层把输入"扭"到一个新空间，让原本线性不可分的 4 个点在隐藏层空间里变得可分。
- **数参数**（关键基本功）：
  - 输入→隐藏：权重矩阵 `W1` 形状 (2,2)，共 4 个；偏置 `b1` 形状 (2,)，共 2 个。
  - 隐藏→输出：权重矩阵 `W2` 形状 (2,1)，共 2 个；偏置 `b2` 形状 (1,)，共 1 个。
  - 合计 4+2+2+1 = **9 个参数**。
  - 通用公式：某层参数 = (上一层大小 × 本层大小) + 本层大小。

💻 动手（9分钟）——用数组把这套权重定义出来并数一遍

```python
import numpy as np

W1 = np.random.randn(2,2)*0.1
b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1
b2 = np.zeros(1)
total = W1.size + b1.size + W2.size + b2.size
print("W1形状", W1.shape, "b1形状", b1.shape)
print("W2形状", W2.shape, "b2形状", b2.shape)
print("参数总数:", total)     # 应为 9
```

👀 你该看到的输出：参数总数 = 9。

⚠️ 常见坑

- 最容易漏掉偏置 b。记住：**每层都有偏置**，所以参数 = 权重 + 偏置，不是只数 W。

✅ 打卡：把这个 2−2−1 网络画在纸上（输入 2 个圆、隐藏 2 个圆、输出 1 个圆，连线），每条连线标 "W"，每个圆旁标 "b"。

### Day 46 前向传播：逐层计算

🎯 目标：用 NumPy 写出 2−2−1 网络的一次前向（单个样本）。

📖 理论（8分钟）  
前向传播就是"从输入算到输出"，逐层做 `z = x@W + b` 然后激活：

```
z1 = x @ W1 + b1        # 线性变换到隐藏层
a1 = tanh(z1)           # 隐藏层激活（非线性！）
z2 = a1 @ W2 + b2       # 线性变换到输出层
a2 = sigmoid(z2)        # 输出层激活 → 最终概率
```

- `a1` 是关键：它把线性变换"扭"成非线性，没有它网络就退化回一条直线（回顾 Day 44）。

💻 动手（9分钟）

```python
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))

x  = np.array([0.0, 1.0])          # 单个 XOR 样本 (0,1)
W1 = np.random.randn(2,2)*0.1
b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1
b2 = np.zeros(1)

z1 = x @ W1 + b1
a1 = np.tanh(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

print("z1 形状:", z1.shape, "a1 形状:", a1.shape)
print("z2 形状:", z2.shape, "输出概率:", a2)
```

👀 你该看到的：z1/a1 形状都是 (2,)，z2/a2 形状 (1,)，输出是一个 0~1 的概率（随机权重下没意义，但形状对就行）。

⚠️ 常见坑

- `x @ W1`：x 是 (2,)、W1 是 (2,2) → 结果 (2,)。若写成 `W1 @ x` 会报形状错。口诀：**输入在左、权重在右，维度对齐中间**。
- 偏置 b1 是 (2,)，和 (2,) 的 z1 能直接加（逐元素）——这里用了 Day 3 的广播。

✅ 打卡：把 x 换成 (1,0) 再跑一次，确认输出概率变了（随机权重下不一定对，但形状一致）。

### Day 47 前向传播的矩阵形式（一次算一个 batch）

🎯 目标：把 4 个 XOR 样本叠成矩阵，一次前向算出全部 4 个输出——这是后面批量训练的基础。

📖 理论（8分钟）

- 单个样本时 x 是 (2,)；把 4 个样本竖着叠成 `X`，形状就是 (4,2)。
- 前向公式不变，只是输入输出都"加了第 0 维（batch 维）"：
  ```
  Z1 = X @ W1 + b1     # X:(4,2)  W1:(2,2)  → Z1:(4,2)
  A1 = tanh(Z1)        # (4,2)
  Z2 = A1 @ W2 + b2    # A1:(4,2) W2:(2,1)  → Z2:(4,1)
  A2 = sigmoid(Z2)     # (4,1)
  ```
- 偏置 `b1` 是 (2,) 会**广播**到每一行（4 行）——又是 Day 3 广播的功劳。

💻 动手（9分钟）

```python
import numpy as np

X  = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)   # 4 个样本
W1 = np.random.randn(2,2)*0.1
b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1
b2 = np.zeros(1)

Z1 = X @ W1 + b1
A1 = np.tanh(Z1)
Z2 = A1 @ W2 + b2
A2 = 1/(1+np.exp(-Z2))

print("Z1 形状:", Z1.shape, "A2 形状:", A2.shape)
print("4 个样本的输出概率:\n", A2)
```

👀 你该看到的：Z1 形状 (4,2)、A2 形状 (4,1)，A2 是 4 行 1 列的概率。

⚠️ 常见坑

- `b1` 形状 (2,) 能加在 (4,2) 上，是因为广播把 (2,) 复制成 (4,2)。写成 `b1.reshape(1,2)` 再 `+` 更直观，直接用 (2,) 也 OK（NumPy 自动从最后一维对齐）。
- 别把 W2 写成 (1,2)——应该是 (2,1) 才能让 (4,2)@(2,1)=(4,1)。

✅ 打卡：对比 Day 46 单个样本的结果，确认"批量版只是把样本堆在一起一次算"，逻辑完全一样。

### Day 48 把网络封装成类

🎯 目标：用第 2 周学的类，把权重和前向封装成 `NeuralNetwork`，以后调用像 `net.forward(X)` 一样干净。

📖 理论（8分钟）

- 类把"数据（W1,b1,W2,b2）"和"操作（forward）"绑在一起。`__init__` 里存权重，`forward` 里写前向公式。
- 这正是 Day 9 `Linear` 类的升级版——之前是单层线性，现在是两层带激活。如果对类还生，回 `Python二级衔接补充.md` 第 2 节再看一眼。
- 权重初始化仍乘 `0.1`（Day 9 养成的习惯，避免一开始数值太大）。

💻 动手（9分钟）

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

net = NeuralNetwork()
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
print(net.forward(X))
```

👀 你该看到的：一个 (4,1) 的概率数组（随机初始化，值在 0~1，但还没训练所以和 XOR 真值不符是正常的）。

⚠️ 常见坑

- `forward` 里用 `self.W1` 而不是 `W1`——忘记 `self.` 会报"找不到变量"。类里定义的属性，方法内都要加 `self.`。
- 初始化里 `W1` 不乘 `0.1` 会怎样？随机 `randn` 数值可能很大，经过 `tanh/sigmoid` 容易饱和（梯度≈0），后面训练会极慢甚至不动——所以 `0.1` 别省。

✅ 打卡：给 `NeuralNetwork` 加一个 `summary()` 方法，打印 4 个权重的形状和参数总数（应为 9），调用 `net.summary()` 验证。

### Day 49 复盘日：画前向传播计算图

🎯 目标：合上资料，把 2−2−1 前向传播"可视化"，彻底搞清每个中间量的形状。

📖 理论（8分钟）：回顾整条链路，每步的形状要能在脑子里立刻浮现：

```
X  (4,2)
 │  @ W1 (2,2)  + b1 (2,)  →  Z1 (4,2)
 │  tanh                →  A1 (4,2)
 │  @ W2 (2,1)  + b2 (1,)  →  Z2 (4,1)
 │  sigmoid             →  A2 (4,1)
```

💻 动手（9分钟）：二选一

- 方案 A（纸上）：画出上面那张图，每个箭头标运算、每个节点标形状。
- 方案 B（matplotlib）：用圆点+箭头画出 2−2−1 结构（输入 2 圆、隐藏 2 圆、输出 1 圆），在旁边注释形状。

⚠️ 常见坑

- 复盘重点是"形状怎么变的"，不是"激活函数怎么算"。把 X→Z1→A1→Z2→A2 的形状串起来，后面学反向传播时你会感谢今天的自己（反向传播就是这些形状的"反向"）。

✅ 打卡：不用写新代码，把 Day 46/47/48 的三份前向代码各跑一遍，再手画一张计算图存到 `week07` 文件夹。能不看资料说出每一步形状，就算过关。

---

---

## 第 8 周：反向传播（Day 50–56，最难也最关键）

> ⚠️ 本周是全计划最难的一周，但也是"真正理解神经网络"的门槛。别怕数学：你**不需要会背公式**，只需看懂"误差从后往前传、每层梯度 = 本层输入 × 上游误差"这个核心直觉。所有导数都用 NumPy 算，不用手推积分。卡住就看 3Blue1Brown 神经网络第 3 集。  
> 📌 二级友好：Day 50 的"计算图 + 链式法则"用最朴素语言讲，当作"流水线倒查责任"；Day 53 的梯度公式直接给"输入 × 误差"口诀，不用管它怎么来的。

### Day 50 计算图思想：前向算值，反向传导数

🎯 目标：建立"计算图"和"链式法则"的直觉，为反向传播打底。

📖 理论（8分钟）

- **计算图**：把复杂计算拆成一串小步骤，每步一个节点。如 `f(x) = (x+1)²` 拆成 `u = x+1`，`f = u²`。
- **前向**：代入 x 算出每个节点的值（u 和 f 的具体数字）。
- **反向**：想知道"x 动一点点 f 会动多少"（即 df/dx），从后往前逐节点传导数：
  - `df/du = 2u`（f=u² 的导数）
  - `du/dx = 1`（u=x+1 的导数）
  - 链式法则：`df/dx = df/du × du/dx = 2u × 1 = 2(x+1)`
- 直觉：全局梯度 = 沿途每一步"局部梯度"相乘，像流水线倒查"哪一步对最终产品影响最大"。

💻 动手（9分钟）——手算 + 数值验证

```python
import numpy as np

def f(x): return (x+1)**2
def df_dx(x): return 2*(x+1)              # 解析梯度（手推）
def num_grad(x, h=1e-5):                  # 数值梯度（永远是对的"标准答案"）
    return (f(x+h) - f(x-h)) / (2*h)

x = 3.0
print("解析梯度 df/dx(3) =", df_dx(x))    # 2*(3+1)=8
print("数值梯度 ≈", num_grad(x))          # 应≈8
```

👀 你该看到的：两者都约等于 8，几乎一样。

⚠️ 常见坑

- 数值梯度 `(f(x+h)-f(x-h))/(2h)` 是"中心差分"，比 `(f(x+h)-f(x))/h` 更准，记住用中心差分。
- 这是第 17 天数值导数的升级版，今天把它用到"多节点函数"上。

✅ 打卡：把 f 改成 `f(x) = 3x² + 2x + 1`，手推 df/dx 再用数值梯度验证。

### Day 51 输出层误差公式 δ_L = (ŷ−y)·σ′(z)

🎯 目标：拿到反向传播的"起点子弹"——输出层的误差信号 δ_L。

📖 理论（8分钟）

- 反向传播从"损失对输出的误差"开始，逐层往回传。对**交叉熵损失 + sigmoid 输出**，数学上能推出：
  - `δ_L = (ŷ − y)`（注意是预测减真实，不是反过来）
- `δ_L` 含义：损失 L 对每个输出 z 的梯度，它就是向后传的"误差子弹"。
- 标题里的 `(ŷ−y)·σ′(z)` 是**更通用**的写法：损失对激活值的导数 `∂L/∂ŷ` 乘以激活导数 `σ′(z)` 合起来就是 δ_L；对交叉熵+sigmoid 这个组合恰好化简为 `(ŷ−y)`。两种等价，先记住"交叉熵+sigmoid 时 δ_L = ŷ−y"即可。
- 🎬 对照 3Blue1Brown 神经网络第 3 集，看它对 δ 的可视化解释。

💻 动手（9分钟）——验证 δ_L

```python
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))

y_true = 1.0
z = 0.5
y_pred = sigmoid(z)
delta_L = y_pred - y_true          # 交叉熵+sigmoid 的 δ_L
print("y_pred:", round(y_pred,4), "δ_L:", round(delta_L,4))
```

👀 你该看到的：y_pred≈0.622，δ_L≈−0.378。

⚠️ 常见坑

- **符号**：δ_L = (ŷ − y) 还是 (y − ŷ)？梯度下降更新是 `w -= lr · dL/dw`，而 `dL/dz = ŷ − y`，所以这里是 **ŷ 减 y**。记反了训练会往反方向跑。
- XOR 用 sigmoid 输出，固定记 `ŷ−y`；如果输出层换成别的激活，δ_L 公式会变。

✅ 打卡：把 y_true 改成 0、z 改成 −1，再算一次 δ_L，确认符号和大小合理。

### Day 52 误差逐层回传：δ_隐藏 = (W₂ᵀ·δ_L) ⊙ tanh′(z)

🎯 目标：搞懂误差子弹 δ 怎么从输出层"倒流"回隐藏层。不要求背，要求看懂。

📖 理论（8分钟）

- 输出层的 δ_L 传回隐藏层公式：
  - `δ_1 = (W₂ᵀ · δ_L) ⊙ tanh′(z1)`
  - `W₂ᵀ · δ_L`：把误差通过权重"倒流"回隐藏层（权重转置 = 反向通道）
  - `⊙ tanh′(z1)`：再乘隐藏层激活的导数（每个神经元"还活不活跃"）
- 通用规律：**δ 从后往前，每过一层就"左乘权重的转置 + 逐元素乘该层激活导数"**。
- 直觉：误差像水，从输出往输入倒流，经过每层时被"管道粗细（权重）"和"阀门开度（激活导数）"调制。

💻 动手（9分钟）——只验证一个具体值（不用背公式）

```python
import numpy as np
W2 = np.array([[0.3],[-0.2]])
delta_L = np.array([[0.1]])
z1 = np.array([0.5, -0.3])
tanh_prime = 1 - np.tanh(z1)**2          # tanh'(z) = 1 - tanh(z)^2
delta_1 = (W2.T @ delta_L) * tanh_prime  # 矩阵乘 + 逐元素乘
print("δ_1 形状:", delta_1.shape, "值:", delta_1)
```

👀 你该看到的：δ_1 形状 (1,2)，值由公式算出。

⚠️ 常见坑

- `W₂ᵀ · δ_L` 用矩阵乘 `@`，不是逐元素 `*`；后面的激活导数是逐元素 `⊙`（就是 `*`）。两者别混。
- 这天的重点是"看懂结构"，公式以后写代码直接照抄，不用背。

✅ 打卡：把 W2 换成 (2,1) 随机值、δ_L 换成 (4,1) 批量误差，跑一遍确认形状对。

### Day 53 权重梯度 = 本层输入 × 上游误差

🎯 目标：写出 2−2−1 网络的 `backward()`，把误差转成每个权重的梯度（先只算、不更新）。

📖 理论（8分钟）  
梯度 = 本层接收的输入（转置）× 流回来的误差。对 2−2−1：

```
∂L/∂W2 = A1ᵀ @ δ_L        # 输入是 A1，误差是 δ_L
∂L/∂b2 = 按行求和(δ_L)
∂L/∂W1 = Xᵀ @ δ_1         # 输入是 X，误差是 δ_1
∂L/∂b1 = 按行求和(δ_1)
```

- 口诀：**"权重梯度 = 输入转置 × 误差；偏置梯度 = 误差求和"**。这就是 Day 20 梯度下降要的零件。
- 所有梯度都按"批量平均"（除以样本数 n）得到，后续更新时口径一致。

💻 动手（9分钟）——在 Day 48 的类里加 backward

```python
import numpy as np

class Net:
    def __init__(self):
        self.W1 = np.random.randn(2,2)*0.1
        self.b1 = np.zeros(2)
        self.W2 = np.random.randn(2,1)*0.1
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1/(1+np.exp(-self.z2))
        return self.a2

    def backward(self, x, y):
        n = len(x)
        delta_L = self.a2 - y                       # (ŷ−y)
        dW2 = self.a1.T @ delta_L / n
        db2 = np.sum(delta_L, axis=0, keepdims=True) / n
        delta_1 = (delta_L @ self.W2.T) * (1 - self.a1**2)
        dW1 = x.T @ delta_1 / n
        db1 = np.sum(delta_1, axis=0, keepdims=True) / n
        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}

net = Net()
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)
net.forward(X)
grads = net.backward(X, y)
for k,v in grads.items():
    print(k, v.shape)
```

👀 你该看到的：四个梯度形状和对应权重一致（W1 (2,2)、b1 (2,)、W2 (2,1)、b2 (1,)）。

⚠️ 常见坑

- `delta_1 = delta_L @ W2.T`（批量 delta_L 是 (4,1)，W2.T 是 (1,2)，得 (4,2)）。注意是 `delta_L @ W2.T` 不是 `W2.T @ delta_L`——顺序错了形状不对。
- 梯度必须除以 n（样本数），否则学习率要重调。

✅ 打卡：把 backward 返回的四个梯度打印数值，确认都不是全 0（全 0 说明前向没存对中间量）。

### Day 54 梯度校验（gradient check）—— 验证反传写对了

🎯 目标：用"数值梯度"这把尺子，验证你手写 backward 的梯度对不对。这是反向传播的标准验收方法。

📖 理论（8分钟）

- 数值梯度：`∂L/∂W₂[i,j] ≈ (L(W+ε) − L(W−ε)) / (2ε)`（ε 取 1e-5 左右）。
- 把手写 backward 的梯度和数值梯度比一比，相对误差应 < 1e-6（量级）。误差大说明 backward 写错了。
- 损失 L 用交叉熵（和 Day 26 一致）。

💻 动手（9分钟）

```python
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))
def loss_fn(W1,b1,W2,b2,X,y):
    z1 = X@W1+b1; a1 = np.tanh(z1)
    z2 = a1@W2+b2; a2 = sigmoid(z2)
    return -np.mean(y*np.log(a2+1e-8)+(1-y)*np.log(1-a2+1e-8))

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)
W1 = np.random.randn(2,2)*0.1; b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1; b2 = np.zeros(1)
eps = 1e-5
W2_p = W2.copy(); W2_p[0,0]+=eps
W2_m = W2.copy(); W2_m[0,0]-=eps
num_grad = (loss_fn(W1,b1,W2_p,b2,X,y)-loss_fn(W1,b1,W2_m,b2,X,y))/(2*eps)
print("数值梯度 W2[0,0]:", num_grad)
```

👀 你该看到的：打印出一个有限数（如 ±0.x），应和 Day 53 backward 算出的 dW2[0,0] 几乎相等（差 < 1e-6）。

⚠️ 常见坑

- ε 别太小（1e-8 因浮点精度反而误差大），也别太大（1e-2 泰勒展开误差大），用 **1e-5** 最稳。
- 梯度校验只在"调试"时用，平时训练不用（太慢），但能 100% 确认反传对不对。

✅ 打卡：把 Day 53 的 backward 算出的 dW2[0,0] 和这里数值梯度对比，确认相对误差很小。

### Day 55 拼成完整训练循环

🎯 目标：把前向 + 算损失 + 反向 + 更新串起来，训练 XOR 并画 loss 曲线。

📖 理论（8分钟）  
完整一步 = **前向 → 算 loss → 反向（算梯度）→ 更新（w −= lr·grad）**，重复很多轮（epoch）。

- 这正是 Day 31 线性回归训练循环的"升级版"：模型从一条直线变成 2−2−1 网络，损失从 MSE 换成交叉熵，梯度从手推公式变成 backward 自动算。

💻 动手（9分钟）

```python
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)
W1 = np.random.randn(2,2)*0.1; b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1; b2 = np.zeros(1)
lr, n = 0.1, len(X)
losses = []
for epoch in range(5000):
    z1 = X@W1+b1; a1 = np.tanh(z1)
    z2 = a1@W2+b2; a2 = sigmoid(z2)
    loss = -np.mean(y*np.log(a2+1e-8)+(1-y)*np.log(1-a2+1e-8))
    losses.append(loss)
    dL = a2 - y
    dW2 = a1.T@dL/n; db2 = np.sum(dL,0,keepdims=True)/n
    d1 = (dL@W2.T)*(1-a1**2)
    dW1 = X.T@d1/n; db1 = np.sum(d1,0,keepdims=True)/n
    W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2

print("最终预测:\n", a2.round(3))
print("最终损失:", round(float(loss),5))
```

👀 你该看到的：最终预测接近 [0,1,1,0]（顺序对应 4 个样本），损失降到 ~0.01 量级。

⚠️ 常见坑

- 前向时必须**保存中间量**（a1, a2），反向要用。若 backward 单独写函数，记得把 z1/a1/z2/a2 存成属性（像 Day 53 那样）。
- 损失卡在 0.69 不降 → 说明还在"单层"阶段（检查隐藏层激活是否真非线性、权重形状对不对）。

✅ 打卡：用 matplotlib 画 losses 曲线，确认它单调递减到接近 0。

### Day 56 里程碑：XOR 训练成功 🎉

🎯 目标：调参让 4 个输入全部分对、loss < 0.01——你已亲手实现了 1986 年让深度学习复活的反向传播。

📖 理论（8分钟）

- 里程碑标准：4 个 XOR 样本的预测概率，[0,0]/[1,1] 接近 0，[0,1]/[1,0] 接近 1；损失 < 0.01。
- 调参旋钮：训练轮数（不够就加，如 10000）、学习率（0.1 通常 OK，发散就减到 0.05）。
- 这是全计划第一个"真·神经网络"胜利——从 Day 44 单层永远失败的 XOR，到现在多层网络轻松拿下。

💻 动手（9分钟）——在 Day 55 基础上加轮数、打印判定

```python
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0], dtype=float).reshape(-1,1)
W1 = np.random.randn(2,2)*0.1; b1 = np.zeros(2)
W2 = np.random.randn(2,1)*0.1; b2 = np.zeros(1)
lr,n = 0.1,len(X)
for _ in range(10000):
    z1=X@W1+b1; a1=np.tanh(z1); z2=a1@W2+b2; a2=sigmoid(z2)
    dL=a2-y
    W1-=lr*(X.T@((dL@W2.T)*(1-a1**2))/n); b1-=lr*(np.sum((dL@W2.T)*(1-a1**2),0)/n)
    W2-=lr*(a1.T@dL/n); b2-=lr*(np.sum(dL,0)/n)
pred = (a2>0.5).astype(int)
print("预测类别:\n", pred.reshape(-1))
print("真实类别:\n", y.reshape(-1))
print("全部正确:", bool((pred==y).all()))
```

👀 你该看到的：预测类别 = [0,1,1,0]，与真实一致，`全部正确: True`。

⚠️ 常见坑

- 偶尔随机初始化会卡在局部最优（4 个没全对）。解决：重新随机初始化再跑，或多跑几轮。不属于代码错误。
- 学习率 0.5 可能发散（loss 变 NaN），降到 0.1 或 0.05 即稳。

✅ 打卡：截图保存"全部正确: True"的输出，这是你神经网络之路的第一枚勋章。🎉

---

> 📌 **第 8 周（Day 50–56）到此。** 你已用 NumPy 从零实现了反向传播、梯度校验、完整训练循环，并让一个 2−2−1 网络学会了 XOR——这正是 1986 年深度学习突破的核心。下一周（第 9 周）把这枚"核武器"搬到真实数据上：隐藏层宽度/深度、激活函数对比、权重初始化、mini-batch。  
> 下一步是**第 9 周（Day 57–63）：从玩具到真实数据（隐藏层宽度、ReLU vs sigmoid、初始化、mini-batch、训练 MNIST）**。告诉我「继续写第 9 周」我就出下一批。

---

# 第 9 周：从玩具到真实数据（Day 57–63）

> 🎯 **本周目标**：把你第 8 周写好的 XOR 网络，升级成能处理真实图片的神经网络。这一周你会明白三个"为什么"——为什么隐藏层要宽一点、为什么现在都用 ReLU、为什么要好的初始化。最后在 MNIST 手写数字上跑通，目标是测试集准确率 ≥ 90%。

## 第 57 天：隐藏层宽度/深度的影响

🎯 **目标**：直觉上理解"网络越大越能学"，并亲手验证。

📖 **理论（8分钟）**

- 隐藏层神经元数量 = 网络的"容量"。容量太小学不了复杂规律；太大则慢且易过拟合。
- 用 XOR（你已会）做实验：把 `2-2-1` 改成 `2-4-1`、`2-8-1`，看收敛轮数变化。
- 深度（层数）比宽度更难训练（梯度消失问题先不深挖，先体验"宽一点更好训"）。

💻 动手（9分钟）——复制你 Day 55 的训练循环，只改网络结构

```python
# 复用 Day55 的 forward/backward，仅把网络从 2-2-1 改成 2-8-1
# 隐藏层 8 个神经元：W1 形状 (2,8)，b1 形状 (8,)，W2 (8,1)，b2 (1,)
np.random.seed(0)
W1 = np.random.randn(2, 8) * 0.1
b1 = np.zeros(8)
W2 = np.random.randn(8, 1) * 0.1
b2 = np.zeros(1)

# 训练轮数固定 5000，记录最后 loss
# 对比 2-2-1 / 2-4-1 / 2-8-1 谁先到 loss<0.01
```

👀 **你该看到的输出**：`2-8-1` 通常在 5000 轮内就能把 loss 压到 0.01 以下，比 `2-2-1`（可能卡在 0.02）更稳更快收敛。

⚠️ **常见坑**

- 别忘了权重仍要乘 `0.1`（Day 9 养成的习惯），否则 `2-8-1` 的梯度会爆炸。
- 改结构时 `W1`、`b1`、`W2`、`b2` 四个形状要一起改，漏一个就报错。

✅ **打卡**：写出一句结论——"隐藏层加宽后，XOR 收敛更快/更稳"，存成 `week09_day57.py`。

## 第 58 天：激活函数对比——ReLU vs sigmoid

🎯 **目标**：理解为什么现代网络几乎都用 ReLU 当隐藏层激活。

📖 **理论（8分钟）**

- sigmoid 在两端"饱和"（导数≈0），多层叠加会让梯度越传越小（梯度消失），训练慢。
- **ReLU**：`max(0, x)`，右边导数恒为 1，梯度能顺畅流过，训练快。
- 同一网络用 sigmoid 隐藏层 vs ReLU 隐藏层，对比 loss 下降曲线。

💻 动手（9分钟）——把隐藏层激活从 tanh/sigmoid 换成 ReLU 再训练

```python
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)   # 导数：z>0 处为 1，否则 0

# 前向隐藏层改为：a1 = relu(X @ W1 + b1)
# 反向隐藏层 delta 改为：delta1 = (W2.T @ delta2) * relu_grad(z1)
# 输出层仍用 sigmoid（二分类），其余不变
# 分别跑 sigmoid 版和 ReLU 版，各画一条 loss 曲线对比
```

👀 **你该看到的输出**：ReLU 版 loss 下降明显更陡、更快到低位；sigmoid 版前期几乎"趴着不动"。

⚠️ **常见坑**

- ReLU 版可能遇到"神经元死亡"（某次更新后输出恒 0 且梯度 0），初期概率低，遇到就把学习率调小或重新初始化。
- 输出层做二分类**不要**用 ReLU（输出会被压在 0，无法表示接近 1 的概率），保持 sigmoid。

✅ **打卡**：截图两条 loss 曲线，存 `week09_day58.py`。

## 第 59 天：权重初始化——为什么不能全 0 / 太大

🎯 **目标**：理解初始化的作用，并亲手看到坏初始化的后果。

📖 **理论（8分钟）**

- 全 0 初始化：所有神经元学得一模一样，对称无法打破 → 网络"废了"。
- 太大（如 `randn` 不乘系数）：激活值飞到饱和区，梯度≈0，学不动。
- 太小：信号太弱，也学不动。
- 经验法则：`randn(n_in, n_out) * 0.1`（你一直用的）对小网络够用；更大网络用 `1/sqrt(n_in)` 更稳（先认识，不强制）。

💻 动手（9分钟）——三种初始化各跑一次看结果

```python
# 对比三种 W1 初始化（都用 2-4-1 网络、ReLU 隐藏层、训 3000 轮）：
# A) 全 0：np.zeros((2,4))
# B) 过大：np.random.randn(2,4) * 5
# C) 适中：np.random.randn(2,4) * 0.1
# 打印各自最后 loss，观察 A 几乎不动、B 可能发散、C 正常下降
```

👀 **你该看到的输出**：A 的 loss 卡在 ~0.69 附近（对称塌缩）；B 可能 NaN 或乱跳；C 顺利下降到低位。

⚠️ **常见坑**

- 偏置 `b1` 全 0 是 OK 的（只有权重不能全 0），别误伤。
- 看到 NaN 先检查是不是初始化太大 + 学习率太高叠加。

✅ **打卡**：记录"全 0 初始化 → loss 卡 0.69"的现象，存 `week09_day59.py`。

## 第 60 天：mini-batch 训练与 epoch 概念

🎯 **目标**：理解"一个 epoch = 把所有数据过一遍"，并用分批更新加速。

📖 **理论（8分钟）**

- 之前每次更新都用全部 4 个样本（批量梯度下降）。真实数据几千上万，全量更新每步很慢。
- **mini-batch**：每轮（epoch）把数据打乱，切成小批（如 8 个一批），每批更新一次权重。
- 好处：更新更频繁、还能借助随机性跳出局部最优，训练更快。

💻 动手（9分钟）——给 XOR 训练加 mini-batch（batch=2，4 个样本即 2 批/epoch）

```python
# 每个 epoch：
#   1) 用 np.random.permutation 打乱样本索引
#   2) 按 batch=2 切成小批
#   3) 对每个小批做前向+反向+更新（梯度是该批平均）
# 训 2000 个 epoch，打印每 200 epoch 的 loss
# 对比：和 Day55 全量更新比，结果应同样能学会 XOR
```

👀 **你该看到的输出**：loss 同样能降到低位、XOR 4 个全对；训练循环结构清晰（epoch 套 batch）。

⚠️ **常见坑**

- 每 epoch 必须重新打乱，否则顺序固定、batch 内容不变，失去随机性意义。
- batch 梯度要做平均（除以该批样本数），否则 batch 大小影响更新幅度。

✅ **打卡**：写出"1 epoch = 数据全过一遍"的定义，存 `week09_day60.py`。

## 第 61 天：MNIST 数据集认识（28×28 灰度图、10 类）

🎯 **目标**：第一次接触真实数据，看懂它的形状。

📖 **理论（8分钟）**

- MNIST：手写数字 0–9 的灰度图，每张 28×28 = 784 个像素，像素值 0–255。
- 共 70000 张（60000 训练 + 10000 测试），10 个类别。
- 我们要做的事：把 784 个像素"压平"成一行，喂给神经网络，输出 10 个数（每类一个分数）。

💻 动手（9分钟）——加载 MNIST 并可视化前 10 张

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)   # X 形状 (70000, 784)，y 是 0-9 字符串
print("X 形状:", X.shape, "y 形状:", y.shape)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"标签 {y[i]}")
    ax.axis('off')
plt.show()

# 把像素归一化到 0~1（后续训练更稳）：X = X / 255.0
# 划分训练/测试：前 60000 训练，后 10000 测试
```

👀 **你该看到的输出**：打印 `X 形状: (70000, 784) y 形状: (70000,)`；弹出 10 张手写数字图，每张下面标了正确数字。

⚠️ **常见坑**

- `fetch_openml` 首次会下载（几十 MB），慢一点正常；下完会缓存。
- 像素值默认 0–255，必须除以 255 归一化，否则 sigmoid 直接饱和、训不动。
- 标签是字符串 `"0".."9"`，要转成整数或 One-Hot，先 `astype(int)`。

✅ **打卡**：截图可视化图，存 `week09_day61.py`。

## 第 62 天：把网络升级为 784−64−10（输出层 softmax + 交叉熵）

🎯 **目标**：改造你的 `NeuralNetwork` 类，处理 10 分类。

📖 **理论（8分钟）**

- 输出层 10 个神经元，用 **softmax** 把 10 个分数变成"概率分布"（加起来=1）。
- 损失用**交叉熵**：预测越偏离真值，惩罚越大。
- 标签要转成 **One-Hot**（如数字 3 → `[0,0,0,1,0,0,0,0,0,0]`），才能和 10 维输出算交叉熵。
- 10 分类不再用 sigmoid 输出，隐藏层用 ReLU（Day 58 结论）。

💻 动手（9分钟）——改造网络类，加 softmax 输出与交叉熵

```python
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))   # 减最大值防溢出
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(y_true_onehot, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred + eps), axis=1))

# 网络结构：784 -> 64 (ReLU) -> 10 (softmax)
# 前向：a1 = relu(X @ W1 + b1); out = softmax(a1 @ W2 + b2)
# 反向：delta2 = out - y_onehot          (softmax+交叉熵的优雅结果)
#       delta1 = (W2.T @ delta2) * relu_grad(z1)
#       dW2 = a1.T @ delta2 / n;  db2 = delta2.sum(0)/n
#       dW1 = X.T @ delta1 / n;   db1 = delta1.sum(0)/n
# 用 Day61 的 MNIST 前 1000 张做小试验，跑几个 epoch 看 loss 下降
```

👀 **你该看到的输出**：loss 从 ~2.3（随机水平，ln10≈2.3）开始下降；One-Hot 标签形状 `(n,10)`；前向输出每行加起来≈1。

⚠️ **常见坑**

- softmax 必须减最大值再 exp，否则 `exp(大数)` 直接溢出成 inf。
- 交叉熵里 `y_pred` 要加 `eps` 防 `log(0)`（Day 26 学过）。
- `delta2 = out - y_onehot` 是 softmax+交叉熵组合的"免费"结果，不用再额外求导，记口诀即可。

✅ **打卡**：打印前几个样本的预测概率分布（应接近 One-Hot），存 `week09_day62.py`。

## 第 63 天：🎉 里程碑——手写 NN 跑通 MNIST（准确率 ≥ 90%）

🎯 **目标**：用纯 NumPy 手写的网络在 MNIST 测试集上达到 90%+ 准确率。

📖 **理论（8分钟）**

- 把所有零件串起来：mini-batch + ReLU 隐藏层 + softmax 输出 + 交叉熵 + 好的初始化。
- 训练 60000 张可能较慢（纯 NumPy 无 GPU），可先用 10000 张训练、测 10000 张，看能否 ≥ 90%。
- 准确率 = 预测类别（概率最大那列）与真实标签相同的比例。

💻 动手（9分钟）——完整训练 + 测试

```python
# 数据：X_train = X[:10000]/255.0, y_train = y[:10000]
#        X_test  = X[60000:61000]/255.0, y_test = y[60000:61000]
# 标签 One-Hot：y_oh = np.eye(10)[y_train]
# 网络：784-64-10，W1=randn(784,64)*0.1, b1=zeros(64), W2=randn(64,10)*0.1, b2=zeros(10)
# 训练：20 个 epoch，batch=64，lr=0.1，每个 epoch 打乱数据
# 测试：前向算出 out，pred = out.argmax(1)，acc = (pred==y_test).mean()
# 打印：每个 epoch 的 loss 与最终测试准确率
```

👀 **你该看到的输出**：测试准确率 ≥ 90%（通常 92%–95%）；loss 平稳下降。这就是你纯手写神经网络的第一个"真实战绩"。

⚠️ **常见坑**

- 训练慢是正常的（纯 CPU NumPy），10000 张 × 20 epoch 约几分钟，耐心等。
- 准确率上不去先检查：像素是否归一化、标签是否转 int/One-Hot、ReLU 漏没漏。
- 若某次卡在 ~85%，重跑（随机初始化不同）；或把隐藏层加到 128。

✅ **打卡**：截图"测试准确率: 0.93"这类输出，存 `week09_day63.py`。这是你本阶段的最高勋章 🏅——**从零手写、无任何深度学习框架，搞定真实图像分类**。

---

> 📌 **第 9 周（Day 57–63）到此。** 你已完成"纯手写神经网络"的全部内容：从 XOR 到 MNIST 90%+ 准确率，全部用 NumPy 徒手实现，没碰任何框架。这证明你**真正理解了**神经网络，而不是只会调库。  
> 下一步是**第 10 周（Day 64–70）：PyTorch 入门（张量、自动求导、用框架重写 MNIST）**，把你的手写知识映射到工业级工具。告诉我「继续写第 10 周」我就出下一批。

---

# 第 10 周：PyTorch 基础（Day 64–70）

> 🎯 **本周目标**：从你手写的知识过渡到工业框架 PyTorch。你会惊人地发现——前向传播、损失、反向传播、更新权重的"骨架"和你手写的完全一样，只是 PyTorch 把"算梯度"这件事用 `backward()` 一行替你做了。本周学完，你就能用框架几行代码复现上周的 MNIST。

## 第 64 天：为什么用框架 + Tensor 基础

🎯 **目标**：认识 PyTorch 的核心数据结构 Tensor（张量），并能在 NumPy 与 PyTorch 间转换。

📖 **理论（8分钟）**

- 框架三大价值：**自动求导**（不用手写 backward）、**GPU 加速**（大网络必备）、**丰富生态**（现成层、优化器、数据集）。
- **Tensor** = 带"计算图记忆"的数组，用法和 NumPy 几乎一样，只是包在 `torch` 里。
- 与 NumPy 互通：NumPy 数组 ↔ Tensor 可双向转换，方便复用你已有的数据。

💻 动手（9分钟）——创建张量并在 NumPy 间转换

```python
import torch, numpy as np
# 从列表创建
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t, t.shape, t.dtype)
# 常用创建
print(torch.zeros(2, 3))
print(torch.ones(2, 2))
print(torch.rand(2, 3))        # 均匀随机 0~1
print(torch.randn(2, 3) * 0.1) # 正态随机，乘 0.1（和手写习惯一致）
# 与 NumPy 互转
a = np.array([1.0, 2.0, 3.0])
t2 = torch.from_numpy(a)       # NumPy -> Tensor
back = t2.numpy()              # Tensor -> NumPy
print(t2, back)
# 简单运算
print(t + 1, t * 2, t @ t)    # 加、逐元素乘、矩阵乘
```

👀 **你该看到的输出**：张量打印带 `tensor(...)` 前缀；`from_numpy` 和 `.numpy()` 来回转换值不变；矩阵乘 `t @ t` 得 `[[7,10],[15,22]]`（和 NumPy 一致）。

⚠️ **常见坑**

- `torch.tensor` 默认是 `float32`，和 NumPy 的 `float64` 不同，混用时注意类型（一般无大碍）。
- GPU 暂不涉及，Tensor 默认在 CPU 上，先不管 `cuda`。

✅ **打卡**：写三行代码——创建一个 3×3 随机张量、转成 NumPy、再加 1 打印，存 `week10_day64.py`。

## 第 65 天：autograd 自动求导

🎯 **目标**：理解 PyTorch 最核心的魔法——`backward()` 自动算出梯度。

📖 **理论（8分钟）**

- 创建张量时设 `requires_grad=True`，PyTorch 会记录它参与的所有运算，形成"计算图"。
- 调用 `y.backward()` 后，所有 `requires_grad=True` 的叶子张量的 `.grad` 里就存着"y 对它的导数"。
- 这正是你 Day 50 手写的链式法则——现在框架全自动帮你做。

💻 动手（9分钟）——验证 x² 在 x=2 处导数是 4

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2          # 计算图：y = x^2
y.backward()        # 反向传播，求 dy/dx
print(x.grad)       # 应等于 4.0（导数 2x 在 x=2）
```

👀 **你该看到的输出**：打印 `tensor(4.)`——和你 Day 17 数值导数验证的 4 完全一致，但这次是精确解析导数。

⚠️ **常见坑**

- `backward()` 默认只能对标量（单个数字）调用；若 y 是向量需传一个同样形状的权重向量。
- 多次 `backward()` 梯度会**累加**，重复实验前用 `x.grad.zero_()` 清零（下一天会用到）。

✅ **打卡**：再试 `y = 3*x**3 + 2*x`，打印 `x.grad` 验证等于 `9*2²+2 = 38`，存 `week10_day65.py`。

## 第 66 天：nn.Module 与 nn.Linear——搭出 XOR 网络

🎯 **目标**：用框架的"积木"搭出你 Day 56 手写的 2-2-1 网络，体会代码量骤减。

📖 **理论（8分钟）**

- `nn.Linear(in, out)` 就是"线性层 y = x@W + b"，W/b 框架自动管理，你只给输入/输出维度。
- `nn.Tanh()`、`nn.ReLU()` 是激活函数层，直接当积木拼。
- `nn.Sequential(...)` 把多层按顺序串成一条管道，前向时一层接一层传。

💻 动手（9分钟）——用积木搭 XOR 网络

```python
import torch, torch.nn as nn
model = nn.Sequential(
    nn.Linear(2, 2),   # 输入 2 → 隐藏 2
    nn.Tanh(),         # 隐藏层激活（和你手写一致）
    nn.Linear(2, 1),   # 隐藏 2 → 输出 1
    nn.Sigmoid()       # 输出 sigmoid（二分类）
)
print(model)
# 试一次前向
X = torch.tensor([[0.,0],[0,1],[1,0],[1,1]])
print(model(X))       # 随机权重下的输出，约 0.5 附近
```

👀 **你该看到的输出**：打印出网络结构，前向输出 4 个接近 0.5 的值（随机初始化未训练）。这和你 Day 46 手写 `2-2-1` 前向是同一个网络，只是 W/b 不用你手动创建。

⚠️ **常见坑**

- `nn.Linear` 默认会带偏置 `b`；若想完全对照手写（你手写也带 b），不用改。
- 输入要是 `float` 张量，整数会报错——养成 `torch.tensor(..., dtype=torch.float)` 习惯。

✅ **打卡**：把隐藏层从 2 改成 8（`Linear(2,8)` + `Linear(8,1)`），再打印结构，存 `week10_day66.py`。

## 第 67 天：损失函数与优化器——用 torch 重写线性回归

🎯 **目标**：用框架的"损失 + 优化器"替换你 Day 31 手写的梯度下降，代码大幅缩短。

📖 **理论（8分钟）**

- `nn.MSELoss()` 就是你 Day 22 的均方误差；`nn.CrossEntropyLoss()` 是 Day 26 的交叉熵（自带 softmax，标签用整数即可）。
- `torch.optim.SGD(model.parameters(), lr=0.1)` 是随机梯度下降优化器，它接管"更新所有权重"的工作。
- 你不再手写 `dW`、`db`、逐参数更新——框架按计算图自动求导 + 优化器统一更新。

💻 动手（9分钟）——用框架重写第 5 周线性回归

```python
import torch, torch.nn as nn
# 数据：y = 3x + 1 + 噪声（参考 Day30）
x_np = np.linspace(0, 10, 100, dtype=np.float32).reshape(-1, 1)
y_np = (3 * x_np + 1 + np.random.randn(100,1)*0.5).astype(np.float32)
X = torch.from_numpy(x_np); y = torch.from_numpy(y_np)

model = nn.Linear(1, 1)                       # 一个神经元就是 y=wx+b
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()      # 清空旧梯度
    loss.backward()      # 自动算梯度
    opt.step()           # 自动更新权重
print("w,b =", model.weight.item(), model.bias.item())  # 应接近 3, 1
```

👀 **你该看到的输出**：打印的 `w` 接近 3、`b` 接近 1（和你 Day 31 手写结果一致），但代码从几十行缩到十几行。

⚠️ **常见坑**

- 漏掉 `opt.zero_grad()` 会让梯度累加，训练效果变差——记住这是"标准训练循环"的第 1 步。
- 优化器要传 `model.parameters()`（所有可训练参数），不是 `model`。

✅ **打卡**：把 `lr` 改成 0.1，观察是否更快收敛（或发散需调回），存 `week10_day67.py`。

## 第 68 天：标准训练循环模板（背下来）

🎯 **目标**：掌握 PyTorch 的 5 行训练骨架，这是之后所有模型的模板。

📖 **理论（8分钟）**

- 框架训练循环固定四步：
  1. `opt.zero_grad()` —— 清空上轮的梯度
  2. `pred = model(X)` —— 前向
  3. `loss = loss_fn(pred, y)` —— 算损失
  4. `loss.backward()` —— 自动反传
  5. `opt.step()` —— 更新参数
- 包在 `for epoch` 里重复即可。对比你 Day 55 手写循环，骨架完全一致，只差"反传和更新"由框架代劳。

💻 动手（9分钟）——用模板训练一个 2-2-1 网络解 XOR

```python
import torch, torch.nn as nn
X = torch.tensor([[0.,0],[0,1],[1,0],[1,1]])
y = torch.tensor([[0.],[1],[1],[0]])   # 浮点标签
model = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(5000):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    if epoch % 1000 == 0: print(epoch, loss.item())
print("预测:\n", model(X).detach().round())   # 应接近 [[0],[1],[1],[0]]
```

👀 **你该看到的输出**：loss 从约 0.25 降到很低；最后 `round()` 后的预测 = `[[0],[1],[1],[0]]`，和 Day 56 手写版一模一样。

⚠️ **常见坑**

- 打印预测时加 `.detach()`（或 `.detach().numpy()`），否则张量带计算图、不能直接 round/打印友好。
- XOR 偶尔训不过（随机初始化），重跑即可，和手写版同理。

✅ **打卡**：把这 5 行模板抄一遍记到笔记本上——这是你之后所有 PyTorch 代码的骨架。

## 第 69 天：DataLoader 与模型保存

🎯 **目标**：学会分批加载数据、保存/加载训练好的模型。

📖 **理论（8分钟）**

- `DataLoader(dataset, batch_size=64, shuffle=True)` 自动帮你做"打乱 + 分批"，就是你 Day 60 手写的 mini-batch，但一行搞定。
- `TensorDataset(X, y)` 把特征和标签打包成数据集。
- 模型保存：`torch.save(model.state_dict(), 'm.pt')`；加载：`model.load_state_dict(torch.load('m.pt'))`——只存权重，不存结构（结构由你的代码定义）。

💻 动手（9分钟）——用 DataLoader 训练并保存模型

```python
from torch.utils.data import DataLoader, TensorDataset
# 接第 68 天 XOR 数据，但用 DataLoader 喂
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
loss_fn = nn.MSELoss(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(3000):
    for xb, yb in loader:           # 自动分批
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
torch.save(model.state_dict(), "xor_model.pt")   # 保存权重
# 加载回来验证
m2 = nn.Sequential(nn.Linear(2,2), nn.Tanh(), nn.Linear(2,1), nn.Sigmoid())
m2.load_state_dict(torch.load("xor_model.pt"))
print("加载后预测:\n", m2(X).detach().round())
```

👀 **你该看到的输出**：加载回来的模型预测仍 = `[[0],[1],[1],[0]]`，证明权重被正确存盘+读回了。

⚠️ **常见坑**

- `load_state_dict` 前必须先用同样结构 `nn.Sequential(...)` 建一个空模型，否则不知道权重往哪放。
- `DataLoader` 的 `shuffle=True` 等价于你手写的"每 epoch 打乱"。

✅ **打卡**：确认 `xor_model.pt` 文件已生成（在工作目录里能看到），存 `week10_day69.py`。

## 第 70 天：🎉 复盘日——torch 版 XOR vs 手写版

🎯 **目标**：复盘本周，明确"框架帮你省了哪部分"，巩固理解。

📖 **理论（8分钟）**

- 对照 Day 56（手写）与 Day 68（torch）：模型结构相同，损失相同（MSE），训练循环骨架相同。
- **框架替你做的**：手写 `backward()`（逐层算 δ、算 dW/db）、手写 `opt.step()`（逐参数更新）——这两块 PyTorch 用 `loss.backward()` + `opt.step()` 两行替代。
- 你省下的是"容易写错的反向传播代码"，但**理解**一点没省——这正是前 9 周手写的价值。

💻 动手（9分钟）——闭卷用模板再写一个分类小网络

```python
# 不翻答案，用今天记的 5 行模板，训练一个 2-4-1 网络解 XOR（隐藏层 4 个、ReLU 激活）
# 目标：预测 = [[0],[1],[1],[0]]，并画出 loss 随 epoch 的下降曲线
```

👀 **你该看到的输出**：用框架 10 行左右就能解 XOR，loss 曲线平滑下降，预测全对。

⚠️ **常见坑**

- 复盘重点不是"背 API"，而是能说清"前向 / 损失 / 反传 / 更新"四步在你手写和框架里分别对应什么。

✅ **打卡**：写一段 100 字总结——"PyTorch 和我手写的区别只在反传和更新两行，其余思维完全一样"，存 `week10_day70.py`。

---

> 📌 **第 10 周（Day 64–70）到此。** 你已用 PyTorch 复现了前 9 周的核心成果（线性回归、XOR），并掌握了 Tensor、autograd、nn.Module、优化器、DataLoader、模型存读。你手写的知识没有白费——框架只是把"算梯度"自动化了。  
> 下一步是**第 11 周（Day 71–77）：CNN 入门（与智能车辆最相关）**，用框架搭卷积网络，并了解它在车道线/交通标志识别中的应用。告诉我「继续写第 11 周」我就出下一批。

---

# 第 11 周（Day 71–77）：CNN 入门

> 这一周是和**你的智能车辆专业**最贴的一周：卷积神经网络（CNN）就是车道线检测、交通标志识别、行人/车辆检测的底子。前面你训的 MNIST 用的是"全连接网络"（把图拉成一长条再连），而 CNN 尊重图像的"局部性"——它用小窗口扫图，专门吃图像。

## Day 71 卷积：局部感受野、权值共享（从手算开始）

🎯 **目标**：搞懂"卷积"到底是什么操作——用一个窗口在图上滑，逐点做乘加，得到"特征图"。

📖 **理论（8分钟）**

- **为什么不用全连接**：一张 28×28 的图拉平是 784 个数，若下一层 256 个神经元，光这一层就有 784×256≈20 万个参数；CNN 用"局部感受野 + 权值共享"能把参数砍到几百个。
- **卷积核（kernel / filter）**：一个小矩阵（如 3×3）。它在输入图上**从左到右、从上到下**滑动，每停一个位置，就把"窗口覆盖的图块"和"核"做**对应元素相乘再求和**，得到输出图的一个点。
- **输出尺寸**（无 padding、stride=1）：`输入 − 核 + 1`。所以 8×8 图、3×3 核 → 6×6。
- **不同核提取不同特征**：比如"中间是 8、周围是 −1"的核能勾出物体的**边缘轮廓**（平坦区域相乘抵消≈0，边缘处突变被放大）。

💻 **动手（9分钟）**

```python
import numpy as np

def conv2d(img, kernel):
    H, W = img.shape
    K = kernel.shape[0]
    out_h, out_w = H - K + 1, W - K + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i:i+K, j:j+K]
            out[i, j] = np.sum(patch * kernel)   # 对应元素相乘再求和
    return out

np.random.seed(0)
img = np.random.rand(8, 8)
blur = np.full((3, 3), 1/9)               # 模糊核
edge = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])  # 边缘检测核

print("模糊输出形状:", conv2d(img, blur).shape)   # (6, 6)
print("边缘输出:\n", np.round(conv2d(img, edge), 2))
```

👀 **你该看到的输出**：两个输出都是 `(6, 6)`；边缘核的输出在"像素突变"处数值明显大，平坦处接近 0。

⚠️ **常见坑**

- 循环范围要写 `range(H-K+1)`、`range(W-K+1)`，写成 `range(H)` 会索引越界。
- 严格数学上的"卷积"要先把核翻转（180°）再做乘加，但**深度学习里说的卷积其实是"互相关"，不翻转**——别纠结这点，大家都这么用。

✅ **打卡**：能手写 `conv2d` 跑通 8×8→6×6，并口头说出"边缘核为什么能勾轮廓"。

---

## Day 72 卷积层参数：stride / padding / 通道

🎯 **目标**：搞懂 stride（步长）、padding（补边）、通道（channels），会用框架的二维卷积层。

📖 **理论（8分钟）**

- **stride（步长）**：每次滑动跳几格。stride=2 → 输出尺寸直接减半。
- **padding（补边）**：在图四周补一圈 0，防止尺寸被核"吃"掉。
- **输出尺寸公式**（必背）：`(W − K + 2P) / S + 1`（向下取整）。
- **通道**：彩色图有 RGB 3 通道；卷积层输入/输出也可以有多通道。第一层把 1 通道变成 4 通道，就是同时提取 4 种特征（4 张"特征图"）。

💻 **动手（9分钟）**

```python
import torch
import torch.nn as nn

# 手算：28×28, 核3, padding1, stride1 → (28-3+2)/1+1 = 28
layer = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
x = torch.randn(1, 1, 28, 28)        # 形状必须是 (N, C, H, W)
print("输出形状:", layer(x).shape)    # torch.Size([1, 4, 28, 28])
```

👀 **你该看到的输出**：`torch.Size([1, 4, 28, 28])` —— 和手算一致。

⚠️ **常见坑**

- `nn.Conv2d` 吃的是 `(N, C, H, W)` 顺序，**不是** `(N, H, W, C)`，写错会报错。
- PyTorch 的 `padding=P` 是**每边都补 P** 行/列，所以公式里是 `2P`。

✅ **打卡**：能手算任意 (W,K,P,S) 的输出尺寸，并用 `nn.Conv2d` 验证。

---

## Day 73 池化与 CNN 整体结构（LeNet）

🎯 **目标**：理解池化（降采样），并搭一个 LeNet 风格小网络，打印每层形状。

📖 **理论（8分钟）**

- **池化（Pooling）**：把图切成 2×2 小块，取每块最大值（MaxPool）或平均值，尺寸减半。作用：减少计算、增强"平移不变性"（物体稍微挪动也能认出）。
- **LeNet 经典结构**：`Conv → Pool → Conv → Pool → Flatten → FC → FC`。前半段"提取特征"，后半段全连接"做分类"。

💻 **动手（9分钟）**

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 4, 3), nn.ReLU(), nn.MaxPool2d(2),   # 32→30→15
    nn.Conv2d(4, 8, 3), nn.ReLU(), nn.MaxPool2d(2),   # 15→13→6  (注意这里用13)
    nn.Flatten(),
    nn.Linear(8*6*6, 64), nn.ReLU(),
    nn.Linear(64, 10),
)
x = torch.randn(1, 1, 32, 32)
for name, layer in model.named_children():
    x = layer(x)
    print(f"{name:12s} -> {tuple(x.shape)}")
```

> 注：上面 32→30→15→13→6 是逐层算出的（pool 默认 stride=kernel_size 会减半；15/2 向下取整=7 这里用 13 是为了演示，实际请用偶数输入如 28×28 更顺，自己改 32→28 重算）。

👀 **你该看到的输出**：每一层后都打印出 `-> (1, 通道, H, W)`，整条链路形状连贯不报错。

⚠️ **常见坑**

- `MaxPool2d(2)` 默认 `stride=2`，会直接把尺寸减半；奇数尺寸会被向下取整，用偶数输入更省心。
- `Flatten` 之前要算清特征数：`通道 × H × W`，别拍脑袋。

✅ **打卡**：能搭出 LeNet 并逐层打印形状，说出"卷积提特征、全连接做分类"。

---

## Day 74 ⭐ 训练 CNN on MNIST

🎯 **目标**：把上周的 MLP 换成 CNN，训练 1 个 epoch，对比准确率（应更高）。

📖 **理论（8分钟）**

- CNN 用"局部感受野"天然适合图像，通常比同量级 MLP 更准。
- 训练流程和第 68 天的"五步模板"一模一样：`zero_grad → forward → loss → backward → step`。

💻 **动手（9分钟）**（在前面 DataLoader/模型基础上，把模型换成 CNN）

```python
cnn = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28→14
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 14→7
    nn.Flatten(), nn.Linear(16*7*7, 64), nn.ReLU(), nn.Linear(64, 10),
)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
# 训练1个epoch（沿用 Day69/70 的 train_loader、test_loader）
for X, y in train_loader:
    X = X.unsqueeze(1)          # 1×28×28 加通道维 → 1×1×28×28
    opt.zero_grad(); loss = loss_fn(cnn(X), y); loss.backward(); opt.step()
```

（测试集准确率计算同前，把模型换成 `cnn`、输入加通道维即可。）

👀 **你该看到的输出**：CNN 训练 1 个 epoch 的测试准确率比第 63 天 MLP 同轮次**高 1–3%**。

⚠️ **常见坑**

- MNIST 图片是 `1×28×28`，喂给卷积层前必须 `unsqueeze(1)` 加通道维，否则报尺寸错。
- 记得图片已归一化到 0–1（Day 61 做的）。

✅ **打卡**：CNN 跑通、准确率超过当时的 MLP。

---

## Day 75 可视化：卷积核长什么样

🎯 **目标**：看看第一层卷积核学到了什么（边缘、斑点、方向）。

📖 **理论（8分钟）**：卷积核权重就是"特征探测器"，把它画出来能直观理解网络"在看什么"。

💻 **动手（9分钟）**

```python
import matplotlib.pyplot as plt
kernels = cnn[0].weight.detach().numpy()     # 形状 [8, 1, 3, 3]
k = (kernels - kernels.min()) / (kernels.max() - kernels.min())  # 归一到 0~1
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(k[i, 0], cmap="gray"); ax.axis("off")
plt.show()
```

👀 **你该看到的输出**：8 张 3×3 小图，能隐约看到一些边缘/方向性图案。

⚠️ **常见坑**：权重值范围可能不是 0–1，画图前**必须归一化**（减最小除以极差），否则图全黑/全白。

✅ **打卡**：成功画出第一层卷积核图。

---

## Day 76 CNN 在车辆上的应用概念（选做）

🎯 **目标**：建立"CNN 能用在智能车辆感知"的认知，并对接你的毕设（GTSRB 交通标志）。

📖 **理论（8分钟）**

- 车道线检测、交通标志识别（你的毕设）、行人/车辆检测，底层都是 CNN。
- `torchvision.models` 有预训练模型（ResNet、MobileNet 等），加载即可推理。

💻 **动手（9分钟，选做）**

```python
from torchvision import models, transforms
from PIL import Image
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
net.eval()
prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
img = prep(Image.open("一张本地图片.jpg")).unsqueeze(0)
with torch.no_grad():
    probs = net(img).softmax(1)[0]
# 打印 top-5（类别名需配合 weights.meta["categories"]）
```

👀 **你该看到的输出**：预训练模型能正确识别常见物体（猫、狗、汽车等）。

⚠️ **常见坑**：预训练模型对输入有**固定要求**（224×224、特定均值方差归一化），直接喂原图会识别错。

✅ **打卡**：说出 CNN 与车辆应用的对应，并写明"把模型用于 43 类交通标志"只需把输出类别数改成 43、换成 GTSRB 数据集。

---

## Day 77 复盘日：画 CNN 结构图

🎯 **目标**：合上资料，画出 CNN 数据流并标注每层尺寸。

💻 **动手（9分钟）**：在纸上/注释里推导这条链路（输入 1×28×28）：

```
输入 1×28×28
→ Conv(8通道,3×3,padding1) → 1×8×28×28
→ MaxPool(2)                → 1×8×14×14
→ Conv(16通道,3×3,padding1)→ 1×16×14×14
→ MaxPool(2)                → 1×16×7×7
→ Flatten                   → 16*7*7 = 784
→ FC(64) → FC(10)
```

不写完整训练，只要每个尺寸推导正确。

✅ **打卡**：能独立推导任意 CNN 每层的输出尺寸。

---

> 📌 **第 11 周（Day 71–77）到此。** 你已从"手写卷积"走到"用 PyTorch 搭 LeNet 训 MNIST、 visualizing 卷积核"，并把它和你的专业（车道线/交通标志识别）对上了。CNN 是这个计划里离智能车辆最近的一块。  
> 下一步是**第 12 周（Day 78–84）：训练技巧（正则化、Dropout、早停）+ 毕业项目 GTSRB 启动与训练**。告诉我「继续写第 12 周」我就出最后一批，写完整个 84 天计划就完整了。

---

# 第 12 周（Day 78–84）：训练技巧 + 毕业项目 GTSRB

> 这是 84 天计划的**最后一周**。前 11 周你从 NumPy 一路走到 CNN，这一周做两件事：① 学几个让模型更稳的训练技巧；② 用它们真正跑通你的毕设方向——交通标志分类。

## Day 78 训练技巧：正则化、Dropout、早停

🎯 **目标**：掌握三种防过拟合手段，给 CNN 加 Dropout 重训看效果。

📖 **理论（8分钟）**

- **过拟合**：模型训练集很准、测试集很差，等于"背下了训练样本的噪声"。
- **Dropout（随机失活）**：训练时随机把一部分神经元输出置 0（如 20%），强迫网络不依赖某个特定节点——相当于同时训了很多个子网络再集成，泛化更好。
- **早停（Early Stopping）**：盯住验证集损失，不再下降就停，避免在噪声上继续"死磕"。
- **正则化（L2 / weight decay）**：在损失里加"权重平方和"的惩罚，让权重不敢长太大。

💻 **动手（9分钟）**（在 Day 74 的 CNN 基础上改）

```python
import torch.nn as nn
cnn_drop = nn.Sequential(
    nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*7*7, 128), nn.ReLU(),
    nn.Dropout(0.2),            # 关键：训练时随机关 20% 输出
    nn.Linear(128, 10),
)
# 训练 3 个 epoch（同 Day74 流程），对比加 Dropout 前后的测试准确率
```

👀 **你该看到的输出**：加 Dropout 后训练损失可能略高，但测试/验证准确率**更稳或更高**。

⚠️ **常见坑**

- Dropout **只在训练时生效**；评估/测试前必须切到 `model.eval()`，否则随机失活还在起作用，结果会飘。
- 丢弃概率 `p` 太大（如 0.8）会欠拟合，几乎啥都学不到，0.2–0.5 是常用区间。

✅ **打卡**：能说出 Dropout 防过拟合的原理，并跑通加 Dropout 的版本。

---

## Day 79 超参数与调参思路

🎯 **目标**：理解"调参本质是做小实验"，亲手做一次网格实验。

📖 **理论（8分钟）**

- **超参数**：学习率、batch size、隐藏层宽度、Dropout p 等——不是模型学出来的，是人设定的。
- **网格搜索**：把几个候选值两两组合，各跑一遍，挑最好的。

💻 **动手（9分钟）**

```python
results = {}
for lr in [0.01, 0.001]:
    for bs in [32, 128]:
        acc = train_and_eval(lr=lr, batch_size=bs, epochs=2)  # 复用前面的训练函数
        results[(lr, bs)] = acc
for k, v in results.items():
    print(k, f"{v:.3f}")
```

👀 **你该看到的输出**：一张 4 行的小表格，通常 `lr=0.001` 比 0.01 更稳；小 batch 慢但泛化可能更好。

⚠️ **常见坑**

- 真实项目用随机搜索/贝叶斯优化，网格太费时；本计划只做 4 组体会思路。
- **每组实验前固定随机种子**，否则结果不可比（这次好可能只是运气）。

✅ **打卡**：能列出 4 组结果并指出哪组最优、为什么。

---

## Day 80 学习率调度

🎯 **目标**：用学习率调度器让训练后期更稳。

📖 **理论（8分钟）**

- 固定学习率后期容易在最优点附近"来回震荡"；逐步衰减让它切换到"细调"模式。
- **StepLR**：每 N 个 epoch 把 lr 乘 gamma（如 0.5）衰减。

💻 **动手（9分钟）**

```python
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
for epoch in range(6):
    train_one_epoch(cnn, opt)
    scheduler.step()           # 每个 epoch 后调用一次
    # 记录 loss，画曲线
```

👀 **你该看到的输出**：衰减版的 loss 曲线后期下降更平滑、最终更低（对比 Day74 固定 lr）。

⚠️ **常见坑**：调度器的 `step()` 要放在每个 epoch（或每个 batch，看调度类型）**之后**调用，忘了就白配。

✅ **打卡**：能用调度器跑通并解释为什么衰减有帮助。

---

## Day 81 毕业项目启动：交通标志分类（GTSRB）

🎯 **目标**：认识 GTSRB 数据集结构，确立毕设方向。

📖 **理论（8分钟）**

- **GTSRB**：德国交通标志数据集，共 **43 类**（限速、禁行、转向等），几千张彩色图，尺寸从 15×15 到 250×250 不等，每张带类别标签。
- 这正是 CNN 的典型任务，和你的智能车辆专业强相关——你的毕设方向就是它。

💻 **动手（9分钟）**（假设已下载到 `data/gtsrb/`）

```python
import os
from collections import Counter
root = "data/gtsrb/train"
counts = Counter()
for class_dir in os.listdir(root):
    counts[class_dir] = len(os.listdir(os.path.join(root, class_dir)))
print("类别数:", len(counts))                 # 43
import matplotlib.pyplot as plt
plt.bar(list(counts.keys()), list(counts.values()))
plt.title("每类样本数"); plt.show()
```

👀 **你该看到的输出**：类别数 = 43；直方图显示各类样本**不均衡**（有的类上千张，有的几百张）。

⚠️ **常见坑**：数据集较大，第一次跑用**子集**（如前 10 类、前 2000 张）先把流程跑通，再上全量。

✅ **打卡**：能描述 GTSRB 结构并说出类别数 = 43。

---

## Day 82 项目：数据加载与预处理

🎯 **目标**：写好数据管道，把原始图变成模型能吃的张量。

📖 **理论（8分钟）**

- 用 `Dataset` + `DataLoader` 批量化；所有图统一 resize 到 32×32，标准化到均值 0、方差 1。
- 彩色图有 **3 通道**，模型第一层 `in_channels` 要设 3（不是 1）。

💻 **动手（9分钟）**

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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
                          transforms.Normalize([0.5]*3, [0.5]*3)])
loader = DataLoader(GTSRB("data/gtsrb/train", tfm), batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)        # [32, 3, 32, 32] [32]
```

👀 **你该看到的输出**：`torch.Size([32, 3, 32, 32])`、`torch.Size([32])`。

⚠️ **常见坑**

- 彩色图通道数是 3，模型第一层 `in_channels=3`。
- `Normalize` 的均值/方差用**训练集**统计，别用全量数据。

✅ **打卡**：能跑出 DataLoader 并打印一个 batch 形状 `[32,3,32,32]`。

---

## Day 83 项目：训练与评估

🎯 **目标**：训练 GTSRB 分类器，输出准确率和混淆矩阵。

📖 **理论（8分钟）**

- 把 Day 74 的 CNN 改两处：`in_channels=3`（彩色）、`out_channels=43`（类别数），套用"五步训练模板"即可。
- **类别不均衡时，总准确率会误导**——要看混淆矩阵和每类召回率。

💻 **动手（9分钟）**

```python
model = nn.Sequential(
    nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32*8*8, 128), nn.ReLU(),
    nn.Dropout(0.3), nn.Linear(128, 43),
)
# 训练几个 epoch（先用小数据集跑通）；最后用 sklearn 画混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_true, y_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        y_true += y.tolist(); y_pred += model(X).argmax(1).tolist()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, cmap="Blues"); plt.show()
```

👀 **你该看到的输出**：测试准确率（小数据集上就能到较高水平）；混淆矩阵里能看出某两类（如限速 30 与 50）最容易互混。

⚠️ **常见坑**：类别不均衡时别只盯总准确率——混淆矩阵上少数类的误判才是重点。

✅ **打卡**：输出测试准确率 + 混淆矩阵，并指出最易混淆的类别对。

---

## Day 84 🎉 总复盘

🎯 **目标**：回顾 12 周，写总结、定方向。

💻 **动手（9分钟）**

1. 翻看 12 周笔记，列出你最自豪的 **3 个成果**（如：手写 XOR、MNIST 90%、搭 CNN、跑通 GTSRB）。
2. 写约 **300 字**《我现在理解的神经网络》——从"一个神经元 = 加权求和 + 激活"讲到"多层前向 + 反向传播 + 梯度下降训练"。
3. 给下一阶段定个方向，例如：
   - **目标检测（YOLO）**：在图像里框出车辆/行人（智能车辆感知核心）；
   - **语义分割**：把车道线逐像素标出来（车道保持基础）；
   - **Transformer / 多模态**：更前沿的架构。

✅ **打卡**：完成 300 字总结 + 方向规划。**整个 84 天计划到此全部完成！**

---

> 🎊 **恭喜！84 天计划全部完成。** 你从"计算机二级 Python"起步，到能纯手写 NumPy 神经网络解 XOR、跑通 MNIST 90%，再到用 PyTorch 搭 CNN、跑通自己的交通标志分类毕设——这是一次完整的"理解神经网络"之旅。后面是真正的工程与方向深耕，按需继续就好。
