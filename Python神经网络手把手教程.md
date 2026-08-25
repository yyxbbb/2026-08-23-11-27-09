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

💻 **动手（9分钟）**
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)   # 0 到 2π 等分为 100 个点
plt.plot(x, np.sin(x), label="sin(x)")

# 随机取 30 个点当"样本"
pts = np.random.rand(30) * 2 * np.pi
plt.scatter(pts, np.sin(pts), color="red", label="样本点")

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
> 下一步是**第 4 周（Day 22–28）：损失函数、激活函数、概率与交叉熵**。告诉我「继续写第 4 周」我就出下一批。
