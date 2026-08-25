# Python → 神经网络 · 每日练习题（Day 1–14）

> 配套：`Python神经网络手把手教程.md`（Day 1–14）
> 用法：**先自己写出代码跑通，再看「参考答案」**。每题目标：5–10 分钟内独立完成。
> 本文件原则：每天的小题**覆盖当天教程教到的所有知识点**，不提前使用没学过的函数。
> 约定：默认已 `import numpy as np`、`import matplotlib.pyplot as plt`。

---

## 第 1 天 练习题：创建数组（覆盖：六种创建方式、.shape/.dtype/.ndim、reshape、两个常见坑）

**题目**
1. 用 `np.array` 从嵌套列表 `[[1,2],[3,4],[5,6]]` 创建一个 3×2 矩阵 `a`，打印 `a.shape`、`a.dtype`、`a.ndim`。
2. 用 `np.arange(12)` 创建后 `reshape` 成 `(3,4)`，打印结果和 `shape`。
3. 分别创建并打印：`np.zeros((2,3))`、`np.ones((2,2), dtype=int)`、`np.full((2,3), 7)`。
4. **坑（不运行先猜）**：`np.array([1,2],[3,4])`（少了外层括号）会怎样？正确写法是什么？
5. **坑**：`np.arange(10).reshape(3,4)` 会报错吗？为什么？（看元素总数）

**提示**：创建函数当天都教过；`reshape` 的新形状元素总数必须等于原数组。

**参考答案**：
```python
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
```

---

## 第 2 天 练习题：索引、切片与布尔索引（覆盖：单元素、切片、布尔索引、掩码、.size）

**题目**
1. `m = np.arange(12).reshape(3, 4)`。取出「第 1 行第 2 列」的元素，值是多少？（行、列从 0 数）
2. 用切片取出「前 2 行、第 1~2 列」的子矩阵。
3. 用布尔索引取出 `m` 中所有大于 6 的元素，并打印 `big.size`（个数）。
4. 把 `m` 中大于 6 的位置标成 1、其余 0，得到掩码 `mask`（用 `(m > 6).astype(int)`），打印 `mask`。

**提示**：单元素 `m[行, 列]`；切片左闭右开；布尔索引返回一维数组。

**参考答案**：
```python
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
```

---

## 第 3 天 练习题：广播机制（覆盖：规则、加每列、加每行、报错）

**题目**
1. `M = np.ones((3, 4)) * 10`。构造 `v_col = np.array([1, 2, 3]).reshape(3, 1)`，算 `M + v_col`，描述每一列分别多了什么。
2. 构造 `v_row = np.array([1, 2, 3, 4]).reshape(1, 4)`，算 `M + v_row`，描述每一行分别多了什么。
3. 手算并用代码验证：`a = np.array([[1],[2],[3]])`（3×1）与 `b = np.array([[10, 20, 30]])`（1×3）相加，结果是什么形状、什么值？
4. **报错验证**：直接 `np.array([1, 2, 3]) + M`（形状 `(3,)` 和 `(3,4)`）会报什么错？用 `try/except` 捕获并打印错误信息。

**提示**：广播从最后一维往前对齐；`(3,)` 加不到 `(3,4)` 上，要先 `reshape` 成 `(3,1)` 或 `(1,3)`。

**参考答案**：
```python
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
```

---

## 第 4 天 练习题：矩阵乘 vs 逐元素乘（覆盖：@、*、手算验证、转置 .T）

**题目**
1. `A = np.array([[1,2],[3,4]])`，`B = np.array([[5,6],[7,8]])`。分别算 `A @ B` 和 `A * B`。
2. 手算验证 `A @ B` 第 0 行第 0 列的值（1×5 + 2×7）。
3. 算 `A.T` 并打印，再算 `(A.T).T`，验证它等于 `A`。

**提示**：`@` 是矩阵乘（前列=后行），`*` 是对应位置相乘；`.T` 行列互换。

**参考答案**：
```python
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
```

---

## 第 5 天 练习题：随机数与种子（覆盖：randn、seed、mean/std、rand vs randn、不设种子）

**题目**
1. `np.random.seed(0)`，生成 10000 个标准正态分布 `randn` 样本，打印 `mean` 和 `std`，它们应接近什么数？
2. 再次 `seed(0)` 生成同样的 10000 个，用 `np.array_equal` 证明两次完全一致。
3. 用 `np.random.rand(5)`（0~1 均匀分布）生成 5 个数，打印观察——对比 `randn` 的取值范围差异。
4. **不设种子**，`np.random.randn(3)` 连续跑两次，两次结果一样吗？打印说明。

**提示**：标准正态均值≈0、标准差≈1；`rand` 取值在 0~1，`randn` 可取负数且分布更宽；同种子→同结果，不设种子→每次不同。

**参考答案**：
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
```

---

## 第 6 天 练习题：画图（覆盖：plot、scatter、label、legend、xlabel/ylabel/title、show）

**题目**
1. 用 `np.linspace(-3, 3, 100)` 生成 x，画 `y = x**2` 的折线，设 `label="x^2"`。
2. 在同一张图里，随机取 30 个点（`x_sample = np.random.rand(30)*6 - 3`，`y_sample = x_sample**2 + np.random.randn(30)*0.5` 当带噪声样本），用 `plt.scatter` 画出来并设 `label="样本"`。
3. 加 `xlabel`、`ylabel`、`title("抛物线")`、`legend()`、`show()`。

**提示**：同一张图连续调用 `plot`/`scatter` 即可叠加；`legend()` 依赖每个图的 `label`。

**参考答案**：
```python
x = np.linspace(-3, 3, 100)
plt.plot(x, x**2, label="x^2")

x_sample = np.random.rand(30) * 6 - 3
y_sample = x_sample**2 + np.random.randn(30) * 0.5
plt.scatter(x_sample, y_sample, color="red", label="样本")

plt.xlabel("x"); plt.ylabel("y")
plt.title("抛物线")
plt.legend()
plt.show()
```

---

## 第 7 天 练习题：第 1 周综合小测（覆盖：Day 1–6 全部知识点）

**题目**（尽量不翻资料）
1. 创建 `np.arange(9).reshape(3,3)`，打印它的 `shape`、`dtype`、`ndim`。
2. 取它的「第 2 行」和「前两行、第 2~3 列」子矩阵。
3. 用布尔索引取出所有大于 4 的元素，并用 `(m>4).astype(int)` 得到掩码。
4. 写一个 3×3 矩阵，让 `[1,2,3]` 通过广播加到每一列（先 reshape 成正确形状）。
5. 定义 `A=[[1,0],[0,1]]`（单位阵）、`B=[[2,3],[4,5]]`，分别算 `A@B` 和 `A*B`，并算 `B.T`。
6. `np.random.seed(0)` 后生成 `randn(1000)`，打印均值；再把种子设回 0 生成一次，证明两次相同。

**参考答案**：
```python
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
```

---

## 第 8 天 练习题：函数封装（覆盖：默认参数、*args、封装成 make_matrix、可变默认参数坑）

**题目**
1. 写一个 `make_matrix(rows, cols, seed=0)`：先 `np.random.seed(seed)`，返回 `np.random.rand(rows, cols)`。分别调用 `make_matrix(2, 3)` 和 `make_matrix(2, 3, seed=1)`，打印说明两者是否相同。
2. 写一个 `total(*nums)` 返回所有位置参数的和，调用 `total(1, 2, 3, 4)`。
3. **坑验证**：写 `def f(x, lst=[]): lst.append(x); return lst`，连续调用 `f(1)`、`f(2)` 会怎样？说明为什么默认参数用可变对象危险。

**提示**：默认参数可省略；`*nums` 收集所有位置参数；默认参数只在函数定义时创建一次，可变对象会跨调用共享。

**参考答案**：
```python
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
```

---

## 第 9 天 练习题：Linear 类（上）（覆盖：class、__init__、self、y=x@W+b、0.1 初始化、zeros 偏置、forward）

**题目**
1. 从零写一个 `Linear` 类：`__init__(self, in_dim, out_dim, seed=0)` 用 `np.random.randn(in_dim, out_dim) * 0.1` 初始化 `self.W`、用 `np.zeros(out_dim)` 初始化 `self.b`；`forward(self, x)` 返回 `x @ self.W + self.b`。
2. 创建 `layer = Linear(2, 3)`，打印 `layer.W.shape`（应 `(2,3)`）、`layer.b`（3 个 0），并解释为什么 W 要乘 `0.1`。
3. `x = np.array([[1.0, 2.0]])`，算 `layer.forward(x)`，确认输出形状是 `(1, 3)`。

**提示**：`in_dim` 是输入维、`out_dim` 是输出维；`x` 必须是二维（哪怕 1 个样本也要写成 `[[...]]`）。

**参考答案**：
```python
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
```

---

## 第 10 天 练习题：`__call__`（覆盖：__call__ 让实例可调用、layer(x)==forward(x)、必须 return）

**题目**
1. 在第 9 天的 `Linear` 类里加 `__call__(self, x)`，让它返回 `self.forward(x)`。
2. 创建 `layer = Linear(1, 1)`，手动设 `layer.W = np.array([[2.0]])`、`layer.b = np.array([1.0])`，分别用 `layer.forward(np.array([[3.0]]))` 和 `layer(np.array([[3.0]]))` 验证结果一致（应为 `[[7.]]`）。
3. **坑验证**：写个 `__call__` 忘记写 `return`（即 `def __call__(self, x): self.forward(x)` 没有 return），调用 `layer(x)` 会得到什么？说明。

**提示**：`__call__` 让 `对象(参数)` 等价于 `对象.forward(参数)`；忘了 return 调用结果就是「空」。

**参考答案**：
```python
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
```

---

## 第 11 天 练习题：异常处理与文件读写（覆盖：try/except、np.save、np.load、FileNotFoundError、array_equal、save 是复制非引用）

**题目**
1. 把 `np.arange(12).reshape(3, 4)` 保存到 `m.npy`，读回并用 `np.array_equal` 验证与原数组完全一致。
2. 用 `try / except FileNotFoundError` 去读一个不存在的文件 `no_such.npy`，捕获后打印「文件不存在，请检查路径」。
3. **验证 save 是「复制」**：保存后修改原数组 `a[0, 0] = 999`，再读回 `b`，打印 `b[0,0]` 看变没变，说明 save 存的是当时的值还是引用。

**提示**：`np.load` 找不到文件抛 `FileNotFoundError`；`np.save` 写的是数组当时的快照，之后改原数组不影响已存文件。

**参考答案**：
```python
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
```

---

## 第 12 天 练习题：Jupyter 操作（覆盖：新建 notebook、Markdown cell、Code cell、画图、保存）

**题目**：新建一个 Jupyter Notebook，按顺序做：
1. 第一个 cell 设为 **Markdown**，写 `# 第12天练习：Jupyter 试水`，运行它（应渲染成标题）。
2. 新建 **Code** cell，顶部加 `%matplotlib inline`，导入 numpy 和 plt，画 `x = np.linspace(0, 10, 50)` 的 `sin` 曲线并 `plt.show()`。
3. 把它保存为 `day12_practice.ipynb`。

**提示**：Code cell 顶部加 `%matplotlib inline` 让图直接显示；变量在所有 cell 间共享，改了要重跑相关 cell。

**参考答案**：无标准代码，能跑出 sin 图并保存 notebook 即达标。Code cell 内容：
```python
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x))
plt.title("day12")
plt.show()
```

---

## 第 13 天 练习题：距离与相似度（覆盖：euclidean、cosine_sim、np.linalg.norm、方向无关长度）

**题目**
1. 写 `euclidean(a, b)`（返回 `np.linalg.norm(a - b)`）和 `cosine_sim(a, b)`（返回 `a @ b / (np.linalg.norm(a) * np.linalg.norm(b))`）。
2. `a = [1,0,0]`，`b = [0,1,0]`：欧氏距离和余弦相似度各是多少？
3. 把 `b` 换成 `[2,0,0]`（与 a 同向但长度不同）：余弦变不变？欧氏距离变不变？
4. `a` 与自身的余弦应等于 1，验证一下。

**提示**：正交向量余弦=0；同向向量余弦=1，且不受长度影响；欧氏距离会随长度变。

**参考答案**：
```python
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
```

---

## 第 14 天 练习题：第 2 周综合自测（覆盖：Day 8–13 全部知识点）

**题目**（尽量不翻资料）
1. 写 `make_matrix(rows, cols, seed=0)`（默认参数 + 封装），调用两次不同种子验证不同。
2. 写 `total(*nums)` 返回所有参数之和，调用 `total(2, 4, 6)`。
3. 从零写 `Linear` 类（含 `__init__/forward/__call__`），输入 3 维、输出 2 维，创建实例后 `layer(np.array([[1.0, 2.0, 3.0]]))` 跑通。
4. 用 `np.save` 保存 `np.arange(10)`，再 `np.load` 读回并用 `np.array_equal` 验证；另写 `try/except FileNotFoundError` 处理读不存在文件。
5. 写 `euclidean` 和 `cosine_sim`，算 `a=[1,2,3]` 与 `b=[4,5,6]` 的欧氏距离和余弦相似度。
6. 为什么要给随机数设 `seed`？

**参考答案**：
```python
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
```

---

---

## 第 15 天 练习题：向量点积（覆盖：点积代数=几何、交换律、余弦比较方向、np.isclose、点积受长度影响）

**题目**
1. `a=np.array([1.,2.])`, `b=np.array([3.,4.])`。算 `a@b`；再用 `np.linalg.norm(a)*np.linalg.norm(b)*cosine_sim(a,b)` 重建几何值，用 `np.isclose` 验证两者相等。
2. 验证 `a@b == b@a`（用 `np.isclose`）。
3. 给定 `a=[1,2]`, `b=[3,4]`, `c=[-2,1]`，算 `cosine_sim(a,b)` 与 `cosine_sim(a,c)`，说明谁与 a 同向、谁与 a 正交（`cosine_sim` 需自己定义）。
4. **坑**：`d=[100,200]`（d 是 a 的 100 倍）。算 `a@d` 与 `cosine_sim(a,d)`，说明"比较方向要用余弦而非裸点积"。

**提示**：`cosine_sim(a,b)=a@b/(||a||*||b||)`；正交向量余弦=0。

**参考答案**：
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
```

---

## 第 16 天 练习题：矩阵变换（覆盖：M@p 旋转、行/列向量形状、(M@p.T).T、画变换前后、axis equal）

**题目**
1. `M=np.array([[0.,-1.],[1.,0.]])`, `p=np.array([1.,0.])`。用 `(M @ p.T).T` 算变换后位置，打印并说明转到哪。
2. 用 M 作用于单位正方形四角 `corners=np.array([[0.,1.,1.,0.],[0.,0.,1.,1.]])`，打印 `M @ corners` 变换后坐标。
3. 画图：变换前 `'o-'`，变换后 `'s-'`，加 `plt.axis('equal')`、`legend()`、`show()`。
4. **坑**：若把点写成一维 `p1=np.array([1.,0.])`（shape (2,)），直接 `M @ p1` 会怎样？说明为什么要用 `(M @ p.T).T` 或 `p @ M.T`。

**提示**：行向量点 `p @ M.T` 也能算；画图必须 `axis('equal')` 才不变形。

**参考答案**：
```python
import numpy as np, matplotlib.pyplot as plt
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
```

---

## 第 17 天 练习题：数值导数（覆盖：单边差分、解析对比、不同 h 的误差、f=x³ 验证、中心差分更准）

**题目**
1. 写 `num_deriv(f, x, h=1e-5)` 用单边差分 `(f(x+h)-f(x))/h`，验证 f(x)=x² 在 x=3 处≈6。
2. 写 f(x)=x³，验证在 x=2 处数值导数≈12（解析 3x²=12）。
3. 试 h=1e-1,1e-3,1e-5,1e-7 打印各自数值导数，说明 h 太小误差反而变大。
4. **坑**：用单边 `(f(x+h)-f(x))/h` 与中心差分 `(f(x+h)-f(x-h))/(2h)` 各算一次 x² 在 x=3 处，哪个更准（更接近 6）？

**提示**：单边差分误差 O(h)，中心差分 O(h²)；x² 解析导数是 2x。

**参考答案**：
```python
import numpy as np
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

## 第 18 天 练习题：数值梯度（覆盖：偏导、中心差分、梯度向量、在(1,2)验证[2,4]、np.allclose、梯度=上升方向）

**题目**
1. 写 `num_grad(f, xy)` 用中心差分，验证 f(x,y)=x²+y² 在 (1,2) 处≈[2,4]。
2. 验证 f(x,y)=x²+2y² 在 (1,1) 处梯度≈[2,4]（∂/∂x=2x=2，∂/∂y=4y=4）。
3. 用 `np.allclose` 比较数值梯度与解析梯度。
4. **坑（方向验证）**：在 (1,2) 处沿梯度走一小步 `p2=(1,2)+0.1*[2,4]`，比较 f(p2) 与 f(1,2) 谁大？说明梯度指向上升方向。

**提示**：梯度是 `[df/dx, df/dy]`；用 `xy.copy()` 避免改动原数组。

**参考答案**：
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
```

---

## 第 19 天 练习题：链式法则（覆盖：复合、数值导数验证、f'(g(x))·g'(x)、f'在g(x)处求值）

**题目**
1. `g=x²`, `f=u+1`，写 `h=f(g(x))`，用中心差分数值导数验证 h'(3)≈6。
2. 链式法则解析：f'(u)=1, g'(x)=2x → 1·2·3=6，与数值对比。
3. 换一组：`g=sin(x)`, `f=u³`，验证 h(x)=sin(x)³ 在 x=0 处导数≈0（链式法则 f'(0)·g'(0)=0·1=0）。
4. **坑**：本题 f'(u)=3u²，要在 g(0)=0 处算 f'(g(0))=0，而不是 f'(0)（虽然这里都等于0）。写出"链式法则里 f' 必须在 g(x) 处求值"这句话并对照代码。

**提示**：链式法则是 `h'(x)=f'(g(x))·g'(x)`；`np.sin` 本题直接使用。

**参考答案**：
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
```

---

## 第 20 天 练习题：梯度下降（覆盖：更新规则、lr、轨迹图、大lr发散、起点不敏感）

**题目**
1. 写梯度下降求 f(x)=x²−4x 最小值，lr=0.1，起点 x=0，迭代30步，打印最终 x 与 f(x)。
2. 画轨迹（plot xs 曲线 + `'ro-'` 轨迹点 + legend + show）。
3. 把 lr 改成 1.5，打印最终 x，说明发生了什么（发散/震荡）。
4. **坑**：起点改成 x=10、lr=0.1，最终还能收敛到 2 吗？验证梯度下降对起点不敏感（凸函数）。

**提示**：更新 `x = x - lr*df(x)`；df(x)=2x−4；轨迹点用 `history` 列表存每步 x。

**参考答案**：
```python
import numpy as np, matplotlib.pyplot as plt
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
```

---

## 第 21 天 练习题：第 3 周综合自测（覆盖：Day 15–20 全部知识点）

**题目**（尽量不翻资料）
1. 用 NumPy 验证 `a@b == |a||b|cosθ` 且 `a@b==b@a`（用 `np.isclose`）。
2. 旋转矩阵 `[[0,−1],[1,0]]` 把 (1,0) 变到哪？画出来。
3. 数值导数验证 f(x)=x³ 在 x=2 处≈12。
4. 数值梯度验证 f(x,y)=x²+2y² 在 (1,1) 处≈[2,4]。
5. 链式法则验证 h=f∘g（g=x², f=u+1）在 x=3 处 h'≈6。
6. 从零用梯度下降求 f(x)=x²−4x 最小值并画轨迹。

**参考答案**：综合前 6 天代码拼接即可（见各天答案）。打卡标准：6 题独立做对，存成 `week03_review.ipynb`。

---

> 📌 **Day 15–21 练习题到此。** 每天小题已覆盖当天教程全部知识点。做完这批，你对"向量→导数→梯度→梯度下降"这条数学线就通了。
> 告诉我「继续写第 4 周」，我就出下一批（损失函数、激活函数、概率与交叉熵的教程 + 练习题）。
