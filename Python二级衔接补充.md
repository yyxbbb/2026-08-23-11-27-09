# Python 二级 → 神经网络学习 衔接补充

> 这份文档是给**计算机二级 Python 水平**的同学看的。
> 你已经在二级里学会了：变量、循环（`for`/`while`）、条件（`if`）、**函数**（`def`）、基础列表/字典、基础文件读写。
> 但本学习计划（尤其是从 Day 8 开始）会用到一些二级几乎不教的东西：**类（面向对象）**、`*args`、**异常捕获**、以及整个 **NumPy 数组库**。
> 这份补充专门把你"会"的和"不会的"接起来，让你读教程时不再卡壳。

---

## 0. 先看清：二级会什么、本计划缺什么

| 你已经会的（二级） | 本计划需要、但二级没教的 | 出现在哪天 |
|---|---|---|
| `def 函数名(参数):` 固定参数 | `def f(*nums):` 变长参数 | Day 8 |
| 普通函数 | `class` 类、`self`、`__init__`、`__call__` | Day 9–10、后面全天 |
| `open()` 读文本文件 | `try/except 异常类型 as e` | Day 11 |
| 列表 `list`、字典 `dict` | NumPy 的 `ndarray` 数组 + 向量化运算 | Day 1–14 |
| 高数里的"导数"概念 | 用代码算数值导数、梯度、梯度下降 | Day 15–21 |

**一句话**：你缺的不是"学不会"，而是没人用二级的语言给你讲过这些。下面逐个补上。

---

## 1. 最大的坎：类（Class）从零讲起 ⭐

神经网络里**每一层**都是一个"类"，你 Day 9 就要写第一个类 `Linear`。先把类讲透，后面 80% 的代码你就通了。

### 1.1 二级的你：只会"函数"

二级里你写过一个函数，比如：

```python
def add(a, b):
    return a + b

print(add(2, 3))   # 5
```

特点：**给数据 → 吐结果**，数据和函数是分开的两样东西。

### 1.2 为什么神经网络要用"类"？

一个神经网络的层，既要**存东西**（权重 `W`、偏置 `b`），又要**会算东西**（前向传播 `forward`）。
- 用函数：每次算都要把 `W`、`b` 当参数传来传去，很乱。
- 用类：把"数据"和"会做的事"打包成一个对象，像一台小机器——里面有零件（`W`、`b`），也能按按钮运行（`forward`）。

### 1.3 直接拆 Day 9 的 `Linear` 类

教程 Day 9 的完整代码：

```python
import numpy as np

class Linear:                      # ① 类的定义（一张"图纸"）
    def __init__(self, in_dim, out_dim, seed=0):   # ② 建对象时自动调用
        np.random.seed(seed)
        self.W = np.random.randn(in_dim, out_dim) * 0.1   # ③ 把权重存进"自己"
        self.b = np.zeros(out_dim)                        # ③ 把偏置存进"自己"

    def forward(self, x):          # ④ 这个对象"会做的事"之一
        return x @ self.W + self.b

layer = Linear(in_dim=2, out_dim=3)   # ⑤ 照图纸造一个真实对象
x = np.array([[1.0, 2.0]])
print(layer.forward(x))           # ⑥ 让这个对象干活
```

逐行翻译成二级能懂的话：

- **① `class Linear:`** —— 定义一张叫 `Linear` 的"图纸"。注意结尾有冒号，下面所有缩进的代码都属于这个类。
- **② `def __init__(self, in_dim, out_dim, seed=0):`** —— 这是"构造函数"：`layer = Linear(2, 3)` 这句话一执行，Python 就**自动调用**它来"把对象建出来"。`in_dim=2, out_dim=3` 是你传进去的参数。
- **关键：`self` 是什么？** —— `self` = "**这个正在被创建的对象自己**"。你调用 `Linear(2,3)` 时并没有传 `self`，是 Python 偷偷把"新对象"塞给了 `self`。你可以把 `self` 想成函数里的第一个固定摊位，专门用来装"这个对象自己"。
- **③ `self.W = ...` / `self.b = ...`** —— 把权重 `W` 和偏置 `b` 存进"这个对象自己的口袋里"。以后随时能 `layer.W`、`layer.b` 拿出来用。这就是类比函数强的地方：**数据被记住了**。
- **④ `def forward(self, x):`** —— 这是这个对象"会做的一件事"（方法），计算 `y = x @ W + b`。注意它也有 `self`，因为要取"自己口袋里的 `W`、`b`"。
- **⑤ `layer = Linear(in_dim=2, out_dim=3)`** —— 照图纸造出一个**真实对象**（叫实例），名字叫 `layer`。
- **⑥ `layer.forward(x)`** —— 让 `layer` 这个对象执行它的 `forward` 方法。

### 1.4 一个生活比喻

- `class Linear` = 一份"蛋糕配方"（图纸）。
- `layer = Linear(2,3)` = 照配方烤出一个真实蛋糕（对象）。
- `self` = 这个具体蛋糕自己。配方里写"往**自己**里放糖"，每个蛋糕都有自己的糖量，互不干扰。
- `self.W` = 这个蛋糕里实际放的糖（数据，被记住）。
- `layer.forward(x)` = 把这个蛋糕拿去称重/装饰（让它干活）。

### 1.5 Day 10 的 `__call__` 是啥？

Day 10 在类里加了：

```python
def __call__(self, x):
    return self.forward(x)
```

`__call__` 是 Python 的"魔法方法"：定义了它之后，你的对象就能**像函数一样被调用**：

```python
print(layer(x))            # 等价于 layer.forward(x)
print(layer.forward(x))    # 两种写法结果一样
```

为什么这么折腾？因为 PyTorch 里层就是 `layer(x)` 这么用的，提前习惯，后面无缝衔接。你只要记住：**`__call__` 让"对象(参数)"变成合法写法**。

### 1.6 现在轮到你验证一下（照教程 Day 9 做）

把 Day 9 的 `Linear` 代码敲一遍，再改一行试试：

```python
# 改这一行，把 * 0.1 去掉，看看 W 的数值是不是变大了
self.W = np.random.randn(in_dim, out_dim)      # 去掉 * 0.1
print(layer.W)
```

你会看到 W 的数值从"零点几"变成"几"，这就是教程说的"初始值太大"的直观感受。改回 `* 0.1` 即可。

---

## 2. Day 8 的 `*args`：能吃任意多个参数的函数

二级里你写的函数参数是**固定个数**：

```python
def f(a, b):
    return a + b
f(2, 3)        # 必须正好 2 个
```

教程 Day 8 出现：

```python
def total(*nums):
    return sum(nums)
print(total(1, 2, 3, 4))   # 10
```

`*nums` 前面的星号意思是：**把调用时传进来的所有位置参数，收集成一个元组** `nums`。
- 上面 `total(1,2,3,4)` 里，`nums` 实际是 `(1, 2, 3, 4)`。
- `sum(nums)` 把元组里的数加起来得 10。

一句话：`*args` 让函数"吃"任意多个参数。教程里目前只是让你"认识"，后面写通用模型时才会常用，先记住这一种写法就够了。

---

## 3. Day 11 的 `try/except`：接住会报错的程序

二级可能没细讲异常。教程 Day 11 的写法：

```python
try:
    np.load("不存在的文件.npy")          # 可能出错的一行
except FileNotFoundError as e:           # 如果真的报这个错
    print("捕获到错误:", e)              # 就执行这里，程序不崩
```

翻译成人话：
- `try:` 里面放"**可能出错**"的代码。
- 如果它真的抛出了 `FileNotFoundError`（文件找不到），程序**不会崩溃**，而是跳去执行 `except` 下面的代码。
- `as e` 是把错误信息存到变量 `e` 里，方便打印出来看。

二级提醒你两点坑（教程也写了）：
1. **别写裸 `except:`**（不带异常类型），它会连你按 `Ctrl+C` 想中断都拦住。
2. **指定具体异常类型**（`FileNotFoundError`）比笼统捕获更安全。

---

## 4. NumPy 的"数组思维"（Day 1–14 的底层逻辑）

二级里你只有**列表** `list`：`a = [1, 2, 3]`。神经网络全靠 **NumPy 数组 `ndarray`**，它和列表最大的不同是：

| 列表（二级） | NumPy 数组（本计划） |
|---|---|
| `a = [1, 2, 3]` | `a = np.array([1, 2, 3])` |
| 想每个元素加 1，要写 `for` 循环 | 直接 `a + 1` 整个数组同时加（**向量化**） |
| 没有"形状"概念 | 有 `shape`，比如 `(2, 3)` 表示 2 行 3 列 |
| 没有矩阵乘 | 有 `@` 矩阵乘（`a @ b`） |

**为什么非用 NumPy 不可**：神经网络本质是"一大堆数字按矩阵乘来乘去"。用列表 + `for` 循环又慢又难写，NumPy 一行 `x @ W + b` 就搞定。

**你天天要盯的 `shape`**：神经网络里"形状对不上"是最常见的报错。比如 `x` 是 `(1, 2)`（1 个样本、2 维），`W` 是 `(2, 3)`（2 进 3 出），`x @ W` 才合法，结果是 `(1, 3)`。教程每天让你 `print` 形状，就是帮你养成"先想形状"的习惯。

> 你二级的列表知识完全够用——把 `np.array(列表)` 理解为"把列表升级成会整体运算的数组"就行。

---

## 5. 数学部分（Day 15–21）说明：不用手推公式

Day 15–21 会讲向量点积、导数、梯度、梯度下降。你可能会担心数学不够。放心：

- **导数/梯度**在这里是用**代码数值计算**的（例如 `(f(x+h)-f(x))/h`），不需要你手推复杂公式。
- 你二级配套的高数（导数概念）已经够用，教程会一步步带。
- 梯度下降（Day 20）本质就是"反复微调一个数让它变小"，和二级里"用循环不断逼近"的思路一致，只是换了 NumPy 写法。

---

## 6. 开始 Day 1 前的自检清单

打开 Python（或 Jupyter），确认你能做到这几件二级该会的事；做不到先回二级复习一下：

- [ ] 能写 `def f(x): return x*2` 并调用 `f(3)`
- [ ] 能写 `for i in range(3): print(i)`
- [ ] 能写 `if x > 0: ... else: ...`
- [ ] 能建列表 `a = [1, 2, 3]` 并取 `a[0]`
- [ ] 能 `import` 一个库（如 `import math`）

如果上面都 OK，你就具备了读这份教程的全部前置能力。剩下 Day 8 以后的类、`*args`、异常，本补充文档已替你补齐。

---

## 7. 教程天数 → 需要的补充知识 对照表（更新到 Day 84）

| 教程天数 | 用到的"超二级"点 | 看本文哪节 |
|---|---|---|
| Day 1–7 | NumPy 数组、reshape、广播、随机 | 第 4、8 节 |
| Day 8 | `*args` 变长参数 | 第 2 节 |
| **Day 9–10** | **类、self、`__init__`、`__call__`** | **第 1 节（重点）** |
| Day 11 | `try/except` 异常 | 第 3 节 |
| Day 15–21 | 数值导数/梯度（代码算，非手推） | 第 5 节 |
| Day 21–23 | 激活函数、softmax / 交叉熵 | 第 9 节 |
| Day 28 | 手写 `backward()`（梯度） | 第 5 节 + 第 10.2 |
| Day 62 | softmax + 交叉熵实战 | 第 9 节 |
| **Day 64** | **PyTorch Tensor** | **第 10.1 节** |
| **Day 66** | **nn.Module / Sequential** | **第 10.3 节** |
| **Day 67** | **optim 三步走** | **第 10.4 节** |
| **Day 69** | **DataLoader / 模型保存** | **第 11 节** |
| **Day 70+** | **CNN（Conv2d / MaxPool / Dropout）** | **第 12 节** |
| Day 70–84（毕业项目） | `transforms` 预处理流水线、`PIL.Image` 读图、`os.listdir` 遍历、`StepLR` 学习率衰减 | 第 13 节 |

---

## 8. NumPy 进阶：广播、reshape、常用创建函数（Day 1–14 每天都要用）

### 8.1 广播 Broadcasting（Day 3 正式讲，但天天在用）
二级里你只对"形状完全一样"的列表做运算。NumPy 允许**形状不同也能算**，规则叫"广播"：从最后一个维度往前比，维度大小相等、或其中一个是 1、或其中一个不存在，就能自动扩展对齐。教程 Day 3 的例子：

```python
M = np.ones((3, 4)) * 10          # 3行4列全 10
v_col = np.array([1, 2, 3]).reshape(3, 1)   # 形状 (3,1)
print(M + v_col)                  # 每列都加上 [1,2,3]
```

`M` 是 `(3,4)`，`v_col` 是 `(3,1)`。从末维看：4 vs 1 → 1 可以"复制"成 4；前面 3 vs 3 相等。于是 `v_col` 被复制成 `(3,4)` 再相加。
反例：`np.array([1,2,3])`（形状 `(3,)`）加 `M`(3,4) 会报错，因为末维 3≠4 且都不是 1。
**二级迁移**：把广播想成"自动把短的向量复制成能对齐的矩阵"，不用你写 `for` 循环。

### 8.2 reshape / flatten：给数组"换形状"
- `reshape(行, 列)`：把数据重新排列成新形状，**元素总数不能变**。例：`np.array([1,2,3,4]).reshape(2,2)` 变成 2×2。
- `flatten()` / `ravel()`：把任意多维数组拉成一维。神经网络常把图片 `(28,28)` 拉成 `(784,)` 再喂进去。
教程 Day 67 用 `reshape(-1, 1)` 把一维数据变成"列向量"：`x_np.reshape(-1,1)` 里 `-1` 表示"这一维自动算出来"。

### 8.3 常用创建函数（Day 1 就出现）

| 写法 | 含义 |
|---|---|
| `np.zeros((行,列))` | 全 0 |
| `np.ones((行,列))` | 全 1 |
| `np.full((行,列), 值)` | 填满指定值 |
| `np.arange(n)` | 类似 `range`，但返回数组 |
| `np.random.rand(a,b)` | 均匀随机 0~1（Day 2） |
| `np.random.randn(a,b)` | 标准正态随机（Day 5 已教：均值 0、标准差 1） |

> 小提醒：`rand`（均匀，0~1）和 `randn`（正态，可正可负）不一样；教程 Day 6 用 `rand` 加噪声、Day 5/9 用 `randn` 初始化权重，别混。

---

## 9. 激活函数与概率输出（Day 21–23、62）

### 9.1 为什么需要"激活函数"
没有激活函数的网络，不管叠多少层，本质都只是"矩阵乘 + 加"，表达能力有限。激活函数给网络**非线性**。教程 Day 23 给了三个最常用的，都用你二级会写的 `def` 就能实现：

```python
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def relu(x):    return np.maximum(0, x)
def tanh(x):    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
```

- **ReLU**：负数变 0、正数不变，最简单也最常用。
- **sigmoid**：压到 0~1，适合二分类输出。
- **tanh**：压到 -1~1。

### 9.2 softmax + 交叉熵：多分类的"概率输出 + 损失"
要分 10 类（如 MNIST 数字 0–9）时，最后一层用 **softmax** 把输出变成"和为 1 的概率分布"；衡量预测好坏用**交叉熵**：

```python
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))   # 减最大值防溢出
    return e / e.sum(axis=1, keepdims=True)
```

`z.max(axis=1, keepdims=True)` 那行是数值技巧（防止 `np.exp` 算太大溢出），你只要知道"它让计算稳定"即可，不必手推。
**二级迁移**：softmax 本质就是"把一组数变成比例"，交叉熵就是"预测概率和真实标签差多远"。教程会直接给公式，你照抄、会用就行。

---

## 10. PyTorch 入门四件套（Day 64 起，重点难点）

教程第 10 周起全面用 PyTorch。它把前面你手写的东西（矩阵乘、梯度、类）都封装好了。四个核心概念一次讲清。

### 10.1 Tensor：NumPy 数组的"GPU 版"
`torch.tensor(...)` 和 `np.array(...)` 几乎一样，只是多了"能在显卡上算"：

```python
import torch, numpy as np
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # 从列表建
print(t, t.shape, t.dtype)
print(t + 1, t * 2, t @ t)                    # 加、逐元素乘、矩阵乘，和 NumPy 一样
a = np.array([1.0, 2.0, 3.0])
t2 = torch.from_numpy(a)                      # NumPy -> Tensor
back = t2.numpy()                             # Tensor -> NumPy
```

**你只记**：Tensor 用法 ≈ ndarray；`from_numpy` / `.numpy()` 做互相转换。

### 10.2 autograd：不用手算梯度了
前面 Day 28 你手写 `backward()` 求梯度；PyTorch 的 `loss.backward()` 自动帮你算好，存在每个参数的 `.grad` 里：

```python
loss.backward()      # 自动算梯度
opt.step()           # 优化器按梯度更新参数
```

**二级迁移**：把 `backward()` 想成"一键算出所有导数"，你不用再写链式法则。

### 10.3 nn.Module / nn.Sequential：用现成的"层"
你 Day 9 手写过 `Linear` 类；PyTorch 直接给了 `nn.Linear`，还把 `forward` 写好了，你用 `nn.Sequential` 把层串起来即可：

```python
import torch, torch.nn as nn
model = nn.Sequential(
    nn.Linear(2, 2),    # 输入 2 → 隐藏 2
    nn.Tanh(),          # 激活
    nn.Linear(2, 1),    # 隐藏 2 → 输出 1
    nn.Sigmoid())
print(model(X))        # 像函数一样调用，内部自动 forward
```

注意 `model(X)` 能直接调用，正是因为你 Day 10 学过的 `__call__` 机制——PyTorch 的层都用了它。

### 10.4 optim：更新权重的"自动挡"

```python
opt = torch.optim.SGD(model.parameters(), lr=0.01)   # 随机梯度下降
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()    # 清空上一步梯度
    loss.backward()    # 算梯度
    opt.step()         # 更新权重
```

`model.parameters()` 是"这个模型里所有要学的参数"；`lr` 是学习率（Day 20 学过）。三步走 `zero_grad → backward → step` 是 PyTorch 训练的固定套路，背下来。

---

## 11. 数据流水线：DataLoader 与模型存取（Day 69）

真实数据很多，不能一次全喂。PyTorch 用 `Dataset`（装数据）+ `DataLoader`（分批喂）解决：

```python
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(X, y)                       # 把特征和标签捆一起
loader = DataLoader(dataset, batch_size=2, shuffle=True)   # 每批 2 个、打乱
for xb, yb in loader:                               # 自动一批批给你
    ...训练...
torch.save(model.state_dict(), "model.pt")          # 保存权重
m2.load_state_dict(torch.load("model.pt"))          # 加载回来
```

**二级迁移**：`for xb, yb in loader` 就是个帮你"分批取数据"的循环，和 `for i in range(n)` 思路一样，只是每次拿到一小批。

---

## 12. CNN 概念扫盲（Day 70+，点到为止）

卷积神经网络（CNN）用来处理图片。两个最该知道的层：
- `nn.Conv2d(通道_in, 通道_out, kernel_size=3, padding=1)`：用一个"小窗口"在图上滑动、提取局部特征（边缘、纹理…）。`padding=1` 让输出尺寸不变。
- `nn.MaxPool2d(2)`：把 2×2 区域压成 1 个最大值，缩小图片、保留重要信息。
- `nn.Dropout(0.2)`：训练时随机"关掉"20% 的神经元，防止死记硬背（过拟合）。

图片在 PyTorch 里的形状是 `(批量, 通道, 高, 宽)`，例如单张灰度图 `(1, 1, 28, 28)`。

> CNN 不用一次搞懂。教程 Day 70 起会带你从 2-4-1 小网络一路到 GTSRB 图片分类，遇到不懂的层回看本节省力。

---

## 13. 毕业项目衔接：GTSRB 图片分类用到的"新东西"（Day 70–84）

前面第 10–12 节是 PyTorch 通用基础。毕业项目（GTSRB 交通标志分类）会多用到 4 个二级没见过的工具，这里一次性讲清，免得你读到时卡住。

### 13.1 `torchvision.transforms`：给图片做"预处理流水线"
真实图片大小不一，网络要求统一尺寸、统一数值范围。教程用 `transforms.Compose` 把多个步骤串成一条流水线：

```python
from torchvision import transforms
tfm = transforms.Compose([
    transforms.Resize((32, 32)),      # 统一缩放到 32×32
    transforms.ToTensor(),            # 图片 -> Tensor，形状 (3,32,32)
    transforms.Normalize([0.5]*3, [0.5]*3)   # 把像素从 0~1 变成约 -1~1
])
```
`Compose([...])` 就是"按顺序执行列表里每一步"——和你二级写 `for step in [a, b, c]: step()` 一个意思。`ToTensor()` 这一步把 PIL 图片（用整数 0~255 表示颜色）转成 PyTorch 能算的 Tensor（小数 0~1）。

### 13.2 `PIL.Image`：读一张本地图片
```python
from PIL import Image
img = Image.open("cat.jpg").convert("RGB")   # 打开图片，转成 3 通道彩色
x = tfm(img).unsqueeze(0)                    # 预处理，并加上"批量"那一维
```
`.convert("RGB")` 是保险：有些图是黑白的或带透明通道，统一转成 3 通道，避免后面形状对不上。`unsqueeze(0)` 在第 10.1 讲过——网络一次吃"一批"，所以要在最前面补一个尺寸为 1 的维度，变成 `(1, 3, 32, 32)`。

### 13.3 `os.listdir`：自动遍历文件夹里的图片
GTSRB 数据按"文件夹=类别"组织（每个文件夹里是该类别的所有图片）。用 `os.listdir` 把文件名读出来再循环处理：

```python
import os
root = "data/gtsrb/train"
for class_dir in os.listdir(root):                 # 遍历每个类别文件夹
    class_path = os.path.join(root, class_dir)     # 拼出完整路径
    for f in os.listdir(class_path):               # 遍历该类别下每张图
        img_path = os.path.join(class_path, f)
        ...处理 img_path...
```
`os.path.join(a, b)` 是"拼接路径"，比手写 `a + "/" + b` 更安全（不同系统分隔符不同）。

### 13.4 `StepLR`：让学习率随训练自动变小
训练后期如果学习率太大，参数会在最优点附近来回抖动、收敛不了。教程用 `StepLR` 每隔几轮把学习率乘个系数（如 0.5）衰减：

```python
import torch.optim as optim
scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)   # 每 2 轮 lr 减半
for epoch in range(10):
    ...训练一步（opt.step()）...
    scheduler.step()        # 每个 epoch 末尾调用，自动更新学习率
```
**二级迁移**：把它当成"给优化器挂了个定时器"，到时间就自动把 `lr` 调小，你不用手动改。

> 这 4 个工具只在毕业项目（Day 70+）出现，平时练习用不到，读到那几天回看本节省力。

---

✅ **这份补充读完后**：再打开 `Python神经网络手把手教程.md` 从 Day 1 顺着练。前面 Day 8/9/11 回看第 1–3 节；Day 23/62 回看第 9 节；进入 PyTorch（Day 64+）回看第 10–12 节；做到 GTSRB 毕业项目（Day 70+）回看第 13 节。你不需要先系统学完面向对象或 PyTorch——边用边懂最快。
