# 固定随机种子为 0，生成 10000 个「标准正态分布」样本，打印它们的均值和标准差，它们应接近什么数？
# 再次固定种子为 0 生成同样的 10000 个，验证两次生成的数组完全一致。
# 生成 5 个「0~1 均匀分布」的随机数并打印，观察它和正态分布在取值范围上的差异。
# 不设种子，连续生成两组各 3 个标准正态分布随机数，两次结果一样吗？打印说明。
import numpy as np
np.random.seed(0)
a=np.random.randn(10000)
print("均值为：{:.3f}，标准差为：{:.3f}".format(a.mean(), a.std()))
np.random.seed(0)
b=np.random.randn(10000)
print(np.array_equal(a,b))
print(np.random.rand(5))
print(np.random.randn(5))

