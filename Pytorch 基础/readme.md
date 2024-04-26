# 数据类型

## 数值类型
- 标量，也称 Scalar，是一个只有大小，没有方向的量，比如 1.8、e、10 等
- 向量，也称 Vector，是一个有大小也有方向的量，比如 (1,2,3,4) 等
- 矩阵，也称 Matrix，是多个向量合并在一起得到的量，比如[(1,2,3),(4,5,6)]等

![](./imgs/data_type.png)


- 张量，也称 Tensor,所有维度数 dim > 2 的数组统称为张量。张量的每个维度也叫作轴(Axis)，通常某个维度代表了特定的物理含义

在 PyTorch 中，为了表达方便，一般把标量、向量、矩阵也统称为张量，不作区分，需要根据张量的维度数或形状来自行判断。

```
import torch

t1 = torch.tensor(1.2)
print(t1)

t2 = torch.tensor([1, 2])
print(t2)

t3 = torch.tensor([[0,1],[2,3]])
print(t3)

t4 = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(t4)
```
输出:
```
tensor(1.2000)
tensor([1, 2])
tensor([[0, 1],
        [2, 3]])
tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
```

## 布尔类型
PyTorch 还支持布尔类型(Boolean，简称 bool)的张量。布尔类型的张量只需要传入 Python 语言的布尔类型数据，转换成 PyTorch 内部布尔型即可。

```
a = torch.tensor(True)
print(a)

b = torch.tensor([True,False, False])
print(b)
```
输出:
```
tensor(True)
tensor([ True, False, False])
```

# 数值精度
对于数值类型的张量，可以保存为不同字节长度的精度，较为常用的几种格式：

![](./imgs/data_types.png)

在创建张量时，可以通过 dtype 参数指定张量的保存精度。

精度过低时，可能导致数据溢出:
```
t1 = torch.tensor(123456789, dtype=torch.int16)
t2 = torch.tensor(123456789, dtype=torch.int32)
print(t1)
print(t2)
```
输出:
```
tensor(-13035, dtype=torch.int16)
tensor(123456789, dtype=torch.int32)
```

对于浮点数，高精度数据更加准确：
```
import numpy as np
t1 = torch.tensor(np.pi, dtype=torch.float32)
t2 = torch.tensor(np.pi, dtype=torch.float64)
print(f"{t1:.20f}")
print(f"{t2:.20f}")
```
输出:
```
3.14159274101257324219
3.14159265358979311600
```
## 类型转换
可以使用 `tensor.type` 转换数据类型
```
x = torch.tensor(np.pi, dtype=torch.float64)
if x.dtype != torch.float32:
    x = x.type(torch.float32)
print(x.dtype)
```
输出:
```
torch.float32
```

布尔类型与整型之间相互转换也是合法的:
```
t = torch.tensor([1, 0, -1])
t = t.type(torch.bool)
print(t)
print(t.dtype)
```
输出:
```
tensor([ True, False,  True])
torch.bool
```

对于常见张量类型的转换，可以通过 `long()、int()、float()、double()` 等函数便捷地转换到 `torch.int64、torch.int32、torch.float32、torch.float64` 精度:
```
t = torch.tensor(np.pi, dtype=torch.float64)
t = t.long().int()
print(t.dtype)
print(t)
```
输出:
```
torch.int32
tensor(3, dtype=torch.int32)
```

# 张量的创建

## 直接创建
```
torch.tensor(data, dtype=None, device=None,requires_grad=False)
```
- data: 传入的数据，可以是 `list、 tuple、numpy.ndarray、scalar` 等多种类型进行数据
- dtype: 数据类型
- device: 指定了数据要返回到的设备
- requires_grad: 说明当前量是否需要在计算中保留对应的梯度信息

从 `numpy.ndarray` 创建张量，`numpy.ndarray` 浮点数的默认数据类型是 `float64`:
```
array = np.array([1.,2.,3.])
t = torch.tensor(array)
print(t)

array[1] = 0
print(t)
print(array)
```
输出:
```
tensor([1., 2., 3.], dtype=torch.float64)
tensor([1., 2., 3.], dtype=torch.float64)
[1. 0. 3.]
```

## 创建全 0 或全 1 张量
通过 `torch.zeros()` 和 `torch.ones()` 即可创建任意形状，且内容全 0 或全 1 的张量。基本格式为:
```
torch.zeros(*size, dtype=None...)
torch.ones(*size, dtype=None...)
```

```
t1 = torch.zeros([3, 4])
print(t1)
t2 = torch.ones([2,3,2])
print(t2)
```
输出:
```
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
tensor([[[1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.]]])
```

也可以创建标量:
```
t1 = torch.zeros([])
t2 = torch.ones([])
print(t1)
print(t2)
print(type(t1))
print(t2.dtype)
```
输出:
```
tensor(0.)
tensor(1.)
<class 'torch.Tensor'>
torch.float32
```

通过 `torch.zeros_like, torch.ones_like` 可以方便地新建与某个张量形状一致，且内容为全 0 或全 1 的张量:
```
x = torch.tensor([[1.,2.],[3.,4.]])
t1 = torch.ones_like(x)
t2 = torch.zeros_like(x)
print(t1)
print(t2)
```
输出:
```
tensor([[1., 1.],
        [1., 1.]])
tensor([[0., 0.],
        [0., 0.]])
```

通过 `torch.eye` 可以创建单位矩阵，单位矩阵是指主对角线上的元素都为 1 的矩阵，基本格式为:
```
torch.eye(n, m=None...)
```
```
t1 = torch.eye(4)
print(t1)

t2 = torch.eye(4,3)
print(t2)
```
输出:
```
tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 0.]])
```

## 创建未初始化和指定值初始化的张量
有时需要创建未初始化的张量，可以使用 `torch.full` 函数实现:
```
t1 = torch.empty([2,3])
print(t1)
```

有时也需要全部初始化为某个自定义数值的张量，可以使用 `torch.full` 函数实现:
```
t1 = torch.full([], -1)
t2= torch.full([3,2],9)
print(t1)
print(t2)
```
输出:
```
tensor(-1)
tensor([[9, 9],
        [9, 9],
        [9, 9]])
```

## 创建已知分布的张量

`torch.randn` 可以创建均值为 0，标准差为 1 的正态分布的张量:
```
t1 = torch.randn([3,4,2])
print(t1)
```

`normal_` 可以创建均值为 mean，标注差为 stddev 的任意正态分布的张量:
```
t1 = torch.empty([2,3])
t1 = t1.normal_(mean=0.5, std=2)
print(t1)
```
`function_` 表示原地操作。

`torch.rand` 创建采样自 `[0,1)` 区间的均匀分布的张量：
``` 
t1 = torch.rand([3,3])
print(t1)
```

`torch.randint` 可以创建 `[low,high)` 均匀生成的随机整数:
```
t1 = torch.randint(1,20,[3,4])
print(t1)
```

`uniform_` 可以创建采样自 `[minval, maxval)` 区间分布的张量:
```
t1 = torch.empty([2,1])
t1 = t1.uniform_(3, 9)
print(t1)
```

更多分布的例子可以从 `torch.distributions` 子库中获取:
```
from torch.distributions import Bernoulli

dist = Bernoulli(torch.tensor(0.5))
print(dist.sample(torch.Size([8])))
```

## 创建序列

`torch.arange(start, end, step)` 可以创建 `[start, end)` 之间，步长为 `step` 整型序列，不包含 `end` 本身。
```
t1 = torch.arange(10)
print(t1)

t2 = torch.arange(2,8,3)
print(t2)

t3 = torch.arange(10,5,-1)
print(t3)
```
输出:
```
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([2, 5])
tensor([10,  9,  8,  7,  6])
```

`torch.range(start,end,step)` 可以创建 `[start, limit]`，步长为 `step` 的序，但包含 `end` 本身:
```
t1 = torch.range(1,5)
print(t1)

t2 = torch.range(2,8,3)
print(t2)

t3 = torch.range(10,5,-1)
print(t3)
```
输出:
```
tensor([1., 2., 3., 4., 5.])
tensor([2., 5., 8.])
tensor([10.,  9.,  8.,  7.,  6.,  5.])
```

# 张量的转换
## 张量与 python 数值类型之间的转换
可以将 python 的数值类型转换成 torch 的张量类型:
```
x = 1
t1 = torch.tensor(x)
print(t1)
print(t1.dtype)
```
输出:
```
tensor(1)
torch.int64
```

`item` 函数可以将张量转换成 python 的数值类型:
```
t1 = torch.tensor(1.4)
x = t1.item()
print(x)
print(type(x))
```
输出:
```
1.399999976158142
<class 'float'>
```

## 张量与 numpy 之间的转换
除了上文提打的使用 `numpy.nddarray` 直接创建张量外，也可以使用 `from_numpy` 函数创建张量，但是此时两者共享底层存储:
```
array = np.array([1,2,3])
t = torch.from_numpy(array)
print(t)
array[1] = 0
print(t)
```
输出:
```
tensor([1, 2, 3])
tensor([1, 0, 3])
```

从 `torch.tensor` 转换成 `numpy.ndarray` 可以调用 `numpy`，二者仍然共享底层的数据存储:
```
t = torch.tensor([1.,2.,3.])
array = t.numpy()
print(type(array))
array[0] = 8
print(t)
```
输出:
```
<class 'numpy.ndarray'>
tensor([8., 2., 3.])
```

## 张量与 python 的 list 之间的转换
可以直接从 python 的 list 类型创建张量类型：
```
lst = [1.1, 2.2, 3.3]
t1 = torch.tensor(lst)
print(t1)
print(t1.dtype)
```

张量转换成 python 的list 之前，需要先转换成 `numpy.ndarray` 类型，再使用 `tolist` 转换:
```
t1 = torch.tensor([1.,2.,3.])
lst = t1.numpy().tolist()
print(type(lst))
print(lst)
```
输出:
```
<class 'list'>
[1.0, 2.0, 3.0]
```


## CPU 和 GPU 张量转换
```
t1 = torch.tensor([1.,2.])
print(t1.device)
t1 = t1.cuda()
print(t1.device)
t1 = t1.cpu()
print(t1.device)
```
输出:
```
cpu
cuda:0
cpu
```

# 索引和切片

## 索引
张量支持基本的 `[𝑖][𝑗]` 标准索引方式，也支持通过逗号分隔索引号 `[i, j]` 的索引方式。

考虑输入`𝑿` 为 4 张 `32 × 32` 大小的彩色图片，`shape` 为 `[4,32,32,3]`:

```
X = torch.randint(0, 255, (4, 32, 32, 3))
```

读取第1张图片:
```
img1 = X[0]
print(img1.shape)
```
输出:
```
print(img1.shape)
```

读取第1张图片，第2行:
```
img1_2 = X[0, 1]
# 等价于 img1_2 = X[0][1]
print(img1_2.shape)
```
输出:
```
torch.Size([32, 3])
```

读取第1张图片，第2行，第3列:
```
img1_2_3 = X[0, 1, 2]
# 等价于 img1_2 = X[0][1][2]
print(img1_2_3.shape)
```
输出:
```
torch.Size([3])
```

读取第1张图片，第2行，第3列, G 通道:
```
img1_2_3_G = X[0, 1, 2, 1]
print(img1_2_3_G)
```

## 切片

通过 `start: end: step` 切片方式可以方便地提取一段数据，其中 `start` 为开始读取位置的索引，`end` 为结束读取位置的索引(不包含 `end` 位)，`step` 为采样步长。

```
X = torch.randint(0, 255, (4, 32, 32, 3))
# 选取前两张图片
a = X[1:3]
print(a.shape)

# 选取第一张图片
a = X[0, ::]
print(a.shape)

# 间隔采样
a = X[:,::2,::2,:]
print(a.shape)
```
输出:
```
torch.Size([2, 32, 32, 3])
torch.Size([32, 32, 3])
torch.Size([4, 16, 16, 3])
```

`start: end: step` 切片的简写方式:
- 其中从第一个元素读取时 `start` 可以省略，即 `start=0` 是可以省略
- 取到最后一个元素时 `end` 可以省略
- 步长为 1 时 `step` 可以省略
- `::` 表示读取所有元素
- `:` 表示读取所有元素

为了避免出现像 `[: , : , : ,1]` 这样过多冒号的情况，可以使用 `⋯` 符号表示取多个维度上所有的数据，其中维度的数量需根据规则自动推断：当切片方式出现 `⋯` 符号时，`⋯` 符号左边的维度将自动对齐到最左边，`⋯` 符号右边的维度将自动对齐到最右边，此时系统再自动推断 `⋯` 符号代表的维度数量。

![](./imgs/slice.png)

```
X = torch.randint(0, 255, (4, 32, 32, 3))

# 读取前两张图片的 G、B 通道
a = X[0:2,...,1:]
print(a.shape)

# 读取最后两张图片
a = X[:2,...]
print(a.shape)

# 读取R、G通道数据
a = X[...,:2]
print(a.shape)
```
输出:
```
torch.Size([2, 32, 32, 2])
torch.Size([2, 32, 32, 3])
torch.Size([4, 32, 32, 2])
```

# 张量的常用操作
## 获取张量形状
可以使用 `shape` 或 `size()` 来获取。两者的不同之处在于，`shape` 是 `torch.tensor` 的一个属性，而 `size()` 则是一个 `torch.tensor` 拥有的方法。
```
a = torch.zeros([2, 3, 4])
print(a.shape)
print(a.size())
```
输出:
```
torch.Size([2, 3, 4])
torch.Size([2, 3, 4])
```

可以使用 `numel()` 函数直接统计元素数量:
```
a = torch.zeros([2, 3, 4])
print(a.numel())
```
输出：
```
24
```