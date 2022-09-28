<p align="center"><img width="40%" src="logo/pytorch_logo_2018.svg" /></p>

--------------------------------------------------------------------------------

# PyTorch入门教程
PyTorch 是一个基于 Python 的科学计算包，是我们用来构建神经网络的工具。

## Tensor张量

Tensors 类似于 NumPy 的 ndarrays ，同时Tensors 可以使用 GPU 进行计算，可以理解以为一种保存矩阵信息的数据形式。
### 导入我们所需要的库。

```
from __future__ import print_function
import torch
```

### 构造一个5x3的空矩阵：
```
x = torch.empty(5, 3)
print(x)
```
输出：
```
tensor(1.00000e-04 *
       [[-0.0000,  0.0000,  1.5135],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000]])
```

### 构造一个随机初始化的矩阵：
```
x = torch.rand(5, 3)
print(x)
```
输出：
```
tensor([[ 0.6291,  0.2581,  0.6414],
        [ 0.9739,  0.8243,  0.2276],
        [ 0.4184,  0.1815,  0.5131],
        [ 0.5533,  0.5440,  0.0718],
        [ 0.2908,  0.1850,  0.5297]])
```

### 构造一个矩阵全为 0，而且数据类型是 long.
```
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```
输出：
```
tensor([[ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0]])
```
### 创建一个 tensor 基于已经存在的 tensor。
```
x = x.new_ones(5, 3, dtype=torch.double)  
# 随机化
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
```
输出:
```
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]], dtype=torch.float64)
tensor([[-0.2183,  0.4477, -0.4053],
        [ 1.7353, -0.0048,  1.2177],
        [-1.1111,  1.0878,  0.9722],
        [-0.7771, -0.2174,  0.0412],
        [-2.1750,  1.3609, -0.3322]])
```  
### 获取它的维度信息:
```
print(x.size())
```
输出:
```
torch.Size([5, 3])
```

## 对Tensor进行操作

### 加法: 
方式一：
```
y = torch.rand(5, 3)
print(x + y)
```

方式二：
```
print(torch.add(x, y))
```
方法三
```
# adds x to y
y.add_(x)
print(y)
```
Output:
```
tensor([[-0.1859,  1.3970,  0.5236],
        [ 2.3854,  0.0707,  2.1970],
        [-0.3587,  1.2359,  1.8951],
        [-0.1189, -0.1376,  0.4647],
        [-1.8968,  2.0164,  0.1092]])
```

### 我们可以使用标准的**NumPy**类似的索引操作
```
print(x[:, 1])
```
Output:
```
tensor([ 0.4477, -0.0048,  1.0878, -0.2174,  1.3609])
```
### 改变Tensor的大小
```
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  
print(x.size(), y.size(), z.size())
```
Output:
```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

## PyTorch自动微分
autograd 包是 PyTorch 的核心。autograd 为 Tensors 上的所有操作提供自动微分，这意味着我们可以实现后向传播，并且每次迭代都可以不同。我们从 tensor 和 gradients 来举一些例子。我们可以调用 Tensor.backward()来计算导数。

### 计算导数
```
# 创建tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b
```
Output:
```
(tensor(3.), tensor(4., requires_grad=True), tensor(5., requires_grad=True))
```

我们在上面创建了三个张量：x、w 和 b，都是数字。 w 和 b 有一个额外的参数 requires_grad 设置为 True。 
我们通过组合上面的张量来创建一个新的张量y。

```
y = w * x + b
y
```
Output:
```
tensor(17., grad_fn=<AddBackward0>)
```
y 是一个值为 3 * 4 + 5 = 17 的张量。PyTorch 的独特之处在于我们可以自动计算 y w.r.t 的导数。 将 requires_grad 设置为 True 的张量，即 w 和 b。 PyTorch 的这个特性称为 autograd。

要计算导数，我们可以在结果 y 上调用 .backward 方法。

```
# 计算导数
y.backward()
```
y 关于输入张量的导数存储在相应张量的 .grad 属性中，如下所示
```
# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
```

Output:

```
dy/dx: None
dy/dw: tensor(3.)
dy/db: tensor(1.)
```
dy/dw 与 x 具有相同的值，即为3，而 dy/db为1。请注意，x.grad 为 None，因为 x 没有将 requires_grad 设置为 True。

补充： w.grad 中的“grad”是梯度的缩写，是导数的另一种说法。 梯度一词主要用于处理向量和矩阵。

### 4. 与Numpy的互操作性
Numpy是一个流行的开源库，用于Python中的数学和科学计算。我们学习这部分是因为，有一些数据我们需要通过Numpy和Pandas库进行处理，处理过后的数据再转化为Tensor进行处理。

##### 简单介绍Numpy
<details>
<summary> Numpy操作 </summary>
<pre><code>
Numpy的操作和上面的讲述的Tensor的操作大致相同。
* 创建numpy数组
```
import numpy as np
kanto = np.array([73, 67, 43])
kanto
```
Output:
```
array([73, 67, 43])
```
Numpy arrays的格式是ndarray.
```
type(kanto)
```
Output:
```
numpy.ndarray
```
* numpy array相乘

```
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr1 * arr2
```
Output:
```
array([ 4, 10, 18])
```


</code></pre>
</details>

##### Numpy和Tensor之间的相互转换

* 创造一个Numpy array
```
import numpy as np

x = np.array([[1, 2], [3, 4.]])
x
```
Output:
```
array([[1., 2.],
       [3., 4.]])
```
* 将numpy array 转化为tensor

```
y = torch.from_numpy(x)
y
```
Output:
```
tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64)
```
* 将tensor转化为numpy array
```
z = y.numpy()
z
```
Output:
```
array([[1., 2.],
       [3., 4.]])
```

