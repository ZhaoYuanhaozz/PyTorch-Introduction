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
# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b
```
Output:
```
(tensor(3.), tensor(4., requires_grad=True), tensor(5., requires_grad=True))
```

We've created three tensors: x, w, and b, all numbers. w and b have an additional parameter requires_grad set to True. We'll see what it does in just a moment.

Let's create a new tensor y by combining these tensors.

```
# Arithmetic operations
y = w * x + b
y
```
Output:
```
tensor(17., grad_fn=<AddBackward0>)
```
As expected, y is a tensor with the value 3 * 4 + 5 = 17. What makes PyTorch unique is that we can automatically compute the derivative of y w.r.t. the tensors that have requires_grad set to True i.e. w and b. This feature of PyTorch is called autograd (automatic gradients).

To compute the derivatives, we can invoke the .backward method on our result y.

```
# Compute derivatives
y.backward()
```
The derivatives of y with respect to the input tensors are stored in the .grad property of the respective tensors.
```
# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
```

As expected, dy/dw has the same value as x, i.e., 3, and dy/db has the value 1. Note that x.grad is None because x doesn't have requires_grad set to True.

The "grad" in w.grad is short for gradient, which is another term for derivative. The term gradient is primarily used while dealing with vectors and matrices.

