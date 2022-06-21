# Pytorch autograd.grad 函数研究

本 notebook 将对 Pytorch 库中的 **autograd.grad** 函数行为进行详细探究。此文撰写时（2022.06.15） Pytorch 版本为 1.11.0

首先查看其[官方文档](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)，翻译如下：

## TORCH.AUTOGRAD.GRAD
```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, is_grads_batched=False)
``` 
[SOURCE 源代码](https://pytorch.org/docs/stable/_modules/torch/autograd.html#grad)

> Computes and returns the sum of gradients of outputs with respect to the inputs.

计算并返回 输出相对于输入的 梯度的 求和（完全搞不清楚这在说什么）

> grad_outputs should be a sequence of length matching output containing the “vector” in vector-Jacobian product, usually the pre-computed gradients w.r.t. each of the outputs. If an output doesn’t require_grad, then the gradient can be None).

参数 “grad_outputs” 应该是一个序列，长度对应于输出，每个元素是 “向量雅可比积（vector-Jacobian product）” 中的 “向量”。此 “向量”序列 通常代表着先前计算的 关于每个输出的 梯度序列。如果一个输出没有 require_grad 标记，那么其梯度可以是 None。（同样也没看明白）

NOTE 注意 1
> If you run any forward ops, create grad_outputs, and/or call grad in a user-specified CUDA stream context, see Stream semantics of backward passes.
> 如果在用户指定的CUDA流上下文中运行任何正向运算、创建grad_outputs和/或调用grad，请参阅向后传递的流语义。（完全不知所云）

NOTE 注意 2
> only_inputs argument is deprecated and is ignored now (defaults to True). To accumulate gradient for other parts of the graph, please use torch.autograd.backward.
> 参数 “only_inputs” 已弃用，现在将被忽略(默认为True)。要累加计算图中其他部分的梯度，请使用 “torch.autograd.backward”。

Parameters 参数

> outputs (sequence of Tensor) – outputs of the differentiated function.

outputs (Tensor 的序列) – 可微函数的输出序列。

> inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).

inputs (Tensor 的序列) – 应该被计算梯度的输入序列，这些梯度将成为函数的返回值（这些梯度不会被累加到 .grad 中去）

> grad_outputs (sequence of Tensor) – The “vector” in the vector-Jacobian product. Usually gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None.

grad_outputs (Tensor 的序列) – “向量雅可比积（vector-Jacobian product）” 中的 “向量”。通常代表着先前计算的 关于每个输出的 梯度序列。对于标量输出或者不需要计算梯度的输出，“向量”可以为 None。 若对于所有的输出其对应的“向量”都可以是 None，那么该参数可以直接忽略，不指定. 默认值：None。

> retain_graph (bool, optional) – If False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.

retain_graph (布尔值, 可选) – 如果为 False，则用于计算梯度的计算图所占内存将被释放。请注意，在几乎所有情况下，都不需要将此选项设置为 True，此时可以使该函数效率更高（省内存）。默认为参数 create_graph 的值。

> create_graph (bool, optional) – If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. Default: False.

create_graph (布尔值, 可选) – 如果为 True，则将同时构造此导数的计算图，从而允许计算更高阶导数。默认值：False。

> allow_unused (bool, optional) – If False, specifying inputs that were not used when computing outputs (and therefore their grad is always zero) is an error. Defaults to False.

allow_unused (布尔值, 可选) – 如果为 False，则在计算输出对应某个输入的梯度时，若该输入未在该输出的计算图中(因此它们的梯度始终为零)，就会报错。默认为 False。

> is_grads_batched (bool, optional) – If True, the first dimension of each tensor in grad_outputs will be interpreted as the batch dimension. Instead of computing a single vector-Jacobian product, we compute a batch of vector-Jacobian products for each “vector” in the batch. We use the vmap prototype feature as the backend to vectorize calls to the autograd engine so that this computation can be performed in a single call. This should lead to performance improvements when compared to manually looping and performing backward multiple times. Note that due to this feature being experimental, there may be performance cliffs. Please use torch._C._debug_only_display_vmap_fallback_warnings(True) to show any performance warnings and file an issue on github if warnings exist for your use case. Defaults to False.

is_grads_batched (bool, optional) – 如果为 True，则 grad_outputs 中每个张量的第一维将被解释为批次维。我们不是计算单个向量雅可比乘积，而是为批次中的每个“向量”计算一批向量雅可比乘积。我们使用 vmap 原型功能作为后端来向量化对 autograd 引擎的调用，以便可以在单个调用中执行此计算。与手动循环和多次向后执行相比，这应该会带来性能改进。请注意，由于此功能处于实验阶段，因此可能会出现性能骤降。如果您的用例存在警告，请使用 torch._C._debug_only_display_vmap_fallback_warnings(True) 显示任何性能警告，并在github上提交问题。默认为 False。

看完此文档，感觉对函数用法以及每一个参数都做了解释，但是又 **含糊不清**（大佬请退让，以上描述对我来说就是含糊不清）。看来我还是需要进行进一步实验来确定其行为。

## 个人实验分析

> 实验过程乱七八糟，测试了各种各样的情况组合，详细过程见 grad_research.ipynb

综上，基本上摸清了 Pytorch 中 autograd.grad 函数的行为，总结如下：

## 函数调用样例：

```python
import torch

# f1, f2, f3 为三个抽象的向量函数。函数输入一个 1 x m 的向量，输出 1 x n 的向量。
def f1(x:torch.Tensor):
    ...
    return y:torch.Tensor

def f2(x:torch.Tensor):
    ...
    return y:torch.Tensor

def f3(x:torch.Tensor):
    ...
    return y:torch.Tensor

# 上述函数的输入输出形状假设如下：
x1_shape = (1,m1)
x2_shape = (1,m2)
x3_shape = (1,m3)

y1_shape = (1,n1)
y2_shape = (1,n2)
y3_shape = (1,n3)

x1 = torch.rand(size = x1_shape)
x2 = torch.rand(size = x1_shape)
x3 = torch.rand(size = x1_shape)
for x in (x1,x2,x3):
    x.requires_grad_()

y1 = f1(x1)
y2 = f2(x2)
y3 = f3(x3)

v1 = torch.rand(size = y1_shape)
v2 = torch.rand(size = y2_shape)
v3 = torch.rand(size = y3_shape)

(g1,g2) = torch.autograd.grad(
    outputs = (y1,y2,y3), 
    inputs = (x1,x2), 
    grad_outputs=(v1,v2,v3), 
    retain_graph=None, 
    create_graph=False, 
    only_inputs=True, 
    allow_unused=False, 
    is_grads_batched=False
)
""" 该函数行为：
y = v1 @ y1 + v2 @ y2 + v3 @ y3 【@ 代表内积，此时 y 是一个标量了】
g1 = 雅可比矩阵(y,x1) 【形状为 1 x m1】
g2 = 雅可比矩阵(y,x2) 【形状为 1 x m2】
返回 (g1,g2)
"""
```

参考以上样例，得到如下的抽象调用。

## 抽象调用：

```python
(g1,g2,...,gm) = torch.autograd.grad(
    outputs = (y1,y2,...,yn), 
    inputs = (x1,x2,...,xm), 
    grad_outputs=(v1,v2,...vn), 
    retain_graph=None, 
    create_graph=False, 
    only_inputs=True, 
    allow_unused=False, 
    is_grads_batched=False
)
```

文字描述一下，也就是说，首先程序会将 vi 与 yi 两两做内积然后相加，得到合并的标量输出 y，然后对每一个 xj 的每一个分量 xjk（k=1,2,...,s），计算 y 对 xjk 的偏导数，最后把 xjk 组装成 gj；最后 m 个 xj 对应 m 个 gj，合并成一个元组后返回。

### 程序行为：

输入 n 个“输出向量 yi”以及 m 个“输入向量 xj”以及 n 个“权重向量vi”（形状与对应 yi 一致），函数会输出 m 个向量的元组 (g1,g2,...,gm)，细节为 :

$$
\begin{aligned}   \text{输入：}& \vec{y}_j, ~ \vec{x}_i, ~ \vec{v}_j; ~i=1,\dots,m;~ j = 1,\dots,n;~\vec{y}_j,\vec{v}_j\text{形状一致}.\\     \text{令 } y &= \sum_{j=1}^{n} \vec{v}_j \cdot \vec{y}_j, ~ “~ \cdot ~” \text{为向量内积运算}\\ \text{对任意 } i &\in \{ 1,2,\dots,m\},\\ \text{记 } \vec{x}_i &= (x_{i_1}, x_{i_2}, \dots, x_{i_s})， \\ \text{则 } \vec{g}_i &= \left(\frac{\partial y }{\partial \vec{x}_{i_1}}, \cdots, \frac{\partial y}{\partial \vec{x}_{i_s}}\right); \\   \text{输出元组 }& \mathbf{g} = \left(\vec{g}_1, \vec{g}_2, \dots, \vec{g}_m \right) \end{aligned}
$$

文字描述一下，也就是说，首先程序会将 vi 与 yi 两两做内积然后相加，得到合并的标量输出 y，然后对每一个 xj 的每一个分量 xjk（k=1,2,...,s），计算 y 对 xjk 的偏导数，最后把 xjk 组装成 gj；最后 m 个 xj 对应 m 个 gj，合并成一个元组后返回。

### 参数要求： 

任意 xi， 都需要执行 xi.requires_grad_() 
任意 xi，需存在 yj，使得 xi 参与了 yj 的计算，否则需要设置 allow_unused=True 才不会报错，此时输出的对应 gi 为 None。 
任意 yi，对应一个 vi，两者形状需相同。当 yi 为标量时，vi 可以为 None，此时 vi 自动取 1.0；当所有的 yi 均为标量时，grad_outputs 可以取 None（这也是其默认值），此时 grad_outputs 自动取 (1.0,1.0,...1.0) 【n个】。 值得注意的是，实际测试下 grad_outputs 可以输入超过 n 个 vi，但是多出来的会直接忽略。
函数执行一遍后若再次对 yi 求偏导会报错，因为执行一遍后会删除计算图，除非设置 retain_graph=True。 
若之后想求 gi 对 xj 的偏导（也就是高阶导）会报错，因为 gi 没有 连接 xj 的计算图，除非设置 create_graph=True（此时 retain_graph=create_graph=True） 
only_inputs 参数没有效果，已被弃用。
is_grads_batched 默认为 False，若设置为 True 会完全改变程序行为，详解如下：

### is_grads_batched = True 时程序行为：

$$

\begin{aligned}   \text{输入：}& \vec{y}_j, ~ \vec{x}_i, ~ \mathbf{V}_j; ~i=1,\dots,m;~ j = 1,\dots,n;~\vec{y}_j \text{与} \mathbf{V}_j\text{第一维的每个元素形状一致}.\\     \text{记 } \vec{y}_i &= (y_{i_1}, y_{i_2}, \dots, y_{i_p})，\mathbf{V}_i = \begin{pmatrix}     v_{i_{11}} & \cdots & v_{i_{1p}} \\     \vdots & \ddots & \vdots \\     v_{i_{N1}} & \cdots & v_{i_{Np}}   \end{pmatrix} ， \\ \text{令 } \vec{y} &= \sum_{j=1}^{n} \mathbf{V}_j \cdot \vec{y}_j^T, ~ “~ \cdot ~” \text{为矩阵乘法运算}\\ \text{对任意 } i &\in \{ 1,2,\dots,m\},\\ \text{记 } \vec{x}_i &= (x_{i_1}, x_{i_2}, \dots, x_{i_s})， \\ \text{则 } \vec{g}_i &= \nabla_{\vec{x}_i} \vec{y} = \begin{pmatrix}     \frac{\partial y_1 }{\partial \vec{x}_{i_1}} & \cdots & \frac{\partial y_1 }{\partial \vec{x}_{i_s}} \\     \vdots & \ddots & \vdots \\     \frac{\partial y_p }{\partial \vec{x}_{i_1}} & \cdots & \frac{\partial y_p }{\partial \vec{x}_{i_s}}  \end{pmatrix}; \\   \text{输出元组 }& \mathbf{g} = \left(\vec{g}_1, \vec{g}_2, \dots, \vec{g}_m \right) \end{aligned}

$$

## 个人总结

基于以上说明，我自己在使用 torch.autograd.grad() 函数时，首先 outputs 永远只输入一个标量值，同时永远忽略 grad_outputs 参数（也就是让其默认为 1.0）。因为如果有需要自己完全可以在外面做好向量内积相加等操作，为何交给这个行为不写明的函数来做呢。
其次在写通用的求导工具函数时，allow_unused 永远设为 True，然后自行将 None 处理成 0，因为如果某个 xi 没有被使用，其导数数学意义就是 0，让他报错干嘛。
有了上面的总结，实在讨厌原版各种花里胡哨的参数，就让我就来提供一个更清晰的 API 吧，注意下面使用了 is_grads_batched = True 参数。

## 更清晰的 API：

```python
import torch


def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=True) -> torch.Tensor:
    """基于 torch.autograd.grad 函数的更清晰明了的 API，功能是计算一个雅可比矩阵。

    Args:
        y (torch.Tensor): 函数输出向量
        x (torch.Tensor): 函数输入向量
        need_higher_grad (bool, optional): 是否需要计算高阶导数，如果确定不需要可以设置为 False 以节约资源. 默认为 True.

    Returns:
        torch.Tensor: 计算好的“雅可比矩阵”。注意！输出的“雅可比矩阵”形状为 y.shape + x.shape。例如：y 是 n 个元素的张量，y.shape = [n]；x 是 m 个元素的张量，x.shape = [m]，则输出的雅可比矩阵形状为 n x m，符合常见的数学定义。
        但是若 y 是 1 x n 的张量，y.shape = [1,n]；x 是 1 x m 的张量，x.shape = [1,m]，则输出的雅可比矩阵形状为1 x n x 1 x m，如果嫌弃多余的维度可以自行使用 torch.squeeze(Jac) 一步到位。
        这样设计是因为考虑到 y 是 n1 x n2 的张量； 是 m1 x m2 的张量（或者形状更复杂的张量）时，输出 n1 x n2 x m1 x m2 （或对应更复杂形状）更有直观含义，方便用户知道哪一个元素对应的是哪一个偏导。
    """
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y)),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac.resize_(size=(y.shape + x.shape))
    return Jac
```
用这个 API 做一下计算

例一：
```python
def f1(x:torch.Tensor):
    W = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)
    b = torch.tensor([7,8,9],dtype=torch.float)
    return x @ W + b 

x = torch.tensor([0.1,0.2])
x.requires_grad_()
y = f1(x)

J = jacobian(y,x)
# J = tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```

例二：
```python
x = torch.tensor(0.1)
x.requires_grad_()
y = 2 * x + 3
J = jacobian(y,x)
# J = tensor(2.)
```

例三：
```python
def f1(x:torch.Tensor):
    W = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)
    b = torch.tensor([7,8,9],dtype=torch.float)
    return x @ W + b 

x = torch.tensor([[0.1,0.2]]) # 形状：1 x 2
x.requires_grad_()
y = f1(x) # 形状 1 x 3 
J = jacobian(y,x) # 形状 1 x 3 x 1 x 2
# J = tensor([[[[1., 4.]],
         [[2., 5.]],
         [[3., 6.]]]])
J.squeeze_()
# J = tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```

当使用该 API 做网络训练的一环时经常需要用到一次计算一个批次的雅可比矩阵的情况，因此给出如下扩展 API。

### 批量计算 API：

```python
def batched_jacobian(batched_y:torch.Tensor,batched_x:torch.Tensor,need_higher_grad = True) -> torch.Tensor:
    """计算一个批次的雅可比矩阵。
        注意输入的 batched_y 与 batched_x 应该满足一一对应的关系，否则即便正常输出，其数学意义也不明。

    Args:
        batched_y (torch.Tensor): N x y_shape
        batched_x (torch.Tensor): N x x_shape
        need_higher_grad (bool, optional):是否需要计算高阶导数. 默认为 True.

    Returns:
        torch.Tensor: 计算好的一个批次的雅可比矩阵张量，形状为  N x y_shape x x_shape
    """
    sumed_y = batched_y.sum(dim = 0) # y_shape
    J = jacobian(sumed_y,batched_x,need_higher_grad) # y_shape x N x x_shape
    
    dims = list(range(J.dim()))
    dims[0],dims[sumed_y.dim()] = dims[sumed_y.dim()],dims[0]
    J = J.permute(dims = dims) # N x y_shape x x_shape
    return J
```

例：
```python
def f1(x:torch.Tensor):
    W = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float)
    b = torch.tensor([7,8,9],dtype=torch.float)
    return x @ W + b 

batched_x = torch.rand((10,2))
batched_x.requires_grad_()
batched_y = f1(batched_x)
batched_J = batched_jacobian(batched_y,batched_x)
""" batched_J = tensor([[[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]],

        [[1., 4.],
         [2., 5.],
         [3., 6.]]]) """
```

**注意：根据官方文档这并非最高效率实现。 Pytorch 1.11.0 推出了 [functorch](https://pytorch.org/functorch/stable/) beta 版，使用内置的 jacrev 以及 vmap + jacrev 为 pytorch 框架下目前最高效的 “雅可比矩阵 API”  以及 “批量雅可比矩阵 API” 实现。**但是该库目前处于快速迭代期，同时对 windows 平台支持不友好，不如上面的实现稳定性好。
完毕。


完毕。
