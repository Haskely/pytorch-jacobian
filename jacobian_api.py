import torch


def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=True) -> torch.Tensor:
    """基于 torch.autograd.grad 函数的更清晰明了的 API，功能是计算一个雅可比矩阵。

    Args:
        y (torch.Tensor): 函数输出向量
        x (torch.Tensor): 函数输入向量
        need_higher_grad (bool, optional): 是否需要计算高阶导数，如果确定不需要可以设置为 False 以节约资源. 默认为 True.

    Returns:
        torch.Tensor: 计算好的“雅可比矩阵”。注意！输出的“雅可比矩阵”形状为 y.shape + x.shape。例如：y 是 n 个元素的张量，y.shape = [n]；x 是 m 个元素的张量，x.shape = [m]，则输出的雅可比矩阵形状为 n x m，符合常见的数学定义。
        但是若 y 是 1 x n 的张量，y.shape = [1,n]；x 是 1 x m 的张量，x.shape = [1,m]，则输出的雅可比矩阵形状为1 x n x 1 x m，如果嫌弃多余的维度可以自行使用 Jac.squeeze() 一步到位。
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
