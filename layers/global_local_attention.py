from math import sqrt

import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_
from wth.torch.utils import tensor_to


class Sigma(nn.Module):
    """
    论文中的 sigma 函数, 可以传入一个参数阈值, 小于该阈值的都将被设为 负无穷，在 softmax 之后变为 0.
    """

    def __init__(self, t):
        super(Sigma, self).__init__()
        self.t = t

    def forward(self, x):
        filters = (x <= self.t)
        x[filters] = -torch.inf
        return x


class GlobalLocalAttentionHead(nn.Module):
    """
    单个注意力头。
    """

    def __init__(self):
        super(GlobalLocalAttentionHead, self).__init__()
        self.W_Q = None
        self.W_V = None
        self.M = None
        self.sigma = Sigma(1)

    def forward(self, x: Tensor) -> Tensor:
        # 尝试初始化参数
        self._try_init_parameter(x)
        # 输入参数维度 [batch, 259, 96]
        # [batch, 259, 96]
        T = x
        # [batch, 259, 96]
        K = T
        # [259, 259] @ [batch, 259, 96] = [batch, 259, 96]
        V = self.W_V @ T
        idx = tensor_to(torch.arange(0, K.shape[1]).unsqueeze(0).unsqueeze(-1))[0]
        # [batch, 259, 96]
        G = torch.cumsum(T, dim=1) / idx  # type: Tensor
        # [259, 259] @ [batch, 259, 96] = [batch, 259, 96]
        Q = self.W_Q @ G
        # [batch, 96, 259]
        K_T = torch.transpose(K, 1, 2)
        # [batch, 259, 96] @ [batch, 96, 259] = [batch, 259, 259]
        A = self.sigma(Q @ K_T) + self.M
        # [batch, 259, 259] @ [batch, 259, 96] = [batch, 259, 96]
        y = torch.softmax(A / sqrt(x.shape[1]), dim=1) @ V
        return y

    def _try_init_parameter(self, x: Tensor):
        token_size = x.shape[1]
        if self.W_Q is None:
            self.W_Q = nn.Parameter(tensor_to(*[torch.zeros(token_size, token_size)])[0])
            xavier_normal_(self.W_Q)
        if self.W_V is None:
            self.W_V = nn.Parameter(tensor_to(*[torch.zeros(token_size, token_size)])[0])
            xavier_normal_(self.W_V)
        if self.M is None:
            self.M = torch.triu(tensor_to(*[torch.ones((token_size, token_size))])[0] * -torch.inf, 1)


class GlobalLocalAttentionLayer(nn.Module):
    """
    全局-局部注意力层。
    """

    def __init__(self, H=16):
        super(GlobalLocalAttentionLayer, self).__init__()
        self.H = 16
        self.modules = [GlobalLocalAttentionHead() for _ in range(self.H)]

    def forward(self, x: Tensor) -> Tensor:
        # 拆分成 多个 张量
        x_list = torch.chunk(x, self.H, dim=1)
        outputs = []
        # 分别进行运算, 后连接
        for model, x in zip(self.modules, x_list):
            output = model(x)
            outputs.append(output)
        return torch.cat(outputs, dim=1)
