import torch
from torch import nn, Tensor


class Selector(nn.Module):
    """
    选择器。

    该选择器旨在解决全局-局部注意力层带来的一个小问题，即如何训练具有多个输入长度的样本，
    因为全局-局部注意力层输出的 Y = [y18, ..., y4n+18] 中的每个 y4i+18 都代表一个特征向量。
    """

    def __init__(self):
        super(Selector, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        # TODO 怎么保证输入输出一致呢？训练时选择，其他都置 0？ batch * n * w 和 batch * 1 * w 维度都不一样

        # 文章中说，测试阶段保留所有的分类结果，那这里就是只选择一个
        if self.train():
            # 训练阶段，随机选择数据进行
            idx = torch.randint(x.shape[1], (x.shape[0],))
            selected = x[torch.arange(x.shape[0]), idx, :]
            return selected
        else:
            # 测试阶段，返回所有的数据。
            return x
