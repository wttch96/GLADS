import torch
from torch import nn, Tensor


class TSWELayer(nn.Module):
    """
    Temporal Sliding Window Embedding Layer

    该层的输入维度是 [batch, n, w]: batch 为批处理大小; n 为最大能处理的窗口数量;w 为窗口的大小文章固定为 8。

    实现主要是 1D-CNN 文章中说明了两种卷积方式:
    1). kernel_size = 8, stride = 4, filter=96      输出维度: [batch, n, 96, 1]
    2). kernel_size = 4, stride = 2, filter=96      输出维度: [batch, n, 96, 3]
    第二种方式更好，后面是 Exchange Layer 可以刚好把 token 混合成一个大 token。

    经过这样的设计，输出的 token 的感受野为 4，[_, i, 96, 3] 即可以从输入的 [_, 4 * i + 1 : 4 * i + 4, w] 中学习信息。


    按照本文的论文，输入的维度为 [batch, n, 8], 最后的输出维度为 [batch, n, 96, 3]。
    即论文中所提的这几个层并不是压缩信息，而是将信息重新编码为大的 token。

    详见 3.3.2 TWSE layer
    """

    def __init__(self, filters=96, kernel_size=4, stride=2):
        super(TSWELayer, self).__init__()
        self.window_embedding = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        n = x.shape[1]
        w = x.shape[2]
        # [batch, n, w] -> [batch * n, 1, w]
        x = x.view(batch * n, 1, w)
        # -> [batch * n, 96, 3]
        x = self.window_embedding(x)

        # -> [batch, n, 96, 3]
        return x.reshape((batch, n, 96, -1))
