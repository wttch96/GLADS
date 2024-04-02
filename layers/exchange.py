from torch import nn, Tensor


class ExchangeBaseBlock(nn.Module):
    """
    交换层的一个基础块。

    每个块包含一个 残差 1D-DepthwiseCNN 一个 残差 1D-PointwiseCNN 和 GELU 激活函数。

    这个块在拥有不错性能的同时增大 token 的感受野。
    """

    def __init__(self, filters=96, kernel_size1=3, stride1=1, kernel_size2=1, stride2=1):
        """
        构造函数，参数默认，除非你要换用别的超参数。
        :param filters: filter_num 依赖于上一层的输入
        """
        super(ExchangeBaseBlock, self).__init__()
        self.filters = filters
        self.depth_wise_cnn = nn.Sequential(
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size1,
                      stride=stride1, groups=filters),
            nn.GELU(),
            nn.BatchNorm1d(num_features=filters)
        )

        self.point_wise_cnn = nn.Sequential(
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size2,
                      stride=stride2),
            nn.GELU(),
            nn.BatchNorm1d(num_features=filters)
        )

        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        n = x.shape[1]

        # [batch, n, 96, 3] -> [batch * n, 96, 3]
        x = x.view(batch * n, self.filters, -1)
        # 残差 1D-DepthwiseCNN
        # [batch * n, 96, 3]
        depth_wise_x = self.depth_wise_cnn(x)
        x = depth_wise_x + x
        # 残差 1D-PointwiseCNN
        # [batch * n, 96, 3]
        point_wise_x = self.point_wise_cnn(x)
        x = point_wise_x + x
        x = self.gelu(x)
        # -> [batch, n, 96, 3]
        return x.view(batch, n, self.filters, -1)


class ExchangeLayer(nn.Module):
    """
    交换层。

    使用 7 个基础交换层块，和一个 Merge (1D-CNN) 来融合 token 数据。

    本文的输出维度是: [batch, n, 96], n 最大为 259。
    """

    def __init__(self, filters=96, block_num=7):
        super(ExchangeLayer, self).__init__()
        self.blocks = [ExchangeBaseBlock() for _ in range(block_num)]
        self.layer = nn.Sequential(*self.blocks)

        self.merge_layer = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=2, stride=2, groups=filters)

    def forward(self, x: Tensor) -> Tensor:
        # [batch, n, 96, 3] -> [batch, n, 96, 3]
        x = self.layer(x)

        batch = x.shape[0]
        n = x.shape[1]
        # 为了可以 1D-CNN 卷积
        # -> [batch * n, 96, 3]
        x = x.view(batch * n, 96, 3)
        # 融合 token 信息
        # -> [batch * n, 96, 1]
        x = self.merge_layer(x)  # type: Tensor
        # -> [batch, n, 96, 1]
        x = x.view(batch, n, 96, 1)
        # -> [batch, n, 96]
        x = x.squeeze(dim=3)
        return x
