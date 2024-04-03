from torch import nn, Tensor


class TaskBaseLayer(nn.Module):
    def __init__(self, out_features: int):
        super(TaskBaseLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class EncapulationTaskLayer(TaskBaseLayer):
    """
    是否恶意流量分类任务。

    训练阶段输入为: batch * 96
    测试阶段为: batch * 259 * 96

    输出为: batch * 2
    测试阶段怎么做？259 选最多的？
    """

    def __init__(self):
        super(EncapulationTaskLayer, self).__init__(2)


class TrafficTypeTaskLayer(TaskBaseLayer):
    """
    流量类型分类任务。


    训练阶段输入为: batch * 96
    测试阶段为: batch * 259 * 96

    输出为: batch * 6
    测试阶段怎么做？259 选最多的？
    """

    def __init__(self):
        super(TrafficTypeTaskLayer, self).__init__(6)


class ApplicationTypeTaskLayer(TaskBaseLayer):
    """
    应用类型任务。

    训练阶段输入为: batch * 96
    测试阶段为: batch * 259 * 96

    输出为: batch * 15
    测试阶段怎么做？259 选最多的？
    """

    def __init__(self):
        super(ApplicationTypeTaskLayer, self).__init__(15)
