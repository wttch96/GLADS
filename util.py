from torch import Tensor
import torch
from wth.torch.utils import tensor_to


def train_label(y: Tensor) -> Tensor:
    """
    将训练的输出转换为标签索引的形式。
    :param y: 训练的输出，维度为 batch * w
    :return: 输出的索引形式，维度为 batch
    """
    return y.argmax(dim=1)


def test_label(y: Tensor) -> Tensor:
    """
    将测试的输出转换为标签索引的形式。

    将最后一个维度求 argmax 后然后统计第2维出现频次最高的。
    :param y: 测试的输出，维度为 batch * n * w
    :return: 输出的索引形式，维度为 batch
    """
    y = y.argmax(dim=2)
    y = y.cpu()
    y, _ = torch.mode(y, dim=1)
    return tensor_to(y)
