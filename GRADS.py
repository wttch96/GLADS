from torch import nn, Tensor

from layers.tswe import TSWELayer
from layers.exchange import ExchangeLayer


class GRADS(nn.Module):

    def __init__(self):
        super(GRADS, self).__init__()
        self.tswe = TSWELayer()

        self.exchange = ExchangeLayer()

    def forward(self, x: Tensor) -> Tensor:
        return self.exchange(self.tswe(x))
