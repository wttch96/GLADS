from torch import nn, Tensor

from layers.global_local_attention import GlobalLocalAttentionLayer
from layers.tswe import TSWELayer
from layers.exchange import ExchangeLayer


class GRADS(nn.Module):

    def __init__(self):
        super(GRADS, self).__init__()
        self.tswe = TSWELayer()

        self.exchange = ExchangeLayer()

        self.global_local_attention = GlobalLocalAttentionLayer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.tswe(x)
        x = self.exchange(x)
        x = self.global_local_attention(x)
        return x
