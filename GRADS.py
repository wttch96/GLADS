from torch import nn, Tensor

from layers.global_local_attention import GlobalLocalAttentionLayer
from layers.selector import Selector
from layers.tswe import TSWELayer
from layers.exchange import ExchangeLayer
from layers.task import EncapulationTaskLayer, TrafficTypeTaskLayer, ApplicationTypeTaskLayer


class GRADS(nn.Module):

    def __init__(self):
        super(GRADS, self).__init__()
        self.tswe = TSWELayer()

        self.exchange = ExchangeLayer()

        self.global_local_attention = GlobalLocalAttentionLayer()

        self.selector = Selector()

        self.ts1 = EncapulationTaskLayer()
        self.ts2 = TrafficTypeTaskLayer()
        self.ts3 = ApplicationTypeTaskLayer()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.tswe(x)
        x = self.exchange(x)
        x = self.global_local_attention(x)
        x = self.selector(x)

        return self.ts1(x), self.ts2(x), self.ts3(x)
