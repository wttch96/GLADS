import torch

from torch import nn

from data.extractor import SessionDataExtractor
from GRADS import GRADS

data = SessionDataExtractor("./session-example-tcp.pcap")
data = torch.from_numpy(data.window).unsqueeze(0).to(torch.float32).to(device="mps")

print(data.shape)

net = GRADS().to(device="mps")

data = net(data)
print(data)