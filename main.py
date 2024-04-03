import torch

from torch import nn

from data.extractor import SessionDataExtractor
from GRADS import GRADS
from util import train_label, test_label

data = SessionDataExtractor("./session-example-tcp.pcap")
data = torch.from_numpy(data.window).unsqueeze(0).to(torch.float32).to(device="mps")

print(data.shape)

net = GRADS().to(device="mps")
net.eval()
data = net(data)
print(test_label(data[0]), test_label(data[1]), test_label(data[2]))