import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scapy.all import PcapReader
from scapy.layers.dhcp import DHCP
from scapy.layers.dns import DNS
from scapy.layers.inet import TCP, UDP, Ether, IP
from scapy.layers.ntp import NTP
from scapy.layers.snmp import SNMP
from scapy.layers.llmnr import LLMNRQuery, LLMNRResponse

import torch
import torch.nn as nn


class InputData:
    hdr_list: list[np.ndarray]
    pay_list: list[np.ndarray]

    def __init__(self, input_file: str, N_p: int = 32, N_b: int = 784, max_len_p: int = 128):
        # 对于每一个 session 文件
        self.input_file = input_file
        self.N_p = N_p
        self.N_b = N_b
        self.max_len_p = max_len_p

        self._read()

    # @property
    # def hdr(self) -> np.ndarray:
    #     hdr = np.concatenate(self.hdr_list)
    #     # 填充0
    #     return np.pad(hdr, (0, (self.N_p - len(self.hdr_list)) * 4), mode='constant', constant_values=0)
    #
    # @property
    # def pay(self) -> np.ndarray:
    #     pay = np.concatenate(self.pay_list)
    #     # 填充0
    #     return np.pad(pay, (0, (self.N_b - len(pay))), mode='constant', constant_values=0)

    def _read(self):
        self.hdr_list = []
        self.pay_list = []
        n_b = 0
        with PcapReader(self.input_file) as reader:
            last_time = None
            server_ip = None
            for packet in reader:  # type: Ether
                # 过滤
                if len(self.hdr_list) == self.N_p and n_b == self.N_b:
                    break

                inter_arrival_time = 0 if last_time is None else packet.time - last_time
                inter_arrival_time *= 1000
                last_time = packet.time

                if packet.haslayer(DNS) or packet.haslayer(LLMNRQuery) or packet.haslayer(
                        LLMNRResponse) or packet.haslayer(DHCP) or packet.haslayer(NTP) or packet.haslayer(SNMP):
                    pass
                if 'IP' in packet:
                    ip = packet.getlayer(IP)  # type: IP
                    if server_ip is None:
                        server_ip = ip.dst

                if 'TCP' in packet:
                    tcp = packet.getlayer(TCP)  # type: TCP
                    hdr = np.array(
                        [len(tcp.original), tcp.window, inter_arrival_time, int(ip.src == server_ip)]).astype(int)
                    if len(self.hdr_list) < self.N_p:
                        self.hdr_list.append(hdr)
                    l = min(len(tcp.payload.original), self.N_b - n_b, self.max_len_p)
                    if l > 0:
                        n_b += l
                        pay = np.frombuffer(tcp.payload.original[0:l], dtype=np.uint8)
                        self.pay_list.append(pay)

                if 'UDP' in packet:
                    pass

            #
            # fig, axes = plt.subplots(1, 1)
            # i1 = Image.fromarray(self.pay_list.reshape(28, -1), mode='L')
            # axes.imshow(i1, cmap='gray')
            # plt.show()


if __name__ == '__main__':
    t = InputData('session-example-tcp.pcap')
    ret = np.array([])
    for i in range(0, len(t.pay_list)):
        arr = np.array([np.log(1 + t.hdr_list[i]), [0, 1, 0, 1]])
        ret = np.append(ret, arr)
        pay = t.pay_list[i]
        pay = pay[0: len(pay) // 4 * 4]
        ret = np.append(ret, pay / 255.0)
    ret = ret.reshape(-1, 4)
    ret = np.array([ret[i:i + 2,:].flatten() for i in range(0, len(ret) - 1)])
    # 2 * 1040 / 8 - 1
    print(ret)
    arr = np.pad(ret, ((0, 259 - len(ret)), (0, 0)), mode="constant", constant_values=0)

    # image =Image.fromarray(arr.reshape(-1, 8), mode='L')
    # plt.imshow(image, cmap='gray')
    # plt.show()

    tensor = torch.from_numpy(arr).to(torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    print(tensor.shape)
    cnn = nn.Conv2d(1, 96, (1, 4), stride=(1, 2))

    t = cnn(tensor)
    print(t.shape)
    depthwise = nn.Conv2d(96, 96, (1, 3), stride=1, padding=(0, 1), groups=96)
    t = depthwise(t)
    print(t.shape)
    pointwise = nn.Conv2d(96, 96, 1)
    t = pointwise(t)
    print(t.shape)
    merge = nn.Conv2d(96, 96, (1, 2), stride=(1, 2), groups=96)
    t = merge(t)
    print(t.shape)
    t = torch.transpose(t, 1, 2)
    print(t.shape)
    t = torch.squeeze(t)
    print(t.shape)
