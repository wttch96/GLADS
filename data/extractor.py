from math import ceil
from typing import Optional

import numpy as np
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import PcapReader


class SessionDataExtractor:
    """
    Session 数据文件提取，需要使用解析出来的 Session 单独的 pcap 文件。

    pcap 按 Session 拆分方式参见：https://github.com/wttch96/PcapPreprocess
    :ivar hdr_list: hdr 数据列表
    :ivar pay_list: pay 数据列表
    """
    hdr_list: list[np.ndarray]
    pay_list: list[np.ndarray]

    def __init__(self, pcap_file: str, N_p=32, N_b=784, max_pack_len=128, max_element_len=1040):
        """
        构造函数。
        :param pcap_file: pcap 文件位置
        :param N_p: 论文中参数 N_p 即 hdr 头的个数
        :param N_b: 论文中参数 N_b 即 pay 数据总共保留的长度
        :param max_pack_len: 单个 pack 保留的最大数据长度
        :param max_element_len: 最大能处理的元素个数
        """
        self.pcap_file = pcap_file
        self.N_p = N_p
        self.N_b = N_b
        self.max_pack_len = max_pack_len
        self.max_element_len = max_element_len

        self._read()

    def _read(self):
        """读取 pcap 文件，并从中提取 hdr 和 pay 载体数据。"""
        self.hdr_list = []
        self.pay_list = []
        # 一节截取的 pay 数据长度
        self._pay_count = 0
        with PcapReader(self.pcap_file) as reader:
            last_arrival_time = None
            server_ip = None
            for pack in reader:  # type: Ether
                # 满足条件可以直接跳出循环
                if len(self.hdr_list) == self.N_p and self._pay_count == self.N_b:
                    break

                # TODO 可以在此处添加 pack 过滤，过滤 ip 过滤协议等
                inter_arrival_time = 0 if last_arrival_time is None else pack.time - last_arrival_time
                inter_arrival_time *= 1000
                # 上一个包的到达时间
                last_arrival_time = pack.time

                if pack.haslayer(IP):
                    ip = pack.getlayer(IP)  # type: IP
                    if server_ip is None:
                        # 如果 server_ip 为 None 就把第一个包的 dst 当作 server_ip 以判断流向
                        server_ip = ip.dst
                else:
                    # 没有 ip 层就执行下一个包
                    continue
                direction = ip.src == server_ip

                if pack.haslayer(TCP):
                    tcp = pack.getlayer(TCP)  # type: TCP
                    self._extract_hdr(inter_arrival_time, direction, tcp=tcp)
                    self._extract_pay(tcp.payload)

                if pack.haslayer(UDP):
                    udp = pack.getlayer(UDP)  # type: UDP
                    self._extract_hdr(inter_arrival_time, direction, udp=udp)
                    self._extract_pay(udp.payload)

    def _extract_hdr(self, inter_arrival_time: int, direction: bool,
                     tcp: Optional[TCP] = None, udp: Optional[UDP] = None):
        """
        从给定的 tcp 或者 udp 数据中提取 hdr 四元组数据
        :param inter_arrival_time: 到达时间间隔
        :param direction: 数据流向
        :param tcp: tcp 数据，可选
        :param udp: udp 数据，可选
        """
        hdr = None
        if tcp is not None:
            # 提取 tcp 的数据
            hdr = np.array([len(tcp.original), tcp.window, inter_arrival_time, direction]).astype(int)
        if udp is not None:
            hdr = np.array([len(udp.original), 0, inter_arrival_time, direction]).astype(int)
            # 提取 udp 的数据

        if hdr is not None and len(self.hdr_list) < self.N_p:
            self.hdr_list.append(hdr)

    def _extract_pay(self, payload: Raw):
        """
        从 packet 包的数据载体中截取部分数据

        截取数据包的长度：
        <载体的长度，需要截取的剩余的长度，最大的截取长度> 三者的最小值。
        即:不能超过载体长度，总长度不能超过 self.N_b, 不能超过最大的截取长度
        :param payload: packet 包的数据载体
        """
        pay_len = min(len(payload), self.N_b - self._pay_count, self.max_pack_len)
        if pay_len > 0:
            # 截取数据
            pay = np.frombuffer(payload.original[0:pay_len], dtype=np.uint8)
            # 保存截取的数据
            self.pay_list.append(pay)
            self._pay_count += pay_len

    @property
    def patch(self) -> np.ndarray:
        ret = np.array([])
        for i, pay in enumerate(self.pay_list):  # type: int, [np.ndarray]
            ret = np.append(ret, np.log(1 + self.hdr_list[i]))
            ret = np.append(ret, [1, 0, 1, 0])
            # 保存 4 的整倍数
            length = len(pay) // 4 * 4
            pay = pay[0: length]
            ret = np.append(ret, pay / 255.0)
        ret = np.append(ret, self.hdr_list[i + 1:])
        return ret.reshape(-1, 4)

    @property
    def window(self) -> np.ndarray:
        patch = self.patch
        window = np.array([])
        for i in range(0, len(patch) - 1):
            window = np.append(window, patch[i: i + 2])

        ret = window.reshape(-1, 8)
        # 需要填充的大小
        pad_len = ceil(self.max_element_len * 2 / 8) - 1 - len(ret)

        return np.pad(ret, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
