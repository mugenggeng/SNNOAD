import torch
from DCLS.construct.modules import  Dcls1d

# Will construct kernels of size 7x7 with 3 elements inside each kernel
# m = Dcls1d(3, 16, kernel_count=3, dilated_kernel_size=21)
# input = torch.rand(8, 3, 256)
# output = m(input)
# print(output.shape)
# loss = output.sum()
# loss.backward()
# print(output, m.weight.grad, m.P.grad)
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft

# 假设有一个离散的概率分布，例如概率为 p = [0, 0.2, 0.1, 0.3, 0.4] 对应于离散事件 0, 1, 2, 3, 4
# p = np.array([0., 0., 1, 1, 0.])
#
# # 对离散概率分布进行傅里叶变换
# P = fft(p)
#
# # 对傅里叶变换结果进行平移和缩放
# # 平移: 将原点移动到频率为0的部分
# # 缩放: 根据傅里叶变换的性质，需要除以N（样本数）
# P = fftshift(P) / len(p)
#
# # 计算连续分布的频率响应
# freq = np.linspace(-0.5, 0.5, len(P))
# # print(P)
# # 绘制连续分布图
# plt.plot(freq, np.abs(P))
# plt.title("Continuous Distribution")
# plt.xlabel("Frequency")
# plt.ylabel("Probability Density")
# plt.show()

import numpy as np
import math

# 离散分布的概率值
p = np.array([0.1, 0.2, 0.3, 0.4])
N = len(p)


# 计算傅里叶变换的系数
def fft_to_continuous(p, N):
    n = np.arange(N)  # 创建一个范围数组
    freq = np.fft.fftfreq(N) * (2 * math.pi / N)  # 计算频率
    P = np.fft.fft(p) / N  # 执行傅里叶变换并归一化

    return n, freq, np.abs(P)


# 执行傅里叶变换
n, freq, P = fft_to_continuous(p, N)

# 打印结果
print("n\tfreq\t|P(f)|")
for i in range(len(n)):
    print(f"{n[i]}\t{freq[i]}\t{P[i]:.2f}")