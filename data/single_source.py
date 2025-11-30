'''Steps:
1. Generate near-field signal and covariance matrix
2. Convert complex covariance to real+imag tensor (2×5×5)
3. LSN predicts number of signals → here expectation: 1 source
4. LCN reconstructs clean covariance matrix
5. Apply classical estimator (e.g., MUSIC) OR regression head for DOA (θ, r)
6. Compute RMSE'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import toeplitz, eigh
from math import pi


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

N = 5                     # 5 阵元均匀线性阵列
fc = 2e6                # carrier frequency
wavelength = 3e8 / fc   # 波长
snapshots = 1024
num_samples = 1000       # number simulation samples
SNR_range = np.linspace(-10, 15, num=10)
train_ratio = 0.7
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
save_path = "single_source_dataset.npz"   # saved dataset file

# near-field signal model
def near_field_steering_geometric(theta, r, N, wavelength, d=None):
    if d is None:
        d = wavelength / 4.0 # 阵元间距为 λ/4
    n = np.arange(N)
    r_km = np.sqrt(r**2 + (n*d)**2 - 2*n*d*r*np.sin(theta))
    # tau = (2*pi/lambda) * (r_km - r)
    phase = -2.0 * pi / wavelength * (r_km - r)   # note: a_m = exp(1j*phase) => exp(-j*2pi*(r_km-r)/lambda)
    a = np.exp(1j * phase)
    return a


def generate_sample(SNR_dB, theta=None, r=None, N=N, snapshots=1024, wavelength=wavelength):
    """Generate ONE sample: complex covariance + labels (theta, r)."""

    if theta is None:
        theta = np.random.uniform(-60 * pi/180, 60 * pi/180)  # 随机生成角度，范围 [-60°, 60°]
    if r is None:
        r = np.random.uniform(1e-3, wavelength) # 随机生成距离，范围 [1mm, λ]

    d = wavelength / 4.0  # 阵元间距设定为 λ/4
    a = near_field_steering_geometric(theta, r, N, wavelength, d)

    # set signal power and noise power to match SNR definition
    ps = 1.0
    sigma2 = ps / (10**(SNR_dB/10.0))  # noise power

    # generate signal snapshots with power ps
    s = np.sqrt(ps/2.0) * (np.random.randn(snapshots) + 1j*np.random.randn(snapshots)) # 生成复数信号，每个样本快拍数为 snapshots

    noise = np.sqrt(sigma2/2.0) * (np.random.randn(N, snapshots) + 1j*np.random.randn(N, snapshots)) # 生成复数噪声

    X = a[:,None] * s.reshape(1, -1) + noise # 接收信号矩阵，形状 (N, snapshots)
    R = (X @ X.conj().T) / float(snapshots) # 样本协方差矩阵，形状 (N, N)

    # split real and imaginary
    R_tensor = np.stack([R.real, R.imag], axis=-1)  # (N, N, 2)


    return R_tensor.astype(np.float32), np.array([theta, r], dtype=np.float32)

def generate_dataset(SNR_range=np.arange(-10, 16, 1), 
                     snapshots=1024, 
                     samples_per_SNR=1000, 
                     save=True,
                     save_path="./dataset.npz"):

    X, Y = [], []

    total_samples = len(SNR_range) * samples_per_SNR
    print(f">>> 生成数据集: 总样本数 = {total_samples} ({len(SNR_range)}种SNR，每种{samples_per_SNR}样本)")
    
    sample_counter = 0

    for snr in SNR_range:
        print(f"\n📌 SNR = {snr} dB:")

        for k in range(samples_per_SNR): # 每一种 SNR 生成 samples_per_SNR 个样本

            # 生成一个样本 (生成信号 -> 协方差估计 -> 复数拆分 → tensor)
            R_tensor, label = generate_sample(snr, snapshots=snapshots)

            X.append(R_tensor)   # shape = (N, N, 2)
            Y.append(label)      # label = (theta, r)

            sample_counter += 1
            if sample_counter % 500 == 0:
                print(f"   已完成 {sample_counter}/{total_samples}")

    X = np.array(X)
    Y = np.array(Y)

    if save:
        np.savez(save_path, X=X, Y=Y)
        print(f"\n📁 数据集保存成功: {save_path}")
        print(f"   X.shape = {X.shape}, Y.shape = {Y.shape}")
        # 形状说明: X.shape = (num_samples, N, N, 2), Y.shape = (num_samples, 2)

    return X, Y


if __name__ == "__main__":
    generate_dataset()