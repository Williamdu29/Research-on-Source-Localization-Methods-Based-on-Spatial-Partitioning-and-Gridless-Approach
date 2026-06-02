import numpy as np

from regular_code.config import LSN_CFG
from regular_code.utils.array_signal import near_field_steering_geometric


def generate_sample_lsn(cfg=LSN_CFG):
    K = np.random.randint(1, cfg.N)
    thetas = np.random.uniform(-60 * np.pi / 180, 60 * np.pi / 180, size=K)
    rs = np.random.uniform(1e-6, cfg.wavelength, size=K)
    SNR_dB = np.random.uniform(cfg.SNR_min, cfg.SNR_max)

    X = np.zeros((cfg.N, cfg.snapshots), dtype=np.complex128)
    sigma2_list = []

    for k in range(K):
        a_k = near_field_steering_geometric(thetas[k], rs[k], cfg.N, cfg.wavelength, cfg.d)
        sigma2 = cfg.ps / (10 ** (SNR_dB / 10))
        sigma2_list.append(sigma2)
        s_k = np.sqrt(cfg.ps / 2) * (np.random.randn(cfg.snapshots) + 1j * np.random.randn(cfg.snapshots))
        X += a_k[:, None] * s_k[None, :]

    sigma2_final = np.mean(sigma2_list)
    noise = np.sqrt(sigma2_final / 2) * (
        np.random.randn(cfg.N, cfg.snapshots) + 1j * np.random.randn(cfg.N, cfg.snapshots)
    )
    X = X + noise
    R = (X @ X.conj().T) / float(cfg.snapshots)
    R_tensor = np.stack([R.real, R.imag], axis=-1).astype(np.float32)
    return R_tensor, K


def generate_lsn_dataset(cfg=LSN_CFG):
    X_list = []
    K_list = []

    print(f">>> 开始生成 LSN 训练数据集，共 {cfg.num_samples} 个样本...")
    for i in range(cfg.num_samples):
        R_tensor, K = generate_sample_lsn(cfg)
        X_list.append(R_tensor)
        K_list.append(K)

        if (i + 1) % 50000 == 0:
            print(f"   已完成 {i + 1}/{cfg.num_samples}")

    X = np.array(X_list, dtype=np.float32)
    K_arr = np.array(K_list, dtype=np.int32)

    cfg.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.dataset_path, X=X, K=K_arr)

    print("数据已保存:", cfg.dataset_path)
    print("X.shape =", X.shape)
    print("K.shape =", K_arr.shape)


if __name__ == "__main__":
    generate_lsn_dataset()
