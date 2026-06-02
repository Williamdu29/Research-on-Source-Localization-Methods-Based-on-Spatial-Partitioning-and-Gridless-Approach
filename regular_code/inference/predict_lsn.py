import numpy as np
import torch

from regular_code.config import LSN_CFG
from regular_code.models.lsn import LSN
from regular_code.utils.array_signal import near_field_steering_geometric
from regular_code.utils.device import get_device


def generate_test_sample(K: int, cfg=LSN_CFG):
    thetas = np.random.uniform(-60 * np.pi / 180, 60 * np.pi / 180, size=K)
    rs = np.random.uniform(1e-6, cfg.wavelength, size=K)
    X = np.zeros((cfg.N, cfg.snapshots), dtype=np.complex128)

    sigma2 = cfg.ps / (10 ** (10 / 10))
    for k in range(K):
        a_k = near_field_steering_geometric(thetas[k], rs[k], cfg.N, cfg.wavelength, cfg.d)
        s_k = np.sqrt(cfg.ps / 2) * (np.random.randn(cfg.snapshots) + 1j * np.random.randn(cfg.snapshots))
        X += a_k[:, None] * s_k[None, :]

    noise = np.sqrt(sigma2 / 2) * (
        np.random.randn(cfg.N, cfg.snapshots) + 1j * np.random.randn(cfg.N, cfg.snapshots)
    )
    R = ((X + noise) @ (X + noise).conj().T) / float(cfg.snapshots)
    return np.transpose(np.stack([R.real, R.imag], axis=-1).astype(np.float32), (2, 0, 1))


def main():
    device = get_device()
    model = LSN(num_sources=LSN_CFG.N - 1).to(device)
    model.load_state_dict(torch.load(LSN_CFG.checkpoint_path, map_location=device))
    model.eval()

    print(f">>> 成功加载 {LSN_CFG.checkpoint_path}")
    for K_true in [1, 2, 3, 4]:
        x = generate_test_sample(K_true)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            K_pred = torch.argmax(prob, dim=1).item() + 1

        print(f"真实 K = {K_true}, 预测 K = {K_pred}, prob = {prob.detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
