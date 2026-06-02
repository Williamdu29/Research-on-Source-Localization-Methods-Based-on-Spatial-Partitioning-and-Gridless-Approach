import numpy as np
import torch
from torch.utils.data import Dataset

from regular_code.config import LCN_CFG
from regular_code.utils.array_signal import generate_R_from_omega_phi
from regular_code.utils.covariance import R_to_tensor_2ch, build_R1_R2_strict, build_u_A, normalize_R1_phase


class OmegaPhiDataset(Dataset):
    def __init__(self, num_samples: int, cfg=LCN_CFG):
        self.num_samples = num_samples
        self.cfg = cfg

        omega_max_phys = 2.0 * np.pi * cfg.d / cfg.wavelength
        self.omega_min = -omega_max_phys if cfg.omega_min is None else cfg.omega_min
        self.omega_max = omega_max_phys if cfg.omega_max is None else cfg.omega_max
        self.phi_min = cfg.phi_min
        self.phi_max = cfg.phi_max

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        omega = np.random.uniform(self.omega_min, self.omega_max)
        phi = np.random.uniform(self.phi_min, self.phi_max)
        SNR_dB = np.random.uniform(self.cfg.SNR_dB_min, self.cfg.SNR_dB_max)

        R = generate_R_from_omega_phi(omega, phi, self.cfg, SNR_dB)
        x = R_to_tensor_2ch(R)

        R1, R2 = build_R1_R2_strict(R)
        if self.cfg.normalize_R1:
            R1 = normalize_R1_phase(R1)
        u = build_u_A(R1, R2)

        return torch.from_numpy(x), torch.from_numpy(u)
