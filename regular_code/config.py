from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SignalConfig:
    N: int = 5
    snapshots: int = 1024
    ps: float = 1.0
    fc: float = 2e6
    use_training_d: bool = True
    K: int = 1

    @property
    def wavelength(self) -> float:
        return 3e8 / self.fc

    @property
    def d(self) -> float:
        return self.wavelength / 4.0 if self.use_training_d else self.wavelength / 2.0


@dataclass
class LSNConfig(SignalConfig):
    SNR_min: float = -10.0
    SNR_max: float = 15.0
    num_samples: int = 500_000
    dataset_path: Path = PROJECT_ROOT / "data" / "LSNdataset.npz"
    checkpoint_path: Path = PROJECT_ROOT / "LSN_trained.pth"


@dataclass
class LCNConfig(SignalConfig):
    SNR_dB: float = 10.0
    SNR_dB_min: float = -10.0
    SNR_dB_max: float = 15.0
    phi_min: float = 0.01
    phi_max: float = float(np.pi - 0.01)
    omega_min: Optional[float] = None
    omega_max: Optional[float] = None
    normalize_R1: bool = True
    diagonal_loading_alpha: float = 1e-3
    omega_grid_min: float = -np.pi / 2
    omega_grid_max: float = np.pi / 2
    omega_grid_num: int = 20001
    phi_grid_min: float = 0.0
    phi_grid_max: float = np.pi
    phi_grid_num: int = 20001
    checkpoint_path: Path = PROJECT_ROOT / "LCN_trained_omega_phi_A_normR1.pth"


LSN_CFG = LSNConfig()
LCN_CFG = LCNConfig()
