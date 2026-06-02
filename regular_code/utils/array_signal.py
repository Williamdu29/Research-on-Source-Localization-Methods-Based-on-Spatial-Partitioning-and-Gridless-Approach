from typing import Optional

import numpy as np


def near_field_steering_geometric(theta, r, N, wavelength, d):
    n = np.arange(N)
    r_n = np.sqrt(r**2 + (n * d) ** 2 - 2 * n * d * r * np.sin(theta))
    phase = -2.0 * np.pi / wavelength * (r_n - r)
    return np.exp(1j * phase)


def generate_R_from_omega_phi(omega: float, phi: float, cfg, SNR_dB: Optional[float] = None) -> np.ndarray:
    N = cfg.N
    p = (N - 1) // 2
    m = np.arange(-p, p + 1, dtype=np.float64)
    a = np.exp(-1j * (omega * m + phi * (m ** 2)))

    snr = cfg.SNR_dB if SNR_dB is None else SNR_dB
    sigma2 = cfg.ps / (10.0 ** (snr / 10.0))
    s = np.sqrt(cfg.ps / 2) * (np.random.randn(cfg.snapshots) + 1j * np.random.randn(cfg.snapshots))
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(N, cfg.snapshots) + 1j * np.random.randn(N, cfg.snapshots))

    X = a[:, None] * s[None, :] + noise
    return (X @ X.conj().T) / cfg.snapshots


def theta_r_to_omega_phi(theta_deg: float, r: float, fc: float, use_training_d: bool = True):
    wavelength = 3e8 / fc
    d = wavelength / 4.0 if use_training_d else wavelength / 2.0
    theta = np.deg2rad(theta_deg)
    omega = -2.0 * np.pi * d * np.sin(theta) / wavelength
    phi = np.pi * d * d * np.cos(theta) ** 2 / (wavelength * r)
    return float(omega), float(phi)


def omega_to_theta_deg(omega: float, wavelength: float, d: float):
    s = -omega * wavelength / (2.0 * np.pi * d)
    return float(np.arcsin(np.clip(s, -1.0, 1.0)) * 180.0 / np.pi)


def theta_phi_to_r(theta_deg: float, phi: float, wavelength: float, d: float):
    if phi <= 1e-12:
        return float("inf")
    theta = np.deg2rad(theta_deg)
    return float((np.pi * d * d * np.cos(theta) ** 2) / (wavelength * phi))
