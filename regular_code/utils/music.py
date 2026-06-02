import numpy as np
from numpy.linalg import eigh

from regular_code.utils.covariance import diagonal_loading, hermitianize


def music_2elem_single(R2x2: np.ndarray, grid: np.ndarray, cfg):
    R_use = hermitianize(R2x2)
    R_use = diagonal_loading(R_use, cfg.diagonal_loading_alpha)

    _, evecs = np.linalg.eigh(R_use)
    En = evecs[:, :1]

    P = np.zeros_like(grid, dtype=np.float64)
    for i, mu in enumerate(grid):
        a = np.array([1.0, np.exp(-1j * 2.0 * mu)], dtype=np.complex128)[:, None]
        denom = (a.conj().T @ En @ En.conj().T @ a).real.item()
        P[i] = 1.0 / (denom + 1e-12)

    return float(grid[int(np.argmax(P))])


def music_spectrum(R, theta_grid, r_grid, K=1, wavelength=1.0):
    eigvals, eigvecs = eigh(R)
    idx = np.argsort(eigvals)[::-1]
    Un = eigvecs[:, idx[K:]]
    P = np.zeros((len(theta_grid), len(r_grid)))
    M = R.shape[0]

    for i, theta in enumerate(theta_grid):
        for j, r in enumerate(r_grid):
            a = steering_vector(theta, r, M, wavelength)
            denom = np.linalg.norm(a.conj().T @ Un) ** 2
            P[i, j] = 1.0 / (denom.real + 1e-12)

    return P


def steering_vector(theta, r, M, wavelength=1.0):
    d = wavelength / 4
    omega = -2 * np.pi * d * np.sin(theta) / wavelength
    phi = np.pi * d**2 * np.cos(theta) ** 2 / (wavelength * r)
    m = np.arange(M)
    return np.exp(-1j * (omega * m + phi * m**2))


def wrap_phi_to_0_pi(phi_hat: float):
    return float(np.mod(phi_hat, np.pi))


def wrap_omega_to_physical_interval(omega_hat: float, wavelength: float, d: float):
    omega_max = 2.0 * np.pi * d / wavelength
    low, high = -omega_max, omega_max
    period = np.pi
    w = omega_hat
    while w < low:
        w += period
    while w > high:
        w -= period
    return float(np.clip(w, low, high))
