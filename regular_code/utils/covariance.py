import numpy as np


def R_to_tensor_2ch(R: np.ndarray):
    return np.stack([R.real, R.imag], axis=0).astype(np.float32)


def hermitianize(A: np.ndarray):
    return 0.5 * (A + A.conj().T)


def diagonal_loading(R: np.ndarray, alpha: float):
    if alpha <= 0:
        return R
    M = R.shape[0]
    scale = np.trace(R).real / M
    return R + alpha * scale * np.eye(M, dtype=R.dtype)


def mapped_to_idx(mapped: int, p: int) -> int:
    return mapped + p


def get_R_mapped(R: np.ndarray, i_mapped: int, j_mapped: int, p: int):
    return R[mapped_to_idx(i_mapped, p), mapped_to_idx(j_mapped, p)]


def build_R1_R2_strict(R: np.ndarray):
    N = R.shape[0]
    p = (N - 1) // 2
    R1 = np.zeros((p, p), dtype=np.complex128)
    R2 = np.zeros((p, p), dtype=np.complex128)
    for i in range(1, p + 1):
        for j in range(1, p + 1):
            R1[i - 1, j - 1] = get_R_mapped(R, i - j + 1, i - j, p)
            R2[i - 1, j - 1] = get_R_mapped(R, i - j, j - i, p)
    return R1, R2


def normalize_R1_phase(R1: np.ndarray):
    ref = R1[0, 0]
    if np.abs(ref) < 1e-12:
        return R1
    return R1 / ref


def build_u_A(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    u = np.zeros(12, dtype=np.float32)
    u[0] = R1[0, 0].real
    u[1] = R1[0, 1].real
    u[2] = R1[1, 0].real
    u[3] = R1[0, 0].imag
    u[4] = R1[0, 1].imag
    u[5] = R1[1, 0].imag
    u[6] = R2[0, 0].real
    u[7] = R2[0, 1].real
    u[8] = R2[1, 0].real
    u[9] = R2[0, 0].imag
    u[10] = R2[0, 1].imag
    u[11] = R2[1, 0].imag
    return u


def u_A_to_R1_R2(u: np.ndarray):
    u = np.asarray(u).reshape(-1)
    if u.shape[0] != 12:
        raise ValueError(f"u must have length 12, got {u.shape[0]}")

    R1_11 = u[0] + 1j * u[3]
    R1_12 = u[1] + 1j * u[4]
    R1_21 = u[2] + 1j * u[5]
    R1 = np.array([[R1_11, R1_12], [R1_21, R1_11]], dtype=np.complex128)

    R2_11 = u[6] + 1j * u[9]
    R2_12 = u[7] + 1j * u[10]
    R2_21 = u[8] + 1j * u[11]
    R2 = np.array([[R2_11, R2_12], [R2_21, R2_11]], dtype=np.complex128)
    return R1, R2
