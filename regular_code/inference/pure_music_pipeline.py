import numpy as np

from regular_code.config import LCN_CFG
from regular_code.utils.array_signal import generate_R_from_omega_phi, omega_to_theta_deg, theta_phi_to_r
from regular_code.utils.covariance import build_R1_R2_strict, normalize_R1_phase
from regular_code.utils.music import music_2elem_single, wrap_omega_to_physical_interval, wrap_phi_to_0_pi


def estimate_omega_phi_music(omega_true: float, phi_true: float, cfg=LCN_CFG):
    R = generate_R_from_omega_phi(omega_true, phi_true, cfg)
    R1, R2 = build_R1_R2_strict(R)
    if cfg.normalize_R1:
        R1 = normalize_R1_phase(R1)

    omega_grid = np.linspace(cfg.omega_grid_min, cfg.omega_grid_max, cfg.omega_grid_num, dtype=np.float64)
    phi_grid = np.linspace(cfg.phi_grid_min, cfg.phi_grid_max, cfg.phi_grid_num, dtype=np.float64)

    omega_hat_raw = music_2elem_single(R2, omega_grid, cfg)
    omega_hat = wrap_omega_to_physical_interval(omega_hat_raw, cfg.wavelength, cfg.d)

    phi_hat_raw = music_2elem_single(R1, phi_grid, cfg)
    phi_hat = wrap_phi_to_0_pi(phi_hat_raw)

    theta_true_deg = omega_to_theta_deg(omega_true, cfg.wavelength, cfg.d)
    theta_hat_deg = omega_to_theta_deg(omega_hat, cfg.wavelength, cfg.d)
    r_true = theta_phi_to_r(theta_true_deg, phi_true, cfg.wavelength, cfg.d)
    r_hat = theta_phi_to_r(theta_hat_deg, phi_hat, cfg.wavelength, cfg.d)

    print("=== Pure MUSIC estimation ===")
    print(f"theta_true(deg)={theta_true_deg:.6f}, theta_hat(deg)={theta_hat_deg:.6f}")
    print(f"omega_true(rad)={omega_true:.6f}, omega_hat(rad)={omega_hat:.6f} (raw={omega_hat_raw:.6f})")
    print(f"phi_true(rad)  ={phi_true:.6f}, phi_hat(rad)  ={phi_hat:.6f} (raw={phi_hat_raw:.6f})")
    print(f"r_true(m)      ={r_true:.6e}, r_hat(m)       ={r_hat:.6e}")

    return omega_hat, phi_hat, theta_hat_deg, r_hat


def main():
    tests = [
        (1.110721, 0.490874),
        (0.785398, 0.735000),
        (0.000000, 0.981748),
        (-1.110721, 0.490874),
    ]

    for i, (omega_true, phi_true) in enumerate(tests, 1):
        print(f"\n===== Case #{i}: omega_true={omega_true:.6f}, phi_true={phi_true:.6f} =====")
        estimate_omega_phi_music(omega_true, phi_true)


if __name__ == "__main__":
    main()
