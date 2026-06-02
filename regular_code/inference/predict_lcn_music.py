import numpy as np
import torch

from regular_code.config import LCN_CFG
from regular_code.models.lcn import LCN
from regular_code.utils.array_signal import generate_R_from_omega_phi
from regular_code.utils.covariance import R_to_tensor_2ch, build_R1_R2_strict, build_u_A, normalize_R1_phase, u_A_to_R1_R2
from regular_code.utils.device import get_device
from regular_code.utils.music import music_2elem_single, wrap_omega_to_physical_interval, wrap_phi_to_0_pi


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_once_lcn_music(omega_true: float, phi_true: float, model_path=None, cfg=LCN_CFG):
    device = get_device()
    model_path = cfg.checkpoint_path if model_path is None else model_path

    model = LCN(cfg.N).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    R = generate_R_from_omega_phi(omega_true, phi_true, cfg)
    x = torch.from_numpy(R_to_tensor_2ch(R)).unsqueeze(0).to(device)

    R1_gt, R2_gt = build_R1_R2_strict(R)
    R1_gt = normalize_R1_phase(R1_gt) if cfg.normalize_R1 else R1_gt
    u_gt = build_u_A(R1_gt, R2_gt)

    with torch.no_grad():
        u_pred = model(x).cpu().numpy().reshape(-1).astype(np.float32)

    R1_hat, R2_hat = u_A_to_R1_R2(u_pred)
    if cfg.normalize_R1:
        R1_hat = normalize_R1_phase(R1_hat)

    omega_grid = np.linspace(cfg.omega_grid_min, cfg.omega_grid_max, cfg.omega_grid_num, dtype=np.float64)
    phi_grid = np.linspace(cfg.phi_grid_min, cfg.phi_grid_max, cfg.phi_grid_num, dtype=np.float64)

    omega_hat_raw = music_2elem_single(R2_hat, omega_grid, cfg)
    omega_hat = wrap_omega_to_physical_interval(omega_hat_raw, cfg.wavelength, cfg.d)

    phi_hat_raw = music_2elem_single(R1_hat, phi_grid, cfg)
    phi_hat = wrap_phi_to_0_pi(phi_hat_raw)

    print("====================================")
    print(f"[truth] omega={omega_true:.6f}, phi={phi_true:.6f}")
    print(f"[est  ] omega={omega_hat:.6f} (raw={omega_hat_raw:.6f})")
    print(f"[est  ] phi  ={phi_hat:.6f} (raw={phi_hat_raw:.6f})")
    print(f"[u err] mean={float(np.mean(np.abs(u_pred - u_gt))):.6e}, max={float(np.max(np.abs(u_pred - u_gt))):.6e}")

    return omega_hat, phi_hat


def main():
    tests = [
        (0.00, 0.50),
        (0.30, 0.20),
        (-0.30, 0.20),
        (0.80, 0.50),
        (-0.80, 0.50),
        (1.10, 0.49),
        (-1.10, 0.49),
    ]

    for i, (omega_true, phi_true) in enumerate(tests, 1):
        print(f"\n\n===== Case #{i:02d} =====")
        run_once_lcn_music(omega_true, phi_true)


if __name__ == "__main__":
    main()
