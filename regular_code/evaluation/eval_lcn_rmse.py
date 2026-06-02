import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from regular_code.config import LCN_CFG, PROJECT_ROOT
from regular_code.inference.predict_lcn_music import run_once_lcn_music
from regular_code.models.lcn import LCN
from regular_code.utils.array_signal import generate_R_from_omega_phi, omega_to_theta_deg, theta_phi_to_r, theta_r_to_omega_phi
from regular_code.utils.covariance import R_to_tensor_2ch, normalize_R1_phase, u_A_to_R1_R2
from regular_code.utils.device import get_device
from regular_code.utils.music import music_2elem_single, wrap_omega_to_physical_interval, wrap_phi_to_0_pi


def estimate_one_trial(R: np.ndarray, model, device, omega_grid: np.ndarray, phi_grid: np.ndarray, cfg=LCN_CFG):
    x = torch.from_numpy(R_to_tensor_2ch(R)).unsqueeze(0).to(device)

    with torch.no_grad():
        u_pred = model(x).cpu().numpy().reshape(-1).astype(np.float32)

    R1_hat, R2_hat = u_A_to_R1_R2(u_pred)
    if cfg.normalize_R1:
        R1_hat = normalize_R1_phase(R1_hat)

    omega_hat_raw = music_2elem_single(R2_hat, omega_grid, cfg)
    omega_hat = wrap_omega_to_physical_interval(omega_hat_raw, cfg.wavelength, cfg.d)

    phi_hat_raw = music_2elem_single(R1_hat, phi_grid, cfg)
    phi_hat = wrap_phi_to_0_pi(phi_hat_raw)

    theta_hat_deg = omega_to_theta_deg(omega_hat, cfg.wavelength, cfg.d)
    r_hat = theta_phi_to_r(theta_hat_deg, phi_hat, cfg.wavelength, cfg.d)
    return theta_hat_deg, r_hat


def run_rmse_vs_snr(theta_true_deg=30.0, r_true=None, snr_list=None, mc_runs=1000, omega_grid_num=4001, phi_grid_num=4001, cfg=LCN_CFG):
    assert cfg.N == 5, "当前实现按 N=5 对齐（p=2 => R1/R2 为 2x2, u=12）。"
    r_true = cfg.wavelength / 6.0 if r_true is None else r_true
    snr_list = np.arange(-10, 16, 1) if snr_list is None else np.asarray(snr_list)

    omega_true, phi_true = theta_r_to_omega_phi(theta_true_deg, r_true, cfg.fc, cfg.use_training_d)
    omega_grid = np.linspace(cfg.omega_grid_min, cfg.omega_grid_max, omega_grid_num, dtype=np.float64)
    phi_grid = np.linspace(cfg.phi_grid_min, cfg.phi_grid_max, phi_grid_num, dtype=np.float64)

    device = get_device()
    model = LCN(cfg.N).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    model.eval()

    angle_rmse = np.zeros_like(snr_list, dtype=np.float64)
    range_rmse = np.zeros_like(snr_list, dtype=np.float64)

    for i, snr_db in enumerate(snr_list):
        theta_err2 = np.empty(mc_runs, dtype=np.float64)
        r_err2 = np.empty(mc_runs, dtype=np.float64)

        for t in range(mc_runs):
            R = generate_R_from_omega_phi(omega_true, phi_true, cfg, float(snr_db))
            theta_hat_deg, r_hat = estimate_one_trial(R, model, device, omega_grid, phi_grid, cfg)
            theta_err2[t] = (theta_hat_deg - theta_true_deg) ** 2
            r_err2[t] = (r_hat - r_true) ** 2

        angle_rmse[i] = np.sqrt(np.mean(theta_err2))
        range_rmse[i] = np.sqrt(np.mean(r_err2))
        print(f"SNR={snr_db:>4} dB | RMSE(theta)={angle_rmse[i]:.6f} deg | RMSE(r)={range_rmse[i]:.6e} m")

    return omega_true, phi_true, theta_true_deg, r_true, angle_rmse, range_rmse, snr_list


def plot_curves(snr_list, angle_rmse, range_rmse, save_angle_png: Path, save_range_png: Path):
    plt.figure()
    plt.plot(snr_list, angle_rmse, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Angle RMSE (deg)")
    plt.title("Angle RMSE vs SNR (LCN + MUSIC)")
    plt.grid(True)
    plt.savefig(save_angle_png, dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(snr_list, range_rmse, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Range RMSE (m)")
    plt.title("Range RMSE vs SNR (LCN + MUSIC)")
    plt.grid(True)
    plt.savefig(save_range_png, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LCN + MUSIC RMSE over SNR.")
    parser.add_argument("--mc-runs", type=int, default=1000)
    parser.add_argument("--snr-min", type=int, default=-10)
    parser.add_argument("--snr-max", type=int, default=15)
    parser.add_argument("--snr-step", type=int, default=1)
    parser.add_argument("--omega-grid-num", type=int, default=4001)
    parser.add_argument("--phi-grid-num", type=int, default=4001)
    parser.add_argument("--save-npz", type=Path, default=PROJECT_ROOT / "rmse_results.npz")
    parser.add_argument("--save-angle-png", type=Path, default=PROJECT_ROOT / "regular_code" / "evaluation" / "angle_rmse_vs_snr.png")
    parser.add_argument("--save-range-png", type=Path, default=PROJECT_ROOT / "regular_code" / "evaluation" / "range_rmse_vs_snr.png")
    args = parser.parse_args()

    snr_list = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    omega_true, phi_true, theta_true_deg, r_true, angle_rmse, range_rmse, snr_list = run_rmse_vs_snr(
        snr_list=snr_list,
        mc_runs=args.mc_runs,
        omega_grid_num=args.omega_grid_num,
        phi_grid_num=args.phi_grid_num,
    )

    np.savez(
        args.save_npz,
        snr_list=snr_list,
        angle_rmse_deg=angle_rmse,
        range_rmse_m=range_rmse,
        theta_true_deg=theta_true_deg,
        r_true_m=r_true,
        omega_true=omega_true,
        phi_true=phi_true,
        fc=LCN_CFG.fc,
        snapshots=LCN_CFG.snapshots,
        mc_runs=args.mc_runs,
        omega_grid_num=args.omega_grid_num,
        phi_grid_num=args.phi_grid_num,
    )
    plot_curves(snr_list, angle_rmse, range_rmse, args.save_angle_png, args.save_range_png)

    print("\nSaved:")
    print(f"  npz  -> {args.save_npz}")
    print(f"  fig1 -> {args.save_angle_png}")
    print(f"  fig2 -> {args.save_range_png}")
    print("\nTruth:")
    print(f"  theta_true = {theta_true_deg:.2f} deg")
    print(f"  r_true     = {r_true:.6f} m (=lambda/6)")
    print(f"  omega_true = {omega_true:.6f} rad")
    print(f"  phi_true   = {phi_true:.6f} rad")


if __name__ == "__main__":
    main()
