#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_velocity_profile(csv_path="out/racing_line_world.csv",
                             mu=1.1, g=9.81,
                             a_long_max_ratio=0.5,
                             v_max_global=8.0,
                             out_dir="out"):
    """
    Compute a curvature-constrained velocity profile using a forward-backward pass.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load racing line
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    s, x, y, kappa = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    ds = np.gradient(s)

    # --- Parameters ---
    a_lat_max = mu * g  # mu = friction coefficient
    a_long_acc = 0.8 * mu * g
    a_long_brake = 0.6 * mu * g
    kappa_thresh = 0.05

    # --- Effective curvature ---
    kappa_eff = np.copy(np.abs(kappa))
    kappa_eff[kappa_eff < kappa_thresh] = 0.0

    v_lat_max = np.sqrt(a_lat_max / (kappa_eff**0.7 + 1e-6))
    v_lat_max = np.clip(v_lat_max, 0.0, v_max_global)

    # --- Forward pass ---
    v_fwd = np.copy(v_lat_max)
    for i in range(1, len(v_fwd)):
        v_fwd[i] = min(v_lat_max[i],
                    np.sqrt(v_fwd[i-1]**2 + 2*a_long_acc*ds[i-1]))

    # --- Backward pass ---
    v_bwd = np.copy(v_lat_max)
    for i in reversed(range(len(v_bwd)-1)):
        v_bwd[i] = min(v_bwd[i],
                    np.sqrt(v_bwd[i+1]**2 + 2*a_long_brake*ds[i]))

    v_final = np.minimum(v_fwd, v_bwd)
    v_final = np.convolve(v_final, np.ones(5)/5, mode='same')
    # --- Save result ---
    traj = np.column_stack([s, x, y, kappa, v_final])
    out_csv = os.path.join(out_dir, "racing_line_with_speed.csv")
    np.savetxt(out_csv, traj, delimiter=",",
               header="s_m,x_m,y_m,kappa,v_mps", comments="")
    print(f"✓ Saved final trajectory with velocity profile: {out_csv}")

    # --- Visualization ---
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(s, v_final, 'r-', label='Velocity Profile')
    plt.xlabel("Arc length s [m]")
    plt.ylabel("Velocity [m/s]")
    plt.grid(); plt.legend()

    plt.subplot(2,1,2)
    plt.plot(s, np.abs(kappa), 'b-', label='|Curvature|')
    plt.xlabel("Arc length s [m]")
    plt.ylabel("Curvature [1/m]")
    plt.grid(); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "velocity_profile.png"), dpi=200)
    print(f"✓ Saved velocity profile plot: {out_dir}/velocity_profile.png")

    # --- Color-coded path visualization ---
    plt.figure(figsize=(8,8))
    sc = plt.scatter(x, y, c=v_final, cmap='viridis', s=4)
    plt.colorbar(sc, label="Velocity [m/s]")
    plt.axis('equal'); plt.title("Racing Line (Color-coded by Speed)")
    plt.savefig(os.path.join(out_dir, "racing_line_colored.png"), dpi=200)
    print(f"✓ Saved colored racing line: {out_dir}/racing_line_colored.png")

    plt.show()

if __name__ == "__main__":
    compute_velocity_profile()
