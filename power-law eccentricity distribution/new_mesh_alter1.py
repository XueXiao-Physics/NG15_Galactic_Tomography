#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_alter1_uniform_e0bar_dEdfgw_num_nocut.py

1) Load data from the original spectrum/grid files:
     - dEdfgw_num_nocut.npy      six-dimensional spectrum tensor
     - dEdfgw_edges_nocut.npy    corresponding (rho_mesh, gamma_mesh, e0_ori, M_mesh, q_mesh, f_mesh)
     - edges_M40_…_alter.npz     grids for merger-rate integration (rho_den, q_den, z_all)
     - nden_M40_…npy            merger-rate tensor dN

2) Construct the e0bar grid e0bar_vals and compute the xi weight matrix W

3) Apply weighting along axis 2 (original e0) and reorder dimensions

4) Save new_mesh and all coordinate arrays
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # Grids for merger-rate integration
    rho_den, q_den, z_all = np.load("edgesholo_corrected_M40_Q21_Z22_copy.npy", allow_pickle=True)
    dN = np.load("ndenholo_corrected_M40_Q21_Z22_copy.npy", allow_pickle=True)

    # Grids for precomputed spectrum & the original 6D tensor
    edges = np.load("dEdfgw_edges_nocut.npy", allow_pickle=True)
    rho_mesh, gamma_mesh, e0_ori, M_mesh, q_mesh, f_mesh = edges
    dEdfgw_mesh = np.load("dEdfgw_num_nocut.npy", mmap_mode='r')

    return rho_den, q_den, z_all, dN, rho_mesh, gamma_mesh, e0_ori, M_mesh, q_mesh, f_mesh, dEdfgw_mesh


def main():
    # 1) Load all data
    (rho_den, q_den, z_all,
     dN,
     rho_mesh, gamma_mesh, e0_ori, M_mesh, q_mesh, f_mesh,
     dEdfgw_mesh) = load_data()

    # 2) Construct the e0bar grid & xi weights
    N_bar = 20
    e0bar_vals = np.linspace(0.001, 0.999, N_bar)  # uniform discretization in e0bar
    xi_vals = e0bar_vals / (1.0 - e0bar_vals)  # xi = e0bar/(1-e0bar)

    # Spacing of the original e0 grid (assumed evenly spaced)
    spacing = e0_ori[1] - e0_ori[0]

    # Build weight matrix W[e0bar_index, e0_index]
    N_e0 = len(e0_ori)
    W = np.zeros((N_bar, N_e0))
    for i, xi in enumerate(xi_vals):
        W[i] = xi * e0_ori ** (xi - 1) * spacing
    W /= W.sum(axis=1, keepdims=True)  # normalize each row

    # 3) Tensor contraction: weighted integration along the original e0 axis
    #    After tensordot, new_tmp has shape = (e0bar, rho, gamma, M, q, f)
    new_tmp = np.tensordot(W, dEdfgw_mesh, axes=([1], [2]))

    # 4) Reorder to (rho, gamma, e0bar, M, q, f)
    new_mesh = np.transpose(new_tmp, (1, 2, 0, 3, 4, 5))

    # Plot the weight distributions
    plt.figure(figsize=(8, 5))
    for i in range(N_bar):
        plt.plot(e0_ori, W[i], label=f"e0bar={e0bar_vals[i]:.3f}")
    plt.xlabel("Original e0")
    plt.ylabel("Weight w(e0 | e0bar)")
    plt.legend(fontsize="small", ncol=2)
    plt.title("e0 Weight Distributions (Uniform e0bar Grid)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("e0bar_weights.png", dpi=300)
    plt.close()

    # 5) Save new_mesh + all grid coordinates to ensure downstream indexing works
    np.savez(
        "alter1_uniform_e0bar_dEdfgw_num_nocut.npz",
        new_mesh=new_mesh,
        rho_mesh=rho_mesh,
        gamma_mesh=gamma_mesh,
        e0bar_vals=e0bar_vals,
        M_mesh=M_mesh,
        q_mesh=q_mesh,
        f_mesh=f_mesh
    )
    print("Done: new_mesh shape", new_mesh.shape)
    print("Coordinate array lengths:",
          f"rho:{len(rho_mesh)}, gamma:{len(gamma_mesh)}, e0bar:{len(e0bar_vals)}, "
          f"M:{len(M_mesh)}, q:{len(q_mesh)}, f:{len(f_mesh)}")


if __name__ == "__main__":
    main()




