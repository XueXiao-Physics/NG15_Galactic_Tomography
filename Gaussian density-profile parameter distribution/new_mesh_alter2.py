import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def load_data():
    # Grids for precomputed spectrum & the original tensor
    edges = np.load("dEdfgw_edges_nocut_alter2.npy", allow_pickle=True)
    ratio, epc, M_mesh, q_mesh, f_mesh = edges
    dEdfgw_mesh = np.load("dEdfgw_num_nocut_alter2.npy", mmap_mode='r')
    #   data_den = np.load("edges_M40_Q21_Z22_holov1p6_SAM_NANO15Astro_alter_copy.npz", allow_pickle=True)
    _M, _q, _z = np.load("edgesholo_M40_Q21_Z22.npy", allow_pickle=True)
    _dN = np.load("ndenholo_M40_Q21_Z22.npy", allow_pickle=True)

    return ratio, epc, M_mesh, q_mesh, f_mesh, dEdfgw_mesh, _M, _q, _z, _dN


def main():
    # 1) Load all data
    (
        ratio, epc, M_mesh, q_mesh, f_mesh,
        dEdfgw_mesh, _M, _q, _z, _dN) = load_data()

    # 2) Construct the (theta, kappa) grid
    theta_vals = np.linspace(-4.0, 4.0, 18)  # mean of log10(ratio)
    kappa_vals = np.linspace(0.1, 6, 15)  # width (sigma) of log10(ratio)

    log_ratio = np.log10(ratio)

    Nt, Nk = len(theta_vals), len(kappa_vals)
    # spec5d shape: (Nr, Ne, NM, Nq, Nf)
    Nr, Ne, NM, Nq, Nf = dEdfgw_mesh.shape

    # 3) Pre-allocate new_spec (replace ratio axis with theta, kappa)
    new_spec = np.zeros((Nt, Nk, Ne, NM, Nq, Nf))

    # 4) For each (theta, kappa), perform the weighted integral
    for itheta, theta in enumerate(theta_vals):
        for ik, k in enumerate(kappa_vals):
            weights = np.exp(-0.5 * ((log_ratio - theta) / k) ** 2)
            weights /= weights.sum()
            assert np.allclose(np.diff(log_ratio), np.diff(log_ratio)[0], rtol=1e-9,
                               atol=1e-12), "log_ratio is not evenly spaced"
            assert np.isfinite(weights).all() and np.isclose(weights.sum(), 1.0, rtol=1e-12,
                                                             atol=1e-14), f"weights not normalized: sum={weights.sum()}"
            # Weight and collapse along the ratio dimension
            # dEdfgw_mesh axis order: (ratio, epc, M, q, f)
            new_spec[itheta, ik] = np.tensordot(weights, dEdfgw_mesh, axes=([0], [0]))
            """
            for ie, e_val in enumerate(epc):
                # First extract the (M, q, f) 3D slice
                slice_mqf = new_spec[itheta, ik, ie]   # shape = (NM, Nq, Nf)

                # Integrate over M, q using the merger-rate tensor _dN
                # broadcast _dN[:,:,10] -> (NM, Nq, 1)
                weighted = slice_mqf * _dN[:, :, 10, None]
                tmp_M   = np.trapz(weighted, _M, axis=0)   # shape = (Nq, Nf)
                test    = np.trapz(tmp_M, _q, axis=0)      # shape = (Nf,)

                # Plot and save
                plt.figure()
                plt.loglog(f_mesh, test)
                plt.title(f"theta={theta:.1f}, kappa={k:.1f}, epc={e_val:.3f}")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"10_figures_recombined_spectrum/test_th{itheta}_kp{ik}_epc{ie}.png",
                        bbox_inches='tight')
                plt.close()
             """
    print("Done: new_spec shape", new_spec.shape)

    np.savez(
        "dEdfgw_recombined_theta_kappa.npz",
        new_spec=new_spec,
        theta_vals=theta_vals,
        kappa_vals=kappa_vals,
        epc=epc,
        M_mesh=M_mesh,
        q_mesh=q_mesh,
        f_mesh=f_mesh
    )


if __name__ == "__main__":
    main()

