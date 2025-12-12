import numpy as np
import scipy.interpolate as sin
from ptarcade.models_utils import prior

Msun = 1.988409870698051e+30
c    = 299792458.0
pc   = 3.085677581491367e+16
G    = 6.6743e-11

# —— 1) Merger-rate density grid & density tensor —— #
rho_all, q_all, z_all = np.load("edgesholo_M40_Q21_Z22.npy", allow_pickle=True)
dN = np.load("ndenholo_M40_Q21_Z22.npy", allow_pickle=True)

# —— 2) Weighted spectrum & e0bar grid —— #
spec = np.load("dEdfgw_recombined_theta_kappa.npz", allow_pickle=True)
new_spec   = spec["new_spec"]      # (15,12,20,39,20,40)
theta_all  = spec["theta_vals"]    # (15,)
k_all      = spec["kappa_vals"]    # (12,)
epc_all    = spec["epc"]           # (20,)
M_all      = spec["M_mesh"]        # (39,)
q_all      = spec["q_mesh"]        # (20,)
fgwr       = spec["f_mesh"]        # (40,)

# —— 3) Original model logic unchanged —— #
def get_SGWB_from_tab(itheta, ik, iepc, fgw=np.logspace(-9, -6, 30), test=False):
    integrand = new_spec[itheta, ik, iepc, :, :, None, :] * dN[:, :, :, None] / (1e6 * pc)**3
    dEdfgw_total_noz = np.trapz(np.trapz(integrand, np.log(M_all), axis=0), q_all, axis=0)

    dEdfgw_total_noz_shifted = np.zeros(z_all.shape + fgw.shape)
    for iz, z in enumerate(z_all):
        f = sin.interp1d(
            np.log10(fgwr / (1 + z)),
            np.log10(dEdfgw_total_noz[iz]),
            fill_value=0
        )
        dEdfgw_total_noz_shifted[iz] = 10**f(np.log10(fgw))

    dEdfgw_total_noz_shifted = np.nan_to_num(dEdfgw_total_noz_shifted, 0)
    dEdfgw_total = np.trapz(dEdfgw_total_noz_shifted, z_all, axis=0)

    hc = np.sqrt(dEdfgw_total * 4 * G / (np.pi * c**2 * fgw))
    return (hc, integrand) if test else hc


def get_SGWB_from_tab_interp(theta, k, epc, fgw, test=False):
    itheta_nearest = np.argsort(np.abs(theta_all - theta))[:2][::-1]
    ik_nearest     = np.argsort(np.abs(k_all - k))[:2][::-1]
    iepc_nearest   = np.argsort(np.abs(epc_all - epc))[:2][::-1]

    theta_nearest = theta_all[itheta_nearest]
    k_nearest     = k_all[ik_nearest]
    epc_nearest   = epc_all[iepc_nearest]

    hc_values = np.zeros((2, 2, 2) + fgw.shape)
    for i, itheta in enumerate(itheta_nearest):
        for j, ik in enumerate(ik_nearest):
            for p, iepc in enumerate(iepc_nearest):
                hc_values[i, j, p] = get_SGWB_from_tab(itheta, ik, iepc, fgw=fgw)

    hc_interp = sin.RegularGridInterpolator(
        (theta_nearest, k_nearest, epc_nearest),
        np.log10(hc_values),
        bounds_error=False,
        fill_value=None
    )
    hc_best_fit = 10**hc_interp((theta, k, epc))

    if test:
        return (
            hc_best_fit,
            hc_values,
            (itheta_nearest, ik_nearest, iepc_nearest),
            (theta_nearest, k_nearest, epc_nearest),
        )
    return hc_best_fit


name0 = "SGWB"
name1 = "_all_alter2"
name = name0 + "_" + name1

parameters = {
    'log10Norm': prior("Normal", -1.4606646938106742, 1.0856704347789061),
    'theta': prior("Uniform", -4, 4),
    'k': prior("Uniform", 0.1, 6),
    'epc': prior("Uniform", 0.0, 0.999)
}
smbhb = False


def spectrum(f, log10Norm, theta, k, epc):
    Norm = 10**log10Norm
    f = np.array(f)

    hc = get_SGWB_from_tab_interp(theta, k, epc, f)[0]
    Omegah2 = hc**2 * f**2 * 2 * np.pi**2 / 3 / (3.240779289444365e-18)**2
    return Omegah2 * Norm






