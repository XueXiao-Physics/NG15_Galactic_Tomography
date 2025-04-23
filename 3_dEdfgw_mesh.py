import numpy as np
from gw_from_binary import *
from joblib import Parallel, delayed
from joblib_progress import joblib_progress


_M,_q,_z = np.load("edges_M40_Q21_Z22_holov1p6_SAM_NANO15Astro.npy",allow_pickle=True)
_dN      = np.load("nden_M40_Q21_Z22_holov1p6_SAM_NANO15Astro.npy",allow_pickle=True)



rho_all = np.logspace(-4,4,18)
gamma_all = np.linspace(0,2.4,19)
e0_all = np.linspace(0,0.999,20)
fgwr = np.logspace(-10, -5, 40)
dEdfgw_mesh = np.zeros((len(rho_all), len(gamma_all), len(e0_all), len(_M), len(_q), len(fgwr)))





def compute_dEdfgw(irho, igamma, ie0, iMbh, iq, rho, e0, gamma, Mbh, q, fgwr):
    try:
        orbit1 = orbit(rho_multiplier=rho, gamma=gamma, e0=e0, Mbh=Mbh*Msun, q=q, initial_condition="influence_radius")
        dEdfgw = orbit1.dEdfgw(fgwr)
    except:
        dEdfgw = fgwr*0
    return irho, igamma, ie0, iMbh, iq, dEdfgw

for irho, rho in enumerate(rho_all):
    for igamma, gamma in enumerate(gamma_all):
        for ie0, e0 in enumerate(e0_all):
            with joblib_progress(f"{irho},{igamma},{ie0}", len(_M) * len(_q)):
                results = Parallel(n_jobs=-1)(
                    delayed(compute_dEdfgw)(irho, igamma, ie0, iMbh, iq, rho, e0, gamma, Mbh, q, fgwr)
                    for iMbh, Mbh in enumerate(_M)
                    for iq, q in enumerate(_q)
                )
            for irho, igamma, ie0, iMbh, iq, dEdfgw in results:
                dEdfgw_mesh[irho, igamma, ie0, iMbh, iq, :] = dEdfgw
            
            test = np.trapz(np.trapz(dEdfgw_mesh[irho,igamma,ie0]*_dN[:,:,10,None] , _M , axis=0 ) ,_q , axis=0)
            plt.loglog(fgwr , test)
            plt.savefig(f"3_figures/test_{irho}_{igamma}_{ie0}.png",bbox_inches='tight')
            plt.close()


        
    np.save("dEdfgw_num_nocut",dEdfgw_mesh)
    np.save("dEdfgw_edges_nocut",np.array([rho_all,gamma_all,e0_all,_M,_q,fgwr],dtype='object'),allow_pickle=True)