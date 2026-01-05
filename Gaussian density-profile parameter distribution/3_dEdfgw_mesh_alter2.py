import numpy as np
from gw_from_binary import *
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import os
import matplotlib.pyplot as plt

_M, _q, _z = np.load("edgesholo_M40_Q21_Z22.npy", allow_pickle=True)
_dN = np.load("ndenholo_M40_Q21_Z22.npy", allow_pickle=True)

ratio_all = np.logspace(-18,18,30) #18,30
e0_all = np.linspace(0,0.999,20) #20
fgwr = np.logspace(-10, -5, 40)
dEdfgw_mesh = np.zeros((len(ratio_all), len(e0_all), len(_M), len(_q), len(fgwr)))

def compute_dEdfgw(iratio, ie0, iMbh, iq, ratio, e0, Mbh, q, fgwr):
    try:
        orbit2 = orbit_2(ratio=ratio, e0=e0, Mbh=Mbh*Msun, q=q, initial_condition="1pc")
        dEdfgw = orbit2.dEdfgw(fgwr)
    except:
        dEdfgw = fgwr*0
    return iratio, ie0, iMbh, iq, dEdfgw


for iratio, ratio in enumerate(ratio_all):
    for ie0, e0 in enumerate(e0_all):
        with joblib_progress(f"{iratio},{ie0}", len(_M) * len(_q)):
            results = Parallel(n_jobs=-1)(
                delayed(compute_dEdfgw)(iratio, ie0, iMbh, iq, ratio, e0, Mbh, q, fgwr)
                for iMbh, Mbh in enumerate(_M)
                for iq, q in enumerate(_q)
            )
        for iratio, ie0, iMbh, iq, dEdfgw in results:
            dEdfgw_mesh[iratio, ie0, iMbh, iq, :] = dEdfgw
            
            
        test = np.trapz(np.trapz(dEdfgw_mesh[iratio,ie0]*_dN[:,:,10,None] , _M , axis=0 ) ,_q , axis=0)
        plt.loglog(fgwr , test)
        plt.savefig(f"figures/test_{iratio}_{ie0}.png",bbox_inches='tight')
        plt.close()
        

    np.save("dEdfgw_num_nocut_alter2",dEdfgw_mesh)
    np.save("dEdfgw_edges_nocut_alter2",np.array([ratio_all,e0_all,_M,_q,fgwr],dtype='object'),allow_pickle=True)

