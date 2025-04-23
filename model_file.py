import numpy as np
import scipy.interpolate as sin
from ptarcade.models_utils import prior


Msun = 1.988409870698051e+30
c    = 299792458.0
pc   = 3.085677581491367e+16
G    = 6.6743e-11

M_all,q_all,z_all = np.load("edges_M40_Q21_Z22_holov1p6_SAM_NANO15Astro.npy",allow_pickle=True)
dN      = np.load("nden_M40_Q21_Z22_holov1p6_SAM_NANO15Astro.npy",allow_pickle=True)


rho_all,gamma_all,e0_all,M_all,q_all,fgwr = np.load("dEdfgw_edges_nocut.npy",allow_pickle=True)
dEdfgw_mesh = np.load("dEdfgw_num_nocut.npy",allow_pickle=True)




def get_SGWB_from_tab(irho,igamma,ie0,fgw=np.logspace(-9,-6,30),test=False):

    integrand = dEdfgw_mesh[irho,igamma,ie0,:,:,None,:] * dN[:,:,:,None] / (1e6*pc)**3  
    dEdfgw_total_noz = np.trapz(np.trapz( integrand , np.log(M_all),axis=0) , q_all,axis=0)

    dEdfgw_total_noz_shifted = np.zeros(z_all.shape + fgw.shape )

    for iz,z in enumerate(z_all):
        f = sin.interp1d(np.log10( fgwr / (1+z) ) , np.log10(dEdfgw_total_noz[iz]),fill_value=0)
        dEdfgw_total_noz_shifted[iz] = 10**f(np.log10(fgw))

    dEdfgw_total_noz_shifted = np.nan_to_num(dEdfgw_total_noz_shifted,0)
    dEdfgw_total = np.trapz( dEdfgw_total_noz_shifted , z_all , axis=0)

    hc = np.sqrt( dEdfgw_total * 4  * G / (np.pi * c**2 * fgw) )
    if test == True:
        return hc,integrand
    return hc 

def get_SGWB_from_tab_interp(rho,gamma,e0,fgw,test=False):
    irho_nearest = np.argsort(np.abs(rho_all - rho))[:2][::-1]
    igamma_nearest = np.argsort(np.abs(gamma_all - gamma))[:2][::-1]
    ie0_nearest = np.argsort(np.abs(e0_all - e0))[:2][::-1]

    rho_nearest = rho_all[irho_nearest]
    gamma_nearest = gamma_all[igamma_nearest]
    e0_nearest = e0_all[ie0_nearest]


    hc_values = np.zeros((2, 2, 2) + fgw.shape)
    for i, irho in enumerate(irho_nearest):
        for j, igamma in enumerate(igamma_nearest):
            for k, ie0 in enumerate(ie0_nearest):
                hc_values[i, j, k] = get_SGWB_from_tab(irho, igamma, ie0, fgw=fgw)

                # Interpolation in log scale for rho
    log_rho = np.log10(rho)
    log_rho_nearest = np.log10(rho_nearest)
    
    # Use current scale for gamma and e0
    gamma_nearest = gamma_all[igamma_nearest]
    e0_nearest = e0_all[ie0_nearest]
                
    hc_interp = sin.RegularGridInterpolator(
        (log_rho_nearest, gamma_nearest, e0_nearest),
        np.log10(hc_values),
        bounds_error=False,
        fill_value=None
    )
    
    hc_best_fit = 10**hc_interp((log_rho, gamma, e0))
    if test == True:
        return hc_best_fit , hc_values , (irho_nearest, igamma_nearest, ie0_nearest), (rho_nearest, gamma_nearest, e0_nearest)
    return hc_best_fit 





name0 = "SGWB"
name1 = "all"
name = name0 +"_"+ name1


parameters = {
            'log10Norm' : prior("Normal", -1.4606646938106742 , 1.0856704347789061 ),
            'log10rho' : prior("Uniform", np.log10( rho_all.min() ) , np.log10( rho_all.max() ) ),
            'gammap1' : prior("Uniform" , 0+1 , 2.4+1),
            'e0p1' : prior("Uniform", e0_all.min()+1 , e0_all.max()+1)
            #'e0p1' : prior("Uniform", 1+0.0+0 , 1+0.0+0.001)
            }

smbhb = False


def spectrum( f , log10Norm , log10rho, gammap1 , e0p1 ):

    Norm = 10**log10Norm
    rho = 10**log10rho
    gamma = gammap1 - 1
    e0 = e0p1 - 1
    f = np.array(f)

    hc = get_SGWB_from_tab_interp(rho,gamma,e0,f)[0] 
    #print(f.shape,hc.shape)
   

    Omegah2 = hc**2  * np.array(f)**2 * 2 * np.pi**2 / 3 / (3.240779289444365e-18)**2

    return np.array(Omegah2) * Norm




