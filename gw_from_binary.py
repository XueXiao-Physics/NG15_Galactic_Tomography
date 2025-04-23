import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import jn
import scipy.integrate as si
import scipy.interpolate as sin
import scipy.constants as sc
import astropy.constants as ac
import time 
from astropy.io import fits
import warnings

#np.seterr(all='raise')
warnings.filterwarnings("ignore", category=RuntimeWarning) 

Msun = 1.988409870698051e+30
c    = 299792458.0
pc   = 3.085677581491367e+16
G    = 6.6743e-11
emax = 1-1e-16
emin = 0


def gn(n,e):
    #e = np.clip(e, emin, emax)
    ne = n*e
    j_n = jn(n,ne)
    j_nm2 = jn(n-2,ne)
    j_np2 = jn(n+2,ne)

    gn1 = j_nm2 - 2*e*jn(n-1,ne) + 2/n*j_n + 2*e*jn(n+1,ne) - j_np2
    gn2 = j_nm2 - 2*j_n + j_np2
    gn3 = j_n

    return np.abs( n**4 / 32 * (  gn1**2 + (1-e)*(1+e) * gn2**2 + 4/3/n**2 * gn3**2  ) )


def nmax(e):
    #e = np.clip(e, emin, emax)
    c1 = -1.01678
    c2 = 5.57372
    c3 = -4.9271
    c4 = 1.68506
    
    npeak = (2 * ( 1 + c1 * e + c2 * e**2 + c3 * e**3 + c4 * e**4 ) * ((1-e)*(1+e))**(-3/2) )
    return 3*npeak

def rinf(Mbh , rho1pc, r1pc , gamma):
    if rho1pc ==0:
        return np.inf
    else:
        res = np.power( (3 - gamma) * Mbh * r1pc**(-gamma) / ( 2 * np.pi * rho1pc) , 1/(3-gamma))
        return res


def Fe_We(e):
    result = (1 + (73/24) * e**2 + (37/96) * e**4) 
    return result

def Ge_We(e):
    result = e*(1 + 121/304*e**2) * (1-e**2)
    return result

def We(e):
    result = (1-e**2)**(7/2)
    return result


class orbit():

    def __init__(self, Mbh = 1e9 * Msun , q = 0.9 , rho_multiplier = 1 , gamma = 2 , \
                 initial_condition = "influence_radius"  , e0 = 0.999 , reverse_integration = 1):


        self.Mbh = Mbh
        self.q = q
        self.H = 18
        self.rho_multiplier = rho_multiplier
        self.rho1pc = 1e5 * Msun/pc**3 * self.rho_multiplier #26339*Msun/pc**3 * self.rho_multiplier
        self.r1pc   = 1 * pc


        # initial Condition
        self.e0 = e0
        if initial_condition == "influence_radius": # Dynamic hardening radius based on the powerlaw density profile
            self.r_init = rinf(Mbh , self.rho1pc , self.r1pc , gamma )
            
        elif initial_condition == "1pc": # Force the harding radius at 1pc
            self.r_init = 1*pc
            
        elif initial_condition == "mixed":
            self.r_init = min(1*pc , rinf(Mbh , self.rho1pc , self.r1pc , gamma ))
        else:
            raise

        # Relevant properties
        self.sig_init  = np.sqrt( G * Mbh / self.r_init )
        self.rho_init  = self.rho1pc * ( self.r_init / self.r1pc ) ** (-gamma)
        self.rhosig_init = self.rho_init / self.sig_init

        self.fac1 = ( G * Mbh ) **3  / c**5 
        self.fac2 = G * self.rho_init / self.sig_init
        self.eta = q / (1+q)**2
        self.ah = self.r_init * self.eta / 4

        # Integration
        rmin = 1e-6 * pc
        rmax = self.r_init

        if rmin >= rmax:
            raise ValueError
        

        dlog10r = 0.03
        ndlog10r = (np.log10(rmax) - np.log10(rmin)) / dlog10r
        t_eval = 10**( np.arange(ndlog10r)[1:-1] * dlog10r + np.log10(rmin) )

        self.dedloga_res  = si.solve_ivp( self.dedloga , [ np.log(rmax) , np.log(rmin) ]  ,  [self.e0] , \
                                         t_eval = np.log(t_eval[::-1]) , atol = 1e-7 , rtol=1e-7)
        a_all = np.exp(self.dedloga_res.t[::-1])
        e_all = self.dedloga_res.y[0][::-1]

        if reverse_integration > 1:
            ndlog10r = (np.log10(rmax*reverse_integration) - np.log10(rmax)) / dlog10r
            t_eval   = 10**( np.arange(ndlog10r)[1:-1] * dlog10r + np.log10(rmax) )
            self.dedloga_res_rev  = si.solve_ivp( self.dedloga , [ np.log(rmax) , np.log(rmax*reverse_integration) ]  ,  [self.e0] , \
                                                 t_eval = np.log(t_eval)  , atol = 1e-7 , rtol=1e-7)
            a_all = np.concatenate([ a_all , np.exp(self.dedloga_res_rev.t)])
            e_all = np.concatenate([ e_all , self.dedloga_res_rev.y[0] ])

        




       # form forb to a and e
        forb_all = self.forb_a( a_all )
        self.a_all = a_all
        self.e_all = e_all
        #self.e_all[self.e_all>=0.999999] = 0.999999

        self.forb_min = forb_all.min()
        self.forb_max = forb_all.max()
        self.forb_dense = np.logspace( np.log10(self.forb_min) , np.log10(self.forb_max),30000 )

        self.e_forb = lambda forb : np.interp( np.log10(forb) , np.log10(forb_all[::-1]) , e_all[::-1])
        #self.e_forb = lambda forb : np.interp( forb , self.forb_dense , self.deda_res.sol(self.a_forb(self.forb_dense))[0] )



    
    def forb_a(self,a): # here forb is in Hz
        return np.sqrt( G * self.Mbh / (a)**3 ) / 2 / np.pi

    def a_forb(self,forb):
        return ( G * self.Mbh / ( 2 * np.pi * forb)**2 )**(1/3)

    def Ke(self,e):
        e = np.asarray(e)
        result = np.full_like(e, np.nan, dtype=np.float64)
        mask = (e >= 0) & (e < 1)
        result[mask] = 0.3 * e[mask] * (1-e[mask]**2)**0.6 
        return result
    
    def Ka(self,a):
        return (1 + a / 0.2 / self.ah )**( -1 )

    def dadt_gw_We(self,a,e):
        if np.any(a <= 0):
            return np.full_like(a, np.nan) if isinstance(a, np.ndarray) else np.nan
        return -64 / 5  * self.fac1 / a**3 * self.eta * Fe_We(e) 
    
    def dadt_3b_We(self,a,e):
        return - self.H * self.fac2 * a**2 * We(e)
    
    def dedt_gw_We(self,a,e):
        if np.any(a <= 0):
            return np.full_like(a, np.nan) if isinstance(a, np.ndarray) else np.nan
        return -304 / 15 * self.fac1 / a**4 * self.eta * Ge_We(e) 
    
    def dedt_3b_We(self,a,e):
        return + self.H * self.Ka(a) * self.Ke(e) * self.fac2 * a  * We(e)

    def deda(self,a,e):
        if e<0 or e>1:
            return np.nan
        else:
            dedt_gw_We = self.dedt_gw_We(a,e)
            dedt_3b_We = self.dedt_3b_We(a,e)
            dadt_gw_We = self.dadt_gw_We(a,e)
            dadt_3b_We = self.dadt_3b_We(a,e)
            return (dedt_gw_We + dedt_3b_We) / (dadt_gw_We + dadt_3b_We)

    def dedloga(self,loga,e):
        a = np.exp(loga)
        return self.deda(a,e) * a
    




    #dfdt, but we need to know a(e)
    def dforbr_dt_forbr(self,forbr):
        a = self.a_forb(forbr)
        e = self.e_forb(forbr)
        dadt = ( self.dadt_gw_We(a,e) + self.dadt_3b_We(a,e) ) / We(e)
        prefac = - a**(-5/2) * 3* np.sqrt( G * self.Mbh ) / 4 / np.pi
        return dadt * prefac





    # GW emission
    def dEdfgw( self , fgwr_list , ncut=100  ):
        nmax_global = nmax(self.e_all.max())
        if nmax_global <= ncut:
            n = np.arange(1,nmax_global+1)
        else:
            n1 = np.arange(1,ncut+1)
            n2 = 2**np.arange(0,np.log(nmax_global-ncut)/np.log(2)) + ncut
            n = np.concatenate( [n1,n2] )

        forbr = fgwr_list[:,None] / n[None,:]
        mask = ( forbr <= self.forb_max ) * ( forbr >= self.forb_min )
        e = self.e_forb(forbr)

        
        # for small n
        gne_tab = gn(n[None,:],e)
        dEdt = 32/5 * c**5 / G * (G * self.Mbh * 2 * np.pi * forbr / c**3 )**(10/3) * self.eta**2 * gne_tab * mask
        dforb_dt = self.dforbr_dt_forbr(forbr)

        dEdfgw = dEdt / ( dforb_dt * n[None,:] )

        # sum up
        #print(nmax_global)
        if nmax_global <= ncut:
            dEdfgw_total = np.sum(dEdfgw,axis=1) 
        else:
            dEdfgw_total = np.sum(dEdfgw[:,:ncut],axis=1) 
            dEdfgw_total += np.trapz( dEdfgw[:,ncut:] , n2 , axis=1 )



        return dEdfgw_total
    
    
 