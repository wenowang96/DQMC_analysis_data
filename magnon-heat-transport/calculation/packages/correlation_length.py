import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import sys
sys.path.append("../../../../dqmc_edit/thermaltest/Determinant-QMC/util/")
import util
    
def zz_realspace(path):
    U, tp, Ny, Nx, bps, beta = util.load_firstfile(path,
        "metadata/U", "metadata/t'", "metadata/Ny", "metadata/Nx", "metadata/bps", "metadata/beta")
    N = Ny*Nx
    n_sample, sign, zz = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/zz")
    N_SAMPLE, = util.load(path,"meas_uneqlt/n_sample")
    mask = np.logical_and((n_sample == n_sample.max()),(N_SAMPLE == N_SAMPLE.max()))
    print(f"{sum(mask)}/{len(mask)} complete bins")
    sign, = sign[mask],
    zz= zz[mask]
    
    zz.shape = -1, Ny, Nx
    zz_sym = 0.5 * (zz + zz[..., (-np.arange(Ny)) % Ny, :])
    zz_sym2 = 0.5 * (zz_sym + zz_sym[..., :, (-np.arange(Nx)) % Nx])
    zz0 = zz_sym2.copy()
    zzt = np.transpose(zz_sym2, (0, 2, 1))
    if Nx==Ny:
        zz = 0.5 * (zz0 + zzt)
    else:
        zz = zz0

    zz = util.jackknife(sign, zz)
    return zz
    
    
    
def correlation_length_alongx(zz,PLOT=True,start=None,end=None):
    Ny,Nx = zz.shape[-2],zz.shape[-1]
    def exponential_x(x,a,b):
        y = a*(np.exp(-x/b)+np.exp(-(Nx-x)/b))
        return y

    if (   ( zz[0,0,:Nx//2+1]*(-1)**np.arange(Nx//2+1)<0 ).any() ):
        return None
    poptx, pcov = curve_fit(exponential_x, np.arange(Nx//2+1)[start:end],(zz[0,0,:Nx//2+1]*(-1)**np.arange(Nx//2+1))[start:end])
    if PLOT == True:
        plt.errorbar(np.arange(Nx//2+1),
                     np.log(zz[0,0,:Nx//2+1]*(-1)**np.arange(Nx//2+1)),
                     zz[1,0,:Nx//2+1]*0,label='origin')
        
        plt.show()
        
        
        plt.plot(np.arange(Nx//2+1),exponential_x(np.arange(Nx//2+1),poptx[0],poptx[1]),label='fit')
        plt.errorbar(np.arange(Nx//2+1),zz[0,0,:Nx//2+1]*(-1)**np.arange(Nx//2+1),
                     zz[1,0,:Nx//2+1],label='origin')
        plt.legend()
        plt.ylim(0,zz[0,0,start]*(-1)**start*1.2)
        plt.show()
    
    return poptx[1]

def save_spincorrelation(U,sitex,sitey,tp,betas,pathpre,start=None):
    zzreal = np.zeros((len(betas),2,sitey,sitex))
    if start!= None:
        zzreal = np.load(f'zzreal_{sitex:g}*{sitey:g}U{U:g}_tp{tp:g}.npy')
    for j, b in enumerate(betas):
        if start!=None:
            if b<start:
                continue
        path =  pathpre+f"tp{tp:g}beta{b:g}/n1/"
        print(path)
        zzreal[j] = zz_realspace(path)
        np.save(f'zzreal_{sitex:g}*{sitey:g}U{U:g}_tp{tp:g}.npy',zzreal)