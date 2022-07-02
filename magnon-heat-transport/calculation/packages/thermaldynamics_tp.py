import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import sys
sys.path.append("../../../../dqmc_edit/thermaltest/Determinant-QMC/util/")
import util

def loading_thermaldynamics_tp(path):
    U, tp, Ny, Nx, bps, beta = util.load_firstfile(path,
        "metadata/U", "metadata/t'", "metadata/Ny", "metadata/Nx", "metadata/bps", "metadata/beta")
    N = Ny*Nx
    n_sample, sign, double_occ, g00, kk, kv, kn, vv, vn, nn,zz = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/double_occ", "meas_eqlt/g00",
        "meas_eqlt/kk", "meas_eqlt/kv", "meas_eqlt/kn", "meas_eqlt/vv", "meas_eqlt/vn", "meas_eqlt/nn","meas_eqlt/zz")
    N_SAMPLE, = util.load(path,"meas_uneqlt/n_sample")
    mask = np.logical_and((n_sample == n_sample.max()),(N_SAMPLE == N_SAMPLE.max()))
    print(f"{sum(mask)}/{len(mask)} complete bins")
    sign, double_occ, g00 = sign[mask], double_occ[mask], g00[mask]
    kk, kv, kn, vv, vn, nn = kk[mask], kv[mask], kn[mask], vv[mask], vn[mask], nn[mask]
    zz= zz[mask]

    double_occ.shape = -1
    g00.shape = -1, Ny, Nx
    kk.shape = -1, bps, bps, Ny, Nx
    kv.shape = -1, bps, Ny, Nx
    kn.shape = -1, bps, Ny, Nx
    vv.shape = -1, Ny, Nx
    vn.shape = -1, Ny, Nx
    nn.shape = -1, Ny, Nx
    zz.shape = -1, Ny, Nx
    
    return U, tp, Ny, Nx, bps, beta, N, sign, double_occ, g00, kk, kv, kn, vv, vn, nn,zz

def precalculations_tp(path):
    U, tp, Ny, Nx, bps, beta, N, sign, double_occ, g00, kk, kv, kn, vv, vn, nn, zz = loading_thermaldynamics_tp(path)
    
    k_nn = 2*(g00[:, 0, 1] + g00[:, 0, -1] + g00[:, 1, 0] + g00[:, -1, 0])
    k_nnn = 2*tp*(g00[:, 1, 1] + g00[:, -1, -1] + g00[:, 1, -1] + g00[:, -1, 1])
    # interaction energy
    v = U*double_occ
    # total energy
    e = k_nn + k_nnn + v
    k = k_nn + k_nnn
    p = v
    
    # density
    den = 2*(sign - g00[:, 0, 0])

    # energy-energy
    kkq0 = kk.sum((-1, -2))
    kvq0 = kv.sum((-1, -2))
    vvq0 = vv.sum((-1, -2))
    if bps == 4:
        kkq0[:, :, 2:] *= tp
        kkq0[:, 2:, :] *= tp
        kvq0[:, 2:] *= tp
    kkq0 = kkq0.sum((1, 2))
    kvq0 *= -1 * U
    kvq0 = kvq0.sum(1)
    vvq0 *= U*U
    eeq0 = kkq0 + 2*kvq0 + vvq0
    keq0 = kkq0 + kvq0 
    peq0 = kvq0 + vvq0

    # energy-density
    knq0 = kn.sum((-1, -2))
    if bps == 4:
        knq0[:, 2:] *= tp
    knq0 *= -1
    knq0 = knq0.sum(1)
    vnq0 = vn.sum((-1, -2))
    vnq0 *= U
    enq0 = knq0 + vnq0

    
    # density-density
    nnq0 = nn.sum((-1, -2))
    
    
    # spin-spin
    zzq0 = zz.sum((-1, -2))

    return U, beta, N, sign, e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0
    

def calc_C_tp(sign, K, E,N, KE, EN,KN, NN):
    chi_KE = KE/sign - (E/sign)*(K/sign)
    chi_KN = KN/sign - (N/sign)*(K/sign)
    chi_EN = EN/sign - (E/sign)*(N/sign)
    chi_NN = NN/sign - (N/sign)**2    
    return (chi_KE - chi_EN*chi_KN/chi_NN).T 

# heat capacity per unit cell
def get_c_separate_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    ck= (beta*beta/N) * util.jackknife(sign, N*k,N*e, N*den, N*keq0, N*enq0, N*knq0,N*nnq0, f=calc_C_tp)
    cp= (beta*beta/N) * util.jackknife(sign, N*p,N*e, N*den, N*peq0, N*enq0, N*vnq0,N*nnq0, f=calc_C_tp)
    c = (beta*beta/N) * util.jackknife(sign, N*e, N*e, N*den, N*eeq0, N*enq0, N*enq0,N*nnq0, f=calc_C_tp)
    return ck, cp,c 



def get_density_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    return util.jackknife(sign, den)



# energy per unit cell
def get_e_separate_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    return  util.jackknife(sign, k), \
            util.jackknife(sign, p)/U, \
    util.jackknife(sign, e)

def calc_compressibility_tp(eqsign, N, NN):
    chi_NN = NN/eqsign - (N/eqsign)**2    
    return (chi_NN).T


def get_compressibility_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    return (beta/N) * util.jackknife(sign, N*den, N*nnq0,f=calc_compressibility_tp)


def calc_spinsusc_tp(eqsign, zz):
    chi_zz = zz/eqsign
    return (chi_zz).T


def get_spinsusc_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    return (beta/N) * util.jackknife(sign, N*zzq0,f=calc_spinsusc_tp)

def calc_spinwilsonratio_tp(sign, K, E,N, KE, EN,KN, NN,zz):
    chi_KE = KE/sign - (E/sign)*(K/sign)
    chi_KN = KN/sign - (N/sign)*(K/sign)
    chi_EN = EN/sign - (E/sign)*(N/sign)
    chi_NN = NN/sign - (N/sign)**2    
    chi_zz = zz/sign
    return ( chi_zz / (chi_KE - chi_EN*chi_KN/chi_NN) ).T

def get_spinwilsonratio_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    ratio =  4*np.pi**2/3*(beta)/(beta**3) * util.jackknife(sign, N*e, N*e, N*den, N*eeq0, N*enq0, N*enq0,N*nnq0,N*zzq0, f=calc_spinwilsonratio_tp)
    return ratio


def calc_chargewilsonratio_tp(sign, K, E,N, KE, EN,KN, NN):
    chi_KE = KE/sign - (E/sign)*(K/sign)
    chi_KN = KN/sign - (N/sign)*(K/sign)
    chi_EN = EN/sign - (E/sign)*(N/sign)
    chi_NN = NN/sign - (N/sign)**2    
    return ( chi_NN / (chi_KE - chi_EN*chi_KN/chi_NN) ).T

def get_chargewilsonratio_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0):
    ratio =  np.pi**2/3*(beta)/(beta**3) * util.jackknife(sign, N*e, N*e, N*den, N*eeq0, N*enq0, N*enq0,N*nnq0, f=calc_chargewilsonratio_tp)
    return ratio


def get_all_tp(path):
    U, beta, N, sign, e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0 = precalculations_tp(path)
    density       = get_density_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)
    ck,cp,cwhole  = get_c_separate_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)
    ek,ep,ewhole  = get_e_separate_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)    
    spinsusc   = get_spinsusc_tp(U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)
    compressibility   = \
    get_compressibility_tp(      U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)
    chargewilsonratio = \
    get_chargewilsonratio_tp(    U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)
    spinwilsonratio   = \
    get_spinwilsonratio_tp(      U, beta, N, sign,e, k, p, den,kkq0,kvq0,vvq0,eeq0,keq0,peq0,knq0,vnq0,enq0,nnq0,zzq0)
    return density,ck,cp,cwhole,ek,ep,ewhole,spinsusc,compressibility,chargewilsonratio,spinwilsonratio