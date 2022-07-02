import numpy as np
import matplotlib.pyplot as plt
import scipy


def get_precs(precalculations,precalculations_eq):
    n_sample, sign, j2j2,jjn,jnj,jnjn,U,Ny, Nx, beta, L, tp, bps, b2ps,mu,= precalculations
    n_sample_eq, sign_eq, double_occ, g00, kk, kv, kn, vv, vn, nn = precalculations_eq
    taus = np.linspace(0, beta, L+1)
    mask = np.logical_and((n_sample == n_sample.max()),(n_sample_eq == n_sample_eq.max()))
    print(n_sample.max())
    print(f"{sum(mask)}/{len(mask)} complete bins")
    sign, jjn,jnj,jnjn,j2j2 = sign[mask], jjn[mask],jnj[mask],jnjn[mask], j2j2[mask]

    
    if tp==0 and mu==0:
        phsym=True
    else:
        phsym=False

    jjn.shape = -1, L, bps, b2ps, Ny, Nx
    jnj.shape = -1, L, b2ps, bps, Ny, Nx
    jnjn.shape = -1, L, bps, bps, Ny, Nx
    j2j2.shape = -1, L, b2ps, b2ps, Ny, Nx
    
    j2j2q0 = j2j2.sum((-1,-2)) 
    jnjq0 = jnj.sum((-1,-2))
    jjnq0 = jjn.sum((-1,-2))
    jnjnq0 = jnjn.sum((-1,-2))

    
    if tp != 0: 
        parax = np.array([1,0,tp,-tp, 0,0,0,0,0,0,0,0])
        paray = np.array([0,1,tp,tp, 0,0,0,0,0,0,0,0])
        paraex = -np.array([2*tp,0,   1,-1,(1/2+tp*tp)*2,0,2*tp,tp,2*tp*tp/2,-2*tp,-tp,-2*tp*tp/2])
        paraey = -np.array([0,   2*tp,1,1, 0,(1/2+tp*tp)*2,tp,2*tp,2*tp*tp/2,tp,2*tp,  2*tp*tp/2]) 
        paranx = -1/2*np.array([1,0,tp,-tp])*(-U) #(-1) is -t in the hamiltonian , another -1 comes from the prefactor of this term in the thermal current operator
        parany = -1/2*np.array([0,1,tp,tp])*(-U)
    elif tp == 0: 
        parax = np.array([1,0, 0,0,0,0])
        paray = np.array([0,1, 0,0,0,0])
        paraex = -np.array([0,0,1,-1,1,0]) #"-" comes from the definition of the corresponding term in the thermal current, need detailed drivations 
        paraey = -np.array([0,0,1,1, 0,1])
        paranx = -1/2*np.array([1,0])*(-U)
        parany = -1/2*np.array([0,1])*(-U)
    else:
        None
        
    
    jjpara = np.outer(parax,parax)
    jxjxq0 = - jjpara * j2j2q0 #"-" comes from i in the current operator
    jjpara = np.outer(paray,paray)
    jyjyq0 = - jjpara * j2j2q0

    jejpara = 0.5*(np.outer(paraex,parax) + np.outer(parax,paraex))
    jxejxq0 = -  jejpara * j2j2q0
    jejpara = 0.5*(np.outer(paraey,paray) + np.outer(paray,paraey))
    jyejyq0 = -  jejpara * j2j2q0
    
    jejepara = np.outer(paraex,paraex)
    jxejxeq0 = -  jejepara * j2j2q0
    jejepara = np.outer(paraey,paraey)
    jyejyeq0 = -  jejepara * j2j2q0
     
    jnjepara = np.outer(paraex,paranx)
    jnxjexq0 = -  jnjepara * jnjq0
    jnjepara = np.outer(paraey,parany)
    jnyjeyq0 = -  jnjepara * jnjq0
    
    jejnpara = np.outer(paranx,paraex)
    jexjnxq0 = -  jejnpara * jjnq0
    jejnpara = np.outer(parany,paraey)
    jeyjnyq0 = -  jejnpara * jjnq0
    
    jnjpara = np.outer(parax,paranx)
    jnxjxq0 = -  jnjpara * jnjq0
    jnjpara = np.outer(paray,parany)
    jnyjyq0 = -  jnjpara * jnjq0
    
    jjnpara = np.outer(paranx,parax)
    jxjnxq0 = -  jjnpara * jjnq0
    jjnpara = np.outer(parany,paray)
    jyjnyq0 = -  jjnpara * jjnq0
    
    jnjnpara = np.outer(paranx,paranx)
    jnxjnxq0 = -  jnjnpara * jnjnq0
    jnjnpara = np.outer(parany,parany)
    jnyjnyq0 = -  jnjnpara * jnjnq0    
    
    chiq0 = 0.5*(jxjxq0 + jyjyq0).sum((-1,-2))
    chiq0 = 0.5*(chiq0 + chiq0[:, -np.arange(L) % L])
    
    chieq0 = 0.5*(jxejxq0 + jyejyq0).sum((-1,-2))
    chieq0 = 0.5*(chieq0 + chieq0[:, -np.arange(L) % L])
    
    chieeq0 = 0.5*(jxejxeq0 + jyejyeq0).sum((-1,-2))
    chieeq0 = 0.5*(chieeq0 + chieeq0[:, -np.arange(L) % L])
    
    ####
    chineq0 = 0.5*(jnxjexq0+jnyjeyq0).sum((-1,-2))
    chineq0 = 0.5*(chineq0 + chineq0[:, -np.arange(L) % L])
    
    chienq0 = 0.5*(jexjnxq0+jeyjnyq0).sum((-1,-2))
    chienq0 = 0.5*(chienq0 + chienq0[:, -np.arange(L) % L])
    
    chin1q0 = 0.5*(jnxjxq0+jnyjyq0).sum((-1,-2))
    chin1q0 = 0.5*(chin1q0 + chin1q0[:, -np.arange(L) % L])
    
    chin2q0 = 0.5*(jxjnxq0+jyjnyq0).sum((-1,-2))
    chin2q0 = 0.5*(chin2q0 + chin2q0[:, -np.arange(L) % L])
    
    chinnq0 = 0.5*(jnxjnxq0+jnyjnyq0).sum((-1,-2))
    chinnq0 = 0.5*(chinnq0 + chinnq0[:, -np.arange(L) % L])
    
    sign_eq, double_occ, g00 = sign_eq[mask], double_occ[mask], g00[mask]
    kk, kv, kn, vv, vn, nn = kk[mask], kv[mask], kn[mask], vv[mask], vn[mask], nn[mask]

    double_occ.shape = -1
    g00.shape = -1, Ny, Nx
    kk.shape = -1, bps, bps, Ny, Nx
    kv.shape = -1, bps, Ny, Nx
    kn.shape = -1, bps, Ny, Nx
    vv.shape = -1, Ny, Nx
    vn.shape = -1, Ny, Nx
    nn.shape = -1, Ny, Nx
    
    if phsym==True:
        chieq0=0*chieq0
        chin1q0=U/2*chiq0
        chin2q0=U/2*chiq0
    
    
    return chiq0, chieq0, chieeq0, chineq0, chienq0, chin1q0, chin2q0, chinnq0, \
n_sample, sign, U,Ny, Nx, beta, L, tp, bps, b2ps,mu,mask, \
sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn


import sys
sys.path.append("../../../../dqmc_edit/thermaltest/Determinant-QMC/util/")
import util
import maxent
from scipy.interpolate import CubicSpline


def preread(path):
    n_sample, sign, j2j2,jjn,jnj,jnjn= util.load(path,
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/j2j2","meas_uneqlt/jjn", "meas_uneqlt/jnj","meas_uneqlt/jnjn")
    U,Ny, Nx, beta, L, tp, bps, b2ps= util.load_firstfile(path,
        "metadata/U", "metadata/Ny", "metadata/Nx", "metadata/beta", "params/L", "metadata/t'","metadata/bps", "metadata/b2ps")
    mu,= util.load_firstfile(path,
           "metadata/mu")
    return  n_sample, sign, j2j2,jjn,jnj,jnjn,U,Ny, Nx, beta, L, tp, bps, b2ps,mu

def preread_eq(path):
    n_sample_eq, sign_eq, double_occ, g00, kk, kv, kn, vv, vn, nn = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/double_occ", "meas_eqlt/g00",
        "meas_eqlt/kk", "meas_eqlt/kv", "meas_eqlt/kn", "meas_eqlt/vv", "meas_eqlt/vn", "meas_eqlt/nn")
    return  n_sample_eq, sign_eq, double_occ, g00, kk, kv, kn, vv, vn, nn
    
def AnaCon(bs,resamples,mdl,chiq0,krnl,L,beta,taus,sign, w, dw,fixalpha=None,alwaysBT=False):
    if mdl is None:
        mdl = maxent.model_flat(dw)
        useBT = True
    else:
        useBT = False
        
        
    if (alwaysBT == True):
        useBT = True
    
    if bs>0:
        resamples = resamples.astype(int)
    if bs > 0:
        As = np.zeros((bs, len(w)))
        mdls = np.zeros((bs, len(w)))
        if mdl.shape == w.shape:
            mdl = np.broadcast_to(mdl, (bs, len(w)))
        for i in range(bs):
            resample = resamples[i]
            #print(resample)
            f = chiq0[resample].mean(0)
            chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)
            g = 2*chiq0[resample, :L//2+1] / chiq0w0
            if fixalpha==None:
                A = maxent.calc_A(g, krnl, mdl[i] + np.nextafter(0, 1), plot=False, useBT=useBT)
            elif fixalpha!=None:
                A = maxent.calc_A(g, krnl, mdl[i] + np.nextafter(0, 1), plot=False, useBT=True,als=fixalpha)
            mdls[i] = A / A.sum()
            As[i] = A / dw * (chiq0w0/sign[resample].mean()) * np.pi/2
            #if A.sum()<0.5:
            #    print(resample)
    else:
        f = chiq0.mean(0)
        chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)
        g = 2*chiq0[:, :L//2+1] / chiq0w0
        if fixalpha==None:
            A = maxent.calc_A(g, krnl, mdl, plot=True, useBT=useBT)
        if fixalpha!=None:
            A = maxent.calc_A(g, krnl, mdl, plot=True, useBT=True,als=fixalpha)
        A, mdl = A / dw * (chiq0w0/sign.mean()) * np.pi/2, A/A.sum()
    if bs>0:
        return As
    else:
        return A   
    
def the(sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn,U, beta,bps,N,tp):
    # kinetic energy. 2 for spin. negative sign absorbed by anticommutation
    k_nn = 2*(g00[:, 0, 1] + g00[:, 0, -1] + g00[:, 1, 0] + g00[:, -1, 0])
    k_nnn = 2*tp*(g00[:, 1, 1] + g00[:, -1, -1] + g00[:, 1, -1] + g00[:, -1, 1])
    # interaction energy
    v = U*double_occ
    # total energy
    e = k_nn + k_nnn + v
    k = k_nn + k_nnn
    p = v
    
    # density
    den = 2*(sign_eq - g00[:, 0, 0])

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

    EEc = N*eeq0.sum()
    ENc = N*enq0.sum()
    NNc = N*nnq0.sum()
    Ec=N*e.sum()
    Nc=N*den.sum()
    chi_EE = EEc/sign_eq.sum() - (Ec/sign_eq.sum())*(Ec/sign_eq.sum())
    chi_EN = ENc/sign_eq.sum() - (Ec/sign_eq.sum())*(Nc/sign_eq.sum())
    chi_NN = NNc/sign_eq.sum() - (Nc/sign_eq.sum())**2    
    c = (beta*beta/N) * (chi_EE - chi_EN*chi_EN/chi_NN).T 
    chi = (beta/N) * chi_NN
    return c, chi
    
def thermaldynamics_sample(bs,resamples,sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn ,U, beta,bps,N,tp):
    if bs>0:
        resamples = resamples.astype(int)
    if bs > 0:
        Cs = np.zeros(bs)
        chis = np.zeros(bs)
        for i in range(bs):
            resample = resamples[i]
            sign_eq1, double_occ1, g001 , kk1, kv1, kn1, vv1, vn1, nn1 = \
            sign_eq[resample], double_occ[resample], g00[resample] , \
            kk[resample], kv[resample], kn[resample], vv[resample], vn[resample], nn[resample]
            Cs[i],chis[i]  = the(sign_eq1, double_occ1, g001 , kk1, kv1, kn1, vv1, vn1, nn1, U, beta,bps,N,tp)
    else:
        C,chi = the(sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn, U, beta,bps,N,tp)
    if bs>0:
        return Cs,chis
    else:
        return C,chi   

def rebin(a,rebinfactor):
    shapeold = a.shape
    length = (shapeold[0]//rebinfactor)*rebinfactor
    a_shorten = a[:length]
    if len(a.shape)==2:
        shapenew = rebinfactor,shapeold[0]//rebinfactor,shapeold[1]
    elif len(a.shape)==1:
        shapenew = rebinfactor,shapeold[0]//rebinfactor
    anew= a_shorten.reshape(shapenew).mean(0)
    return anew
    
def get_corr(precalculations,precalculations_eq,w,dw,n, 
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,
             bs=0,monitor=False,rebinfactor=1,fixalpha=None,alwaysBT=False):
    
    chiq0, chieq0, chieeq0, chineq0, chienq0, chin1q0, chin2q0, chinnq0, \
n_sample, sign, U,Ny, Nx, beta, L, tp, bps, b2ps,mu,mask, \
sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn = get_precs(precalculations,precalculations_eq)
    taus = np.linspace(0, beta, L+1)


    nbin = mask.sum()

    
    # maxent
    chiq0 /= n_sample.max()
    chieq0 /= n_sample.max()
    chieeq0 /= n_sample.max()
    
    
    chineq0 /= n_sample.max()
    chienq0 /= n_sample.max()
    chin1q0 /= n_sample.max()
    chin2q0 /= n_sample.max()
    chinnq0 /= n_sample.max()

    sign /= n_sample.max()
    
    if rebinfactor !=1:
        chiq0 = rebin(chiq0,rebinfactor)
        chieq0 = rebin(chieq0,rebinfactor)
        chieeq0 = rebin(chieeq0,rebinfactor)
        chineq0 = rebin(chineq0,rebinfactor)
        chienq0 = rebin(chienq0,rebinfactor)
        chin1q0 = rebin(chin1q0,rebinfactor)
        chin2q0 = rebin(chin2q0,rebinfactor)
        chinnq0 = rebin(chinnq0,rebinfactor)

        sign = rebin(sign,rebinfactor)
    print(f"nbin={nbin}\t<sign>={sign.mean()}")
    if rebinfactor !=1:
        print(f"after rebin: nbin={len(sign)}, {sign[0]*n_sample.max()*rebinfactor} samples each bin")
    print(f"{len(sign)*sign[0]*n_sample.max()*rebinfactor} total samples")
    krnl = maxent.kernel_b(beta, taus[0:L//2+1], w, sym=True)

    
    if bs>0:
        resamples = np.zeros((bs,len(sign)))
        for i in range(bs):
            resample = np.random.randint(0, len(sign), len(sign))
            resamples[i] = resample
    else:
        resamples=None
        
            ####
    print('jeje')
    chiq0_jeje = chieeq0 + (-0.5*U)*chieq0 + chienq0 + \
                (-0.5*U)*chieq0 + chiq0*(-0.5*U)**2 + (-0.5*U)*chin2q0 + \
                 chineq0 + (-0.5*U)*chin1q0 +chinnq0
    A_jeje = AnaCon(bs, resamples,mdljeje,chiq0_jeje,krnl,L,beta,taus,sign,w,dw,fixalpha=fixalpha,alwaysBT=alwaysBT)
        
    
    alphap = -U/2*n
    alphak = 0 #(-0.5*U-mu)-alphap
    print('jkjk')    
    chiq0_jkjk = chieeq0 + 2*alphak*chieq0 + alphak**2 *  chiq0 
    A_jkjk = AnaCon(bs, resamples,mdljkjk,chiq0_jkjk,krnl,L,beta,taus,sign,w,dw,fixalpha=fixalpha,alwaysBT=alwaysBT)
    print('jpjp')  
    chiq0_jpjp = chinnq0 + alphap*chin1q0 + alphap*chin2q0 + alphap**2 * chiq0
    A_jpjp = AnaCon(bs, resamples,mdljpjp,chiq0_jpjp,krnl,L,beta,taus,sign,w,dw,fixalpha=fixalpha,alwaysBT=alwaysBT)
    print('j0j0')  
    chiq0_j0j0 = chiq0
    A_j0j0 = AnaCon(bs, resamples,mdlj0j0,chiq0_j0j0,krnl,L,beta,taus,sign,w,dw,fixalpha=fixalpha,alwaysBT=alwaysBT)

    
    C,chi = thermaldynamics_sample(bs,resamples,sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn ,U, beta,bps,Nx*Ny,tp)    
        
    if monitor == True:
            
        print(chiq0_j0j0[:,L//2]/sign)
        for i in range(len(sign)):
            print(i,(chiq0_j0j0[:,L//2]/sign)[i])
    
    return A_jeje,A_jkjk,A_jpjp,A_j0j0,C,chi



def loading(U,tp,sitex,sitey,prefix):
    jeje = np.load(prefix+f"jejeU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jkjk = np.load(prefix+f"jkjkU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jpjp = np.load(prefix+f"jpjpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    j0j0 = np.load(prefix+f"j0j0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    C = np.load(prefix+f"CU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    chi = np.load(prefix+f"chiU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    
    jejes = np.load(prefix+f"jejesU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jkjks = np.load(prefix+f"jkjksU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jpjps = np.load(prefix+f"jpjpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    j0j0s = np.load(prefix+f"j0j0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    Cs = np.load(prefix+f"CsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    chis = np.load(prefix+f"chisU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    
    
    return jeje ,jkjk,jpjp,j0j0,C,chi,jejes,jkjks,jpjps,j0j0s,Cs,chis

def saving(U,tp,b,n,i,j,sitex,sitey,bs,w,dw,jeje,jkjk,jpjp,
j0j0,
C,chi,           
jejes,jkjks,jpjps,j0j0s,Cs,chis,prefix,mdls=None,path=None,modelstyle='annealing',monitor=False,rebinfactor=1,fixalpha=None,alwaysBT=False):
    if path == None:
        path =f"../../../../project_data/thermal_whole/{sitex:g}*{sitey:g}U{U:g}/tp{tp:g}beta{b:g}/n{n:g}/"
    print(path)
    precalculations = preread(path)
    precalculations_eq = preread_eq(path)
    
    if (j==0 and mdls==None) or modelstyle=='all_flat':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        None,None,None,None,
    elif (j==0 and mdls!=None):
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        mdls
    elif modelstyle=='annealing' or modelstyle=='annealing_single':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        jeje[i, j-1]*dw/(jeje[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jkjk[i, j-1]*dw/(jkjk[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jpjp[i, j-1]*dw/(jpjp[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        j0j0[i, j-1]*dw/(j0j0[i, j-1]*dw).sum()+ np.nextafter(0, 1),
    elif modelstyle=='annealing_mean':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        np.mean(jejes[i, j-1]*dw/np.sum(jejes[i, j-1]*dw,axis=1)[:,None],axis=0), \
        np.mean(jkjks[i, j-1]*dw/np.sum(jkjks[i, j-1]*dw,axis=1)[:,None],axis=0), \
        np.mean(jpjps[i, j-1]*dw/np.sum(jpjps[i, j-1]*dw,axis=1)[:,None],axis=0), \
        np.mean(j0j0s[i, j-1]*dw/np.sum(j0j0s[i, j-1]*dw,axis=1)[:,None],axis=0)
    elif modelstyle=='always_model':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        mdls
        
    
    A_jeje,A_jkjk,A_jpjp,A_j0j0,C0,chi0 \
    = get_corr(precalculations,precalculations_eq,w,dw,n,
               mdljeje,mdljkjk,mdljpjp,mdlj0j0,monitor=monitor,
               rebinfactor=rebinfactor,fixalpha=fixalpha,alwaysBT=alwaysBT)

    jeje[i, j] = A_jeje
    jkjk[i, j] = A_jkjk
    jpjp[i, j] = A_jpjp
    j0j0[i, j] = A_j0j0
    
    C[i, j] =C0
    chi[i, j] = chi0

    ################
    np.save(prefix+f"jejeU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jeje)
    np.save(prefix+f"jkjkU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jkjk)
    np.save(prefix+f"jpjpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jpjp)
    np.save(prefix+f"j0j0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  j0j0)
    np.save(prefix+f"CU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  C)
    np.save(prefix+f"chiU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  chi)

    ################$$$$$$$
    if (j==0  and mdls==None) or modelstyle=='all_flat':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        None,None,None,None
    elif (j==0 and mdls!=None):
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        mdls
    elif modelstyle=='annealing':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        jejes[i, j-1]*dw/np.sum(jejes[i, j-1]*dw,axis=1)[:,None], \
        jkjks[i, j-1]*dw/np.sum(jkjks[i, j-1]*dw,axis=1)[:,None], \
        jpjps[i, j-1]*dw/np.sum(jpjps[i, j-1]*dw,axis=1)[:,None], \
        j0j0s[i, j-1]*dw/np.sum(j0j0s[i, j-1]*dw,axis=1)[:,None]
    elif modelstyle=='annealing_single':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        jeje[i, j-1]*dw/(jeje[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jkjk[i, j-1]*dw/(jkjk[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jpjp[i, j-1]*dw/(jpjp[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        j0j0[i, j-1]*dw/(j0j0[i, j-1]*dw).sum()+ np.nextafter(0, 1)
    elif modelstyle=='annealing_mean':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        np.mean(jejes[i, j-1]*dw/np.sum(jejes[i, j-1]*dw,axis=1)[:,None],axis=0), \
        np.mean(jkjks[i, j-1]*dw/np.sum(jkjks[i, j-1]*dw,axis=1)[:,None],axis=0), \
        np.mean(jpjps[i, j-1]*dw/np.sum(jpjps[i, j-1]*dw,axis=1)[:,None],axis=0), \
        np.mean(j0j0s[i, j-1]*dw/np.sum(j0j0s[i, j-1]*dw,axis=1)[:,None],axis=0)
    elif modelstyle=='always_model':
        mdljeje,mdljkjk,mdljpjp,mdlj0j0,= \
        mdls
        
        
    jejes[i, j],jkjks[i, j], jpjps[i, j], j0j0s[i, j], Cs[i, j], chis[i, j] = \
    get_corr(precalculations,precalculations_eq,w,dw,n,mdljeje,mdljkjk,mdljpjp,mdlj0j0,
             bs=bs,rebinfactor=rebinfactor,fixalpha=fixalpha,alwaysBT=alwaysBT)
    
    def kernelatbetaover2(w,beta0):
        return w*np.cosh(0)/np.sinh(w*beta0/2)*beta0/2
    
    
    plt.plot(w, jejes[i, j].T, 'r-', alpha=0.25)
    plt.plot(w, jeje[i, j], 'k-', alpha=1)
    plt.plot(w, jeje[i, j,0]*kernelatbetaover2(w,b), 'b--', alpha=0.5)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jejes$")
    plt.show()
    
    plt.plot(w, jkjks[i, j].T, 'r-', alpha=0.25)
    plt.plot(w, jkjk[i, j], 'k-', alpha=1)
    plt.plot(w, jkjk[i, j,0]*kernelatbetaover2(w,b), 'b--', alpha=0.5)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jkjks$")
    plt.show()
    
    plt.plot(w, jpjps[i, j].T, 'r-', alpha=0.25)
    plt.plot(w, jpjp[i, j], 'k-', alpha=1)
    plt.plot(w, jpjp[i, j,0]*kernelatbetaover2(w,b), 'b--', alpha=0.5)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jpjps$")
    plt.show()
    
    plt.plot(w, j0j0s[i, j].T, 'r-', alpha=0.25)
    plt.plot(w, j0j0s[i, j].T, 'r-', alpha=0.25)
    plt.plot(w, j0j0[i, j], 'k-', alpha=1)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, j0j0s$")
    plt.show()


    ################$$$$$$$$
    np.save(prefix+f"jejesU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jejes)
    np.save(prefix+f"jkjksU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jkjks)
    np.save(prefix+f"jpjpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jpjps)
    np.save(prefix+f"j0j0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  j0j0s)
    np.save(prefix+f"CsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  Cs)
    np.save(prefix+f"chisU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  chis)
#################

