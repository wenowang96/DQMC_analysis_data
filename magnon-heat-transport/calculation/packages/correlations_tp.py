import numpy as np
import matplotlib.pyplot as plt
import scipy


def get_precs_tp(precalculations,precalculations_eq):
    n_sample, sign, j2j2,jjn,jnj,jnjn,js2js2,U,Ny, Nx, beta, L, tp, bps, b2ps,mu,= precalculations
    n_sample_eq, sign_eq, double_occ, g00, kk, kv, kn, vv, vn, nn = precalculations_eq
    taus = np.linspace(0, beta, L+1)
    mask = np.logical_and((n_sample == n_sample.max()),(n_sample_eq == n_sample_eq.max()))
    print(n_sample.max())
    print(f"{sum(mask)}/{len(mask)} complete bins")
    sign, jjn,jnj,jnjn,j2j2 = sign[mask], jjn[mask],jnj[mask],jnjn[mask], j2j2[mask]
    js2js2 = js2js2[mask]

    jjn.shape = -1, L, bps, b2ps, Ny, Nx
    jnj.shape = -1, L, b2ps, bps, Ny, Nx
    jnjn.shape = -1, L, bps, bps, Ny, Nx
    j2j2.shape = -1, L, b2ps, b2ps, Ny, Nx
    js2js2.shape = -1, L, b2ps, b2ps, Ny, Nx
    
    j2j2q0 = j2j2.sum((-1,-2)) 
    js2js2q0 = js2js2.sum((-1,-2)) 
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
    
    jsjspara = np.outer(parax,parax)
    jsxjsxq0 = - jsjspara * js2js2q0 #"-" comes from i in the current operator
    jsjspara = np.outer(paray,paray)
    jsyjsyq0 = - jsjspara * js2js2q0
    
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
    
    chissq0 = 0.5*(jsxjsxq0 + jsyjsyq0).sum((-1,-2))
    chissq0 = 0.5*(chissq0 + chissq0[:, -np.arange(L) % L])
    
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
    
    return chiq0, chieq0, chieeq0, chineq0, chienq0, chin1q0, chin2q0, chinnq0,chissq0, \
n_sample, sign, U,Ny, Nx, beta, L, tp, bps, b2ps,mu,mask, \
sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn


import sys
sys.path.append("../../../../dqmc_edit/thermaltest/Determinant-QMC/util/")
import util
import maxent
from scipy.interpolate import CubicSpline


def preread_tp(path):
    n_sample, sign, j2j2,jjn,jnj,jnjn,js2js2= util.load(path,
        "meas_uneqlt/n_sample", "meas_uneqlt/sign", "meas_uneqlt/j2j2","meas_uneqlt/jjn", "meas_uneqlt/jnj","meas_uneqlt/jnjn","meas_uneqlt/js2js2")
    U,Ny, Nx, beta, L, tp, bps, b2ps= util.load_firstfile(path,
        "metadata/U", "metadata/Ny", "metadata/Nx", "metadata/beta", "params/L", "metadata/t'","metadata/bps", "metadata/b2ps")
    mu,= util.load_firstfile(path,
           "metadata/mu")
    return  n_sample, sign, j2j2,jjn,jnj,jnjn,js2js2,U,Ny, Nx, beta, L, tp, bps, b2ps,mu

def preread_tp_eq(path):
    n_sample_eq, sign_eq, double_occ, g00, kk, kv, kn, vv, vn, nn = util.load(path,
        "meas_eqlt/n_sample", "meas_eqlt/sign", "meas_eqlt/double_occ", "meas_eqlt/g00",
        "meas_eqlt/kk", "meas_eqlt/kv", "meas_eqlt/kn", "meas_eqlt/vv", "meas_eqlt/vn", "meas_eqlt/nn")
    return  n_sample_eq, sign_eq, double_occ, g00, kk, kv, kn, vv, vn, nn
    
def AnaCon_tp(bs,resamples,mdl,chiq0,krnl,L,beta,taus,sign, w, dw):
    if mdl is None:
        mdl = maxent.model_flat(dw)
        useBT = True
    else:
        useBT = False
    
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
            A = maxent.calc_A(g, krnl, mdl[i] + np.nextafter(0, 1), plot=False, useBT=useBT)
            mdls[i] = A / A.sum()
            As[i] = A / dw * (chiq0w0/sign[resample].mean()) * np.pi/2
            #if A.sum()<0.5:
            #    print(resample)
    else:
        f = chiq0.mean(0)
        chiq0w0 = CubicSpline(taus, np.append(f, f[0])).integrate(0, beta)
        g = 2*chiq0[:, :L//2+1] / chiq0w0

        A = maxent.calc_A(g, krnl, mdl, plot=True, useBT=useBT)
        A, mdl = A / dw * (chiq0w0/sign.mean()) * np.pi/2, A/A.sum()
    if bs>0:
        return As
    else:
        return A   
    
def the_tp(sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn,U, beta,bps,N,tp):
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
    
def thermaldynamics_sample_tp(bs,resamples,sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn ,U, beta,bps,N,tp):
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
            Cs[i],chis[i]  = the_tp(sign_eq1, double_occ1, g001 , kk1, kv1, kn1, vv1, vn1, nn1, U, beta,bps,N,tp)
    else:
        C,chi = the_tp(sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn, U, beta,bps,N,tp)
    if bs>0:
        return Cs,chis
    else:
        return C,chi   

def get_corr(precalculations,precalculations_eq,w,dw,n, 
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs,
             bs=0,monitor=False):
    
    chiq0, chieq0, chieeq0, chineq0, chienq0, chin1q0, chin2q0, chinnq0, chissq0,\
n_sample, sign, U,Ny, Nx, beta, L, tp, bps, b2ps,mu,mask, \
sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn = get_precs_tp(precalculations,precalculations_eq)
    taus = np.linspace(0, beta, L+1)


    nbin = mask.sum()
    if bs>0:
        resamples = np.zeros((bs,nbin))
        #resamples = np.zeros((bs,40))
        for i in range(bs):
            resample = np.random.randint(0, nbin, nbin)
            #resample = np.random.randint(0,nbin, 40)
            resamples[i] = resample
    else:
        resamples=None
    
    # maxent
    chiq0 /= n_sample.max()
    chieq0 /= n_sample.max()
    chieeq0 /= n_sample.max()
    chissq0 /= n_sample.max()
    
    
    chineq0 /= n_sample.max()
    chienq0 /= n_sample.max()
    chin1q0 /= n_sample.max()
    chin2q0 /= n_sample.max()
    chinnq0 /= n_sample.max()

    sign /= n_sample.max()
    print(f"nbin={nbin}\t<sign>={sign.mean()}")
    krnl = maxent.kernel_b(beta, taus[0:L//2+1], w, sym=True)

        
            ####
    print('jqjq')
    chiq0_jqjq = chieeq0 + (-0.5*U-mu)*chieq0 + chienq0 + \
                (-0.5*U-mu)*chieq0 + chiq0*(-0.5*U-mu)**2 + (-0.5*U-mu)*chin2q0 + \
                 chineq0 + (-0.5*U-mu)*chin1q0 +chinnq0
    Ajqjq = AnaCon_tp(bs, resamples,mdljqjq,chiq0_jqjq,krnl,L,beta,taus,sign,w,dw)

        
    
    alphap = -U/2-mu
    alphak = 0 #(-0.5*U-mu)-alphap
    print('jp0jp0')
    chiq0_jp0jp0 = chiq0 + \
                  chin2q0 + alphap*chiq0 + \
                  chin1q0 + alphap*chiq0 + \
                  chinnq0 + alphap*chin2q0 + alphap*chin1q0 +  alphap**2 * chiq0
    A_jp0jp0 = AnaCon_tp(bs, resamples,mdljp0jp0,chiq0_jp0jp0,krnl,L,beta,taus,sign,w,dw)
    print('jk0jk0')
    chiq0_jk0jk0 = chieeq0 + 2*alphak*chieq0 + alphak**2 *  chiq0  + \
                   chieq0  + alphak*  chiq0 + \
                   chieq0 + alphak*  chiq0 + \
                   chiq0
    A_jk0jk0 = AnaCon_tp(bs, resamples,mdljk0jk0,chiq0_jk0jk0,krnl,L,beta,taus,sign,w,dw)
    print('jkjk')    
    chiq0_jkjk = chieeq0 + 2*alphak*chieq0 + alphak**2 *  chiq0 
    A_jkjk = AnaCon_tp(bs, resamples,mdljkjk,chiq0_jkjk,krnl,L,beta,taus,sign,w,dw)
    print('jpjp')  
    chiq0_jpjp = chinnq0 + alphap*chin1q0 + alphap*chin2q0 + alphap**2 * chiq0
    A_jpjp = AnaCon_tp(bs, resamples,mdljpjp,chiq0_jpjp,krnl,L,beta,taus,sign,w,dw)
    print('j0j0')  
    chiq0_j0j0 = chiq0
    A_j0j0 = AnaCon_tp(bs, resamples,mdlj0j0,chiq0_j0j0,krnl,L,beta,taus,sign,w,dw)
    print('jqpjqp')  
    chi_jqj = chieq0 + (chin2q0+chin1q0)/2 + chiq0*(-0.5*U-mu)
    chiq0_jqpjqp = chiq0 + chiq0_jqjq + chi_jqj*2
    A_jqpjqp = AnaCon_tp(bs, resamples,mdljqpjqp,chiq0_jqpjqp,krnl,L,beta,taus,sign,w,dw)
    
    print('jsjs')  
    chiq0_jsjs = chissq0
    A_jsjs = AnaCon_tp(bs, resamples,mdljsjs,chiq0_jsjs,krnl,L,beta,taus,sign,w,dw)
    
    C,chi = thermaldynamics_sample_tp(bs,resamples,sign_eq, double_occ, g00 , kk, kv, kn, vv, vn, nn ,U, beta,bps,Nx*Ny,tp)    
        
    if monitor == True:
        print(chiq0_jqpjqp[:,L//2]/sign)
        for i in range(len(sign)):
            print(i,(chiq0_jqpjqp[:,L//2]/sign)[i])
            
        print(chiq0_j0j0[:,L//2]/sign)
        for i in range(len(sign)):
            print(i,(chiq0_j0j0[:,L//2]/sign)[i])
    
    return Ajqjq,A_jp0jp0,A_jk0jk0,A_jkjk,A_jpjp,A_j0j0,A_jqpjqp,A_jsjs,C,chi



def loading_tp(U,tp,sitex,sitey,prefix):
    jqjq = np.load("tpdata/"+prefix+f"jqjqU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jp0jp0 = np.load("tpdata/"+prefix+f"jp0jp0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jk0jk0 = np.load("tpdata/"+prefix+f"jk0jk0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jkjk = np.load("tpdata/"+prefix+f"jkjkU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jpjp = np.load("tpdata/"+prefix+f"jpjpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    j0j0 = np.load("tpdata/"+prefix+f"j0j0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jqpjqp = np.load("tpdata/"+prefix+f"jqpjqpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jsjs = np.load("tpdata/"+prefix+f"jsjsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    C = np.load("tpdata/"+prefix+f"CU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    chi = np.load("tpdata/"+prefix+f"chiU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    
    jqjqs = np.load("tpdata/"+prefix+f"jqjqsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jp0jp0s = np.load("tpdata/"+prefix+f"jp0jp0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jk0jk0s = np.load("tpdata/"+prefix+f"jk0jk0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jkjks = np.load("tpdata/"+prefix+f"jkjksU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jpjps = np.load("tpdata/"+prefix+f"jpjpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    j0j0s = np.load("tpdata/"+prefix+f"j0j0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jqpjqps = np.load("tpdata/"+prefix+f"jqpjqpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jsjss = np.load("tpdata/"+prefix+f"jsjssU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    Cs = np.load("tpdata/"+prefix+f"CsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    chis = np.load("tpdata/"+prefix+f"chisU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    
    
    return jqjq ,jp0jp0,jk0jk0,jkjk,jpjp,j0j0,jqpjqp,jsjs,C,chi,jqjqs,jp0jp0s,jk0jk0s,jkjks,jpjps,j0j0s,jqpjqps,jsjss,Cs,chis

def saving_tp(U,tp,b,n,i,j,sitex,sitey,bs,w,dw,jqjq,jp0jp0,jk0jk0,jkjk,jpjp,
j0j0,jqpjqp,jsjs,
C,chi,           
jqjqs,jp0jp0s,jk0jk0s,jkjks,jpjps,j0j0s,jqpjqps,jsjss,Cs,chis,prefix,path=None,modelstyle='annealing',monitor=False,mdls=None):
    if path == None:
        path =f"../../../../project_data/thermal_whole/{sitex:g}*{sitey:g}U{U:g}/tp{tp:g}beta{b:g}/n{n:g}/"
    print(path)
    precalculations = preread_tp(path)
    precalculations_eq = preread_tp_eq(path)
    
    if (j==0 and mdls==None) or modelstyle=='all_flat':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        None,None,None,None,None,None,None,None
    elif (j==0 and mdls!=None):
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        mdls
    elif modelstyle=='annealing' or modelstyle=='annealing_single':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        jqjq[i, j-1]*dw/(jqjq[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jp0jp0[i, j-1]*dw/(jp0jp0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jk0jk0[i, j-1]*dw/(jk0jk0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jkjk[i, j-1]*dw/(jkjk[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jpjp[i, j-1]*dw/(jpjp[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        j0j0[i, j-1]*dw/(j0j0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jqpjqp[i, j-1]*dw/(jqpjqp[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jsjs[i, j-1]*dw/(jsjs[i, j-1]*dw).sum()+ np.nextafter(0, 1)
    elif modelstyle=='mix':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        None, \
        None, \
        jk0jk0[i, j-1]*dw/(jk0jk0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jkjk[i, j-1]*dw/(jkjk[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        None, \
        j0j0[i, j-1]*dw/(j0j0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        None, \
        jsjs[i, j-1]*dw/(jsjs[i, j-1]*dw).sum()+ np.nextafter(0, 1) 
    Ajqjq,A_jp0jp0,A_jk0jk0,A_jkjk,A_jpjp,A_j0j0,A_jqpjqp,A_jsjs,C0,chi0 \
    = get_corr(precalculations,precalculations_eq,w,dw,n,
               mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs,monitor=monitor)

    jqjq[i, j] = Ajqjq
    jp0jp0[i, j] = A_jp0jp0
    jk0jk0[i, j] = A_jk0jk0
    jkjk[i, j] = A_jkjk
    jpjp[i, j] = A_jpjp
    j0j0[i, j] = A_j0j0
    jqpjqp[i, j] = A_jqpjqp
    jsjs[i, j] = A_jsjs
    
    C[i, j] =C0
    chi[i, j] = chi0


    ################$$$$$$$
    if (j==0 and mdls==None) or modelstyle=='all_flat':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        None,None,None,None,None,None,None,None
    elif (j==0 and mdls!=None):
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        mdls    
    elif modelstyle=='annealing':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        jqjqs[i, j-1]*dw/np.sum(jqjqs[i, j-1]*dw,axis=1)[:,None], \
        jp0jp0s[i, j-1]*dw/np.sum(jp0jp0s[i, j-1]*dw,axis=1)[:,None], \
        jk0jk0s[i, j-1]*dw/np.sum(jk0jk0s[i, j-1]*dw,axis=1)[:,None], \
        jkjks[i, j-1]*dw/np.sum(jkjks[i, j-1]*dw,axis=1)[:,None], \
        jpjps[i, j-1]*dw/np.sum(jpjps[i, j-1]*dw,axis=1)[:,None], \
        j0j0s[i, j-1]*dw/np.sum(j0j0s[i, j-1]*dw,axis=1)[:,None], \
        jqpjqps[i, j-1]*dw/np.sum(jqpjqps[i, j-1]*dw,axis=1)[:,None], \
        jsjss[i, j-1]*dw/np.sum(jsjss[i, j-1]*dw,axis=1)[:,None]
    elif modelstyle=='annealing_single':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        jqjq[i, j-1]*dw/(jqjq[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jp0jp0[i, j-1]*dw/(jp0jp0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jk0jk0[i, j-1]*dw/(jk0jk0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jkjk[i, j-1]*dw/(jkjk[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jpjp[i, j-1]*dw/(jpjp[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        j0j0[i, j-1]*dw/(j0j0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jqpjqp[i, j-1]*dw/(jqpjqp[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jsjs[i, j-1]*dw/(jsjs[i, j-1]*dw).sum()+ np.nextafter(0, 1)
    elif modelstyle=='mix':
        mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs= \
        None, \
        None, \
        jk0jk0[i, j-1]*dw/(jk0jk0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        jkjk[i, j-1]*dw/(jkjk[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        None, \
        j0j0[i, j-1]*dw/(j0j0[i, j-1]*dw).sum()+ np.nextafter(0, 1), \
        None, \
        jsjs[i, j-1]*dw/(jsjs[i, j-1]*dw).sum()+ np.nextafter(0, 1)
        
    jqjqs[i, j], jp0jp0s[i, j],jk0jk0s[i, j], jkjks[i, j], jpjps[i, j], j0j0s[i, j], jqpjqps[i, j],jsjss[i, j],Cs[i, j], chis[i, j] = \
    get_corr(precalculations,precalculations_eq,w,dw,n,mdljqjq,mdljp0jp0,mdljk0jk0,mdljkjk,mdljpjp,mdlj0j0,mdljqpjqp,mdljsjs,
             bs=bs)
    
    plt.plot(w, jqjqs[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jqjqs$")
    plt.show()


    plt.plot(w, jp0jp0s[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jp0jp0s$")
    plt.show()
    
    plt.plot(w, jk0jk0s[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jk0jk0s$")
    plt.show()
    
    plt.plot(w,jkjks[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jkjks$")
    plt.show()
    
    plt.plot(w, jpjps[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jpjps$")
    plt.show()
    
    plt.plot(w, j0j0s[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, j0j0s$")
    plt.show()
    
    plt.plot(w, jqpjqps[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jqpjqps$")
    plt.show()
    
    plt.plot(w, jsjss[i, j].T, 'r-', alpha=0.25)
    plt.xlim(0, 8.5)
    plt.title(rf"$\langle n \rangle = {n}, \beta={b}, jsjss$")
    plt.show()

    ################
    np.save("tpdata/"+prefix+f"jqjqU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jqjq)
    np.save("tpdata/"+prefix+f"jp0jp0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jp0jp0)
    np.save("tpdata/"+prefix+f"jk0jk0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jk0jk0)
    np.save("tpdata/"+prefix+f"jkjkU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jkjk)
    np.save("tpdata/"+prefix+f"jpjpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jpjp)
    np.save("tpdata/"+prefix+f"j0j0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  j0j0)
    np.save("tpdata/"+prefix+f"jqpjqpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jqpjqp)
    np.save("tpdata/"+prefix+f"jsjsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jsjs)
    np.save("tpdata/"+prefix+f"CU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  C)
    np.save("tpdata/"+prefix+f"chiU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  chi)
    ################$$$$$$$$
    np.save("tpdata/"+prefix+f"jqjqsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jqjqs)
    np.save("tpdata/"+prefix+f"jp0jp0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jp0jp0s)
    np.save("tpdata/"+prefix+f"jk0jk0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jk0jk0s)
    np.save("tpdata/"+prefix+f"jkjksU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jkjks)
    np.save("tpdata/"+prefix+f"jpjpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jpjps)
    np.save("tpdata/"+prefix+f"j0j0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  j0j0s)
    np.save("tpdata/"+prefix+f"jqpjqpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jqpjqps)
    np.save("tpdata/"+prefix+f"jsjssU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  jsjss)
    np.save("tpdata/"+prefix+f"CsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  Cs)
    np.save("tpdata/"+prefix+f"chisU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy",  chis)
#################
