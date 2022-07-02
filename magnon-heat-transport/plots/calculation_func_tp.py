from glob import glob
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
#from scipy import integrate
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../../dqmc_edit/thermaltest/Determinant-QMC/util/")
import util
import maxent

from IPython.display import display_html
display_html("""<button onclick="$('.input, .output_stderr, .output_error').toggle();">Toggle Code</button>""", raw=True)


def loading_tp(U,tp,sitex,sitey,prefix):
    jqjq = np.load("../calculation/tpdata/" + prefix+f"jqjqU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jp0jp0 = np.load("../calculation/tpdata/" +prefix+f"jp0jp0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jk0jk0 = np.load("../calculation/tpdata/" +prefix+f"jk0jk0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jkjk = np.load("../calculation/tpdata/" +prefix+f"jkjkU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jpjp = np.load("../calculation/tpdata/" +prefix+f"jpjpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    j0j0 = np.load("../calculation/tpdata/" +prefix+f"j0j0U{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jqpjqp = np.load("../calculation/tpdata/" +prefix+f"jqpjqpU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jsjs = np.load("../calculation/tpdata/" +prefix+f"jsjsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    C = np.load("../calculation/tpdata/" +prefix+f"CU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    chi = np.load("../calculation/tpdata/" +prefix+f"chiU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    
    jqjqs = np.load("../calculation/tpdata/" +prefix+f"jqjqsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jp0jp0s = np.load("../calculation/tpdata/" +prefix+f"jp0jp0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jk0jk0s = np.load("../calculation/tpdata/" +prefix+f"jk0jk0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jkjks = np.load("../calculation/tpdata/" +prefix+f"jkjksU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jpjps = np.load("../calculation/tpdata/" +prefix+f"jpjpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    j0j0s = np.load("../calculation/tpdata/" +prefix+f"j0j0sU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jqpjqps = np.load("../calculation/tpdata/" +prefix+f"jqpjqpsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    jsjss = np.load("../calculation/tpdata/" +prefix+f"jsjssU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    Cs = np.load("../calculation/tpdata/" +prefix+f"CsU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    chis = np.load("../calculation/tpdata/" +prefix+f"chisU{U:g}tp{tp:g}_{sitex:g}*{sitey:g}.npy")
    
    return jqjq ,jp0jp0,jk0jk0,jkjk,jpjp,j0j0,jqpjqp,jsjs, C,chi, \
jqjqs,jp0jp0s,jk0jk0s,jkjks,jpjps,j0j0s,jqpjqps,jsjss, Cs,chis

def calc_tp(betas,ns,U,tp,sitex,sitey,prefix = "transport_"):
    w = np.load(f"../calculation/wU={U:g}.npy")
    Ajqjq ,Ajp0jp0,Ajk0jk0,Ajkjk,Ajpjp,Aj0j0,Ajqpjqp,Ajsjs,C,chi, \
    jqjqs,jp0jp0s,jk0jk0s,jkjks,jpjps,j0j0s,jqpjqps,jsjss,Cs,chis = \
    loading_tp(U,tp,sitex,sitey,prefix)
    
    Ls = np.zeros(( len(ns),len(betas), 2))
    kappas = np.zeros(( len(ns),len(betas),2))
    
    L1s = np.zeros(( len(ns),len(betas),2))
    kappa1s = np.zeros(( len(ns),len(betas),2))
    
    L2s = np.zeros(( len(ns),len(betas),2))
    kappa2s = np.zeros(( len(ns),len(betas),2))
    
    Ss = np.zeros(( len(ns),len(betas), 2))
    sigmas = np.zeros(( len(ns),len(betas),2))
    sigma_sss = np.zeros(( len(ns),len(betas),2))
    
    Ds = np.zeros(( len(ns),len(betas), 2))
    DQs = np.zeros(( len(ns),len(betas), 2))
    DQ1s = np.zeros(( len(ns),len(betas), 2))
    DQ2s = np.zeros(( len(ns),len(betas), 2))

    Dratios,Dratio1s,Dratio2s = \
    np.zeros(( len(ns),len(betas), 2)), \
    np.zeros(( len(ns),len(betas), 2)), \
    np.zeros(( len(ns),len(betas), 2))
    
    for k, n in enumerate(ns[:]):
        for j,beta in enumerate(betas[:]):
            jqjq ,jp0jp0,jk0jk0,jkjk,jpjp,j0j0,jqpjqp,jsjs = \
            jqjqs[k,j] ,jp0jp0s[k,j],jk0jk0s[k,j],jkjks[k,j],jpjps[k,j],j0j0s[k,j],jqpjqps[k,j],jsjss[k,j]

            
            if tp==0 and n==1:
                ks =(jqjq[:,0])*beta
                ss =  np.zeros(j0j0[:,0].shape)
                ls =( jqjq[:,0])/j0j0[:,0]*beta**2
            else:
                ks =( jqjq[:,0]- ( (jqpjqp[:,0]-jqjq[:,0]-j0j0[:,0])/2)**2/j0j0[:,0])*beta
                ss =  (jqpjqp[:,0]-jqjq[:,0]-j0j0[:,0])/2/j0j0[:,0]*beta
                ls =( jqjq[:,0]- ( (jqpjqp[:,0]-jqjq[:,0]-j0j0[:,0])/2)**2/j0j0[:,0])/j0j0[:,0]*beta**2
                
            Ls[k, j,1] = np.std(ls[:],ddof = 1)
            Ss[k, j,1] = np.std(ss[:],ddof = 1)
            kappas[k, j,1] = np.std(ks[:],ddof = 1)

            ds = j0j0[:,0]/chis[k,j]
            dqs = ks/Cs[k,j]
            
            Ds[k, j,1] = np.std(ds[:],ddof = 1)
            DQs[k, j,1] = np.std(dqs[:],ddof = 1)
            
            
            jkj = (jk0jk0 - jkjk - j0j0)[:,0]/2
            jpj = (jp0jp0 - jpjp - j0j0)[:,0]/2

            if tp==0 and n==1:
                kappa1sspart1 = ( jkjk[:,0])*beta
                kappa2sspart1  = ( jpjp[:,0])*beta
            else:
                kappa1sspart1 = ( jkjk[:,0]-jkj**2/j0j0[:,0])*beta
                kappa2sspart1  = ( jpjp[:,0]-jpj**2/j0j0[:,0])*beta
            part2 = (ks-kappa1sspart1-kappa2sspart1)/2

            l1s =( kappa1sspart1 + part2)/j0j0[:,0]*beta
            l2s =( kappa2sspart1 + part2)/j0j0[:,0]*beta


            L1s[k, j,1]= np.std(l1s[:],ddof = 1)
            L2s[k, j,1]= np.std(l2s[:],ddof = 1)
            kappa1s[k, j,1] = np.std(kappa1sspart1+part2,ddof = 1)
            kappa2s[k, j,1] = np.std(kappa2sspart1+part2,ddof = 1)
            DQ1s[k, j,1] = np.std((kappa1sspart1+part2)/Cs[k,j],ddof = 1)
            DQ2s[k, j,1] = np.std((kappa2sspart1+part2)/Cs[k,j],ddof = 1)
            sigmas[k, j,1] = np.std(j0j0[:,0],ddof = 1)
            sigma_sss[k, j,1]= np.std(jsjs[:,0],ddof = 1)
            
            Dratios[k, j,1] = np.std(dqs[:]/ds[:],ddof = 1)
            Dratio1s[k, j,1] = np.std((kappa1sspart1+part2)/Cs[k,j]/ds[:],ddof = 1)
            Dratio2s[k, j,1] = np.std((kappa2sspart1+part2)/Cs[k,j]/ds[:],ddof = 1)
            
            #########################################

            jqjq ,jp0jp0,jk0jk0,jkjk,jpjp,j0j0,jsjs = \
            Ajqjq[k,j] ,Ajp0jp0[k,j],Ajk0jk0[k,j],Ajkjk[k,j],Ajpjp[k,j],Aj0j0[k,j],Ajsjs[k,j]
            jqpjqp= Ajqpjqp[k,j]

            
            
            if tp==0 and n==1:
                kappas[k, j,0] =  (jqjq[0])*beta
                Ss[k, j,0] = 0
                Ls[k, j,0]= ( jqjq[0])/j0j0[0]*beta**2
            else:
                kappas[k, j,0] =  (jqjq[0]- ( (jqpjqp[0]-jqjq[0]-j0j0[0])/2)**2/j0j0[0])*beta
                Ss[k, j,0] = (jqpjqp[0]-jqjq[0]-j0j0[0])/2/j0j0[0]*beta
                Ls[k, j,0]= ( jqjq[0]- ( (jqpjqp[0]-jqjq[0]-j0j0[0])/2)**2/j0j0[0])/j0j0[0]*beta**2
            jkj = (jk0jk0 - jkjk - j0j0)[0]/2
            jpj = (jp0jp0 - jpjp - j0j0)[0]/2
            if tp==0 and n==1:
                kappa1spart1 = ( jkjk[0]) *beta
                kappa2spart1 = ( jpjp[0]) *beta
            else:
                kappa1spart1 = ( jkjk[0]-jkj**2/j0j0[0]) *beta
                kappa2spart1 = ( jpjp[0]-jpj**2/j0j0[0]) *beta
            part2= (kappas[k, j,0] - kappa1spart1 - kappa2spart1 )/2

            kappa1s[k, j,0] = kappa1spart1+part2
            kappa2s[k, j,0] = kappa2spart1+part2

            L1s[k, j,0]= ( kappa1s[k, j,0] )/j0j0[0]*beta
            L2s[k, j,0]= ( kappa2s[k, j,0] )/j0j0[0]*beta
            sigmas[k, j,0] = j0j0[0]
            sigma_sss[k, j,0] = jsjs[0]
            
            Ds[k, j,0] = sigmas[k, j,0]/chi[k,j]
            DQs[k, j,0] = kappas[k, j,0]/C[k,j]
            DQ1s[k, j,0] = kappa1s[k, j,0]/C[k,j]
            DQ2s[k, j,0] = kappa2s[k, j,0]/C[k,j]

            Dratios[k, j,0] = kappas[k, j,0]/C[k,j]/(sigmas[k, j,0]/chi[k,j])
            Dratio1s[k, j,0] = kappa1s[k, j,0]/C[k,j]/(sigmas[k, j,0]/chi[k,j])
            Dratio2s[k, j,0] = kappa2s[k, j,0]/C[k,j]/(sigmas[k, j,0]/chi[k,j])    
                

    return Ls,kappas,L1s,kappa1s,L2s ,kappa2s,sigmas,sigma_sss,Ss,Ds,DQs,DQ1s,DQ2s,Dratios,Dratio1s,Dratio2s
