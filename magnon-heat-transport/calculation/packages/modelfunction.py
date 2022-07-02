import numpy as np
import matplotlib.pyplot as plt

def fitfun_FT(w, a, g1, b, g2, ww):
    return a * np.pi/g1 * 1/np.cosh(np.pi * w/(2*g1)) + b * np.pi/g2 * (0.5/np.cosh(np.pi * (w + ww)/(2*g2)) + 0.5/np.cosh(np.pi * (w - ww)/(2*g2)))

def modelform(w,dw,pars):
    
    inf_val = np.array([2/fitfun_FT(0, *p) for p in pars])
    print(inf_val)

    plt.figure()
    labels = ['jeje','jkjk','jpjp','jj']
    for i in np.arange(4):
        plt.plot(w,fitfun_FT(w, *pars[i])/2,label=labels[i])
        print(fitfun_FT(0,*pars[i])/2)
    plt.legend()
    plt.show()
    
    
    mdljeje = fitfun_FT(w, *pars[0]) * dw
    mdljeje /= mdljeje.sum()
    
    mdljkjk = fitfun_FT(w, *pars[1]) * dw
    mdljkjk /= mdljkjk.sum()
    
    mdljpjp = fitfun_FT(w, *pars[2]) * dw
    mdljpjp /= mdljpjp.sum()
    
    mdljj = fitfun_FT(w, *pars[3]) * dw
    mdljj /= mdljj.sum()
    return (mdljeje,mdljkjk,mdljpjp,mdljj)
    