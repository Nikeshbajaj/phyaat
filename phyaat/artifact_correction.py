'''Artifact Removal Algorithms.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ICA_methods import ICA

#from matplotlib.gridspec import GridSpec
from scipy.stats import kurtosis, skew
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from . import utils


def RemoveArtftICA_CBI_Kur_Iso(X,winsize=128,CorrP=0.8,KurThr=2,ICAMed = 'extended-infomax',verbose=True,
                              window=['hamming',True],hopesize=None,winMeth='custom'):
    '''
    ICAMed = ['fastICA','infomax','extended-infomax','picard']
    
    '''
    win = np.arange(winsize)
    #XR =[]
    nch = X.shape[1]
    #nSeg = X.shape[0]//winsize
    if hopesize is None: hopesize=winsize//2
    if verbose:
        print('ICA Artifact Removal : '+ICAMed)
        
    if winMeth is None:
        xR = _RemoveArtftICA_CBI_Kur_Iso(X,winsize=winsize,CorrP=CorrP,KurThr=KurThr,ICAMed = ICAMed,verbose=verbose)
        
    elif winMeth =='custom':
        M   = winsize
        H   = hopesize
        hM1 = (M+1)//2
        hM2 = M//2
        
        Xt  = np.vstack([np.zeros([hM2,nch]),X,np.zeros([hM1,nch])])
        
        pin  = hM1
        pend = Xt.shape[0]-hM1
        wh   = get_window(window[0],M)

        if len(window)>1: AfterApply = window[1] 
        else: AfterApply =False
        xR   = np.zeros(Xt.shape)
        
        while pin<=pend:
            if verbose:
                utils.ProgBar_float(pin,N=pend,title='',style=2,L=50)
                #pf = pin*100.0/float(pend)
                #pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
                #print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
                
            
            xi = Xt[pin-hM1:pin+hM2]
            if not(AfterApply):
                xi *=wh[:,None]
            xr = ICAremoveArtifact(xi,ICAMed=ICAMed,CorrP=CorrP,KurThr=KurThr)
            if AfterApply: xr *=wh[:,None]
            xR[pin-hM1:pin+hM2] += H*xr  ## Overlap Add method
            pin += H
            
        if verbose:
            pf = 100
            pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
            print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
            print('')
            
        xR = xR[hM2:-hM1]/sum(wh)
    return xR

def _RemoveArtftICA_CBI_Kur_Iso(X,winsize=128,CorrP=0.8,KurThr=2,ICAMed = 'extended-infomax',verbose=True):
    '''
    ICAMed = ['fastICA','infomax','extended-infomax','picard']
    
    '''
    win = np.arange(winsize)
    XR =[]
    nch = X.shape[1]
    nSeg = X.shape[0]//winsize
    if verbose:
        print('ICA Artifact Removal : '+ICAMed)
    
    while win[-1]<X.shape[0]:
        if verbose:
            pf = win[-1]*100.0/float(X.shape[0])
            pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
            print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        
        Xi = X[win,:]
        J =[]
        ica = ICA(n_components=nch,method=ICAMed)
        ica.fit(Xi.T)
        IC = ica.transform(Xi.T).T
        mu = ica.pca_mean_
        W = ica.get_sMatrix()
        #A = ica.get_tMatrix()
        sd = np.std(IC,axis=0)
        ICn = IC/sd
        Wn = W*sd
        Wnr = Wn/np.sqrt(np.sum(Wn**2,axis=1,keepdims=True))
        ICss,frqs = np.unique(np.argmax(Wnr,axis=1), return_counts=True)
        
        j1 = ICss[np.where(frqs/nch>=CorrP)[0]]
        J.append(j1)
        ICss,frqs = np.unique(np.argmin(Wnr,axis=1), return_counts=True)
        j2 = ICss[np.where(frqs/nch>=CorrP)[0]]
        J.append(j2)
        CBI,j3,Fault = CBIeye(Wnr,plotW =False)
        if Fault:
            J.append(j3)
        kur   = kurtosis(ICn,axis=0)

        J.append(np.where(abs(kur)>=KurThr)[0])
        J = list(set(np.hstack(J)))
        
        if len(J)>0:
            #print('------')
            for ji in J:
                W[:,ji]=0
            Xr = np.dot(IC,W.T)+mu
        else:
            Xr = Xi
        if win[0]==0:
            XR = Xr
        else:
            XR = np.vstack([XR,Xr])
        win +=winsize
    if verbose:
        pf = 100
        pbar = '|'+'#'*int(pf)+' '*(99-int(pf))+'|'
        print(str(np.round(pf,2))+'%'+pbar,end='\r', flush=True)
        print('')
    return XR

def CBIeye(Wnr,plotW =True):   
    ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    f1stLayer =['AF3','AF4']
    f1stLyInx =[0,13]
    f2stLyInx =[1,2,11,12]
    CBI = np.sum(abs(Wnr[f1stLyInx,:]),axis=0)
    j = np.argmax(CBI)
    if plotW:
        #sns.heatmap(Wnr)
        PlotICACom(abs(Wnr),title=np.around(CBI,2))
        print('#IC ',j)
        print('1st  ',(Wnr[f1stLyInx,j]))
        print('2nd  ',(Wnr[f2stLyInx,j]))
        print([x>y for x in abs(Wnr[f1stLyInx,j]) for y in abs(Wnr[f2stLyInx,j])])
        print([x>y for x in Wnr[f1stLyInx,j] for y in Wnr[f2stLyInx,j]])
    Artifact = np.prod([x>y for x in abs(Wnr[f1stLyInx,j]) for y in abs(Wnr[f2stLyInx,j])])
    #Artifact = np.prod([x>y for x in Wnr[f1stLyInx,j] for y in Wnr[f2stLyInx,j]])
    return CBI,j,Artifact

def ICAremoveArtifact(x,ICAMed='extended-infomax',CorrP=0.8,KurThr=2.0):
    nch = x.shape[1]
    J =[]
    ica = ICA(n_components=nch,method=ICAMed)
    ica.fit(x.T)
    IC = ica.transform(x.T).T
    mu = ica.pca_mean_
    W = ica.get_sMatrix()
    #A = ica.get_tMatrix()
    sd = np.std(IC,axis=0)
    ICn = IC/sd
    Wn = W*sd
    Wnr = Wn/np.sqrt(np.sum(Wn**2,axis=1,keepdims=True))
    ICss,frqs = np.unique(np.argmax(Wnr,axis=1), return_counts=True)

    j1 = ICss[np.where(frqs/nch>=CorrP)[0]]
    J.append(j1)
    ICss,frqs = np.unique(np.argmin(Wnr,axis=1), return_counts=True)
    j2 = ICss[np.where(frqs/nch>=CorrP)[0]]
    J.append(j2)
    CBI,j3,Fault = CBIeye(Wnr,plotW =False)
    if Fault:
        J.append(j3)
    kur   = kurtosis(ICn,axis=0)

    J.append(np.where(abs(kur)>=KurThr)[0])
    J = list(set(np.hstack(J)))

    if len(J)>0:
        #print('------')
        for ji in J:
            W[:,ji]=0
        xr = np.dot(IC,W.T)+mu
    else:
        xr = x
    return xr

def PlotICACom(W, title=None):
    from mne.channels import read_montage
    from mne.viz.topomap import plot_topomap
    ch_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    montage = read_montage('standard_1020',ch_names)
    epos = montage.get_pos2d()
    ch = montage.ch_names
    eOrder = [ch_names.index(c) for c in ch]
    mask = np.ones(14).astype(int)
    
    fig, ax = plt.subplots(2,7,figsize=(15,5))
    i,j=0,0
    for k in range(14):
        #e=np.random.randn(14)
        e = W[:,k]
        plot_topomap(e[eOrder],epos,axes=ax[i,j],show=False,cmap='jet',mask=mask)
        for kk in range(len(eOrder)):
            ax[i,j].text(epos[kk,0]/3.99,epos[kk,1]/3,ch_names[eOrder[kk]],fontsize=6)
        if title is None:
            ax[i,j].set_title(str(k))
        else:
            ax[i,j].set_title(str(title[k]))
        j+=1
        if j==7:
            i+=1
            j=0
    #plt.axis('off')
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    plt.show()

