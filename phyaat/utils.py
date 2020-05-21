'''Utilities for PhyAAt.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np



A=['\\','-','/','|']

def ProgBar(i,N,title='',style=1,L=50,selfTerminate=True,delta=None):
    
    pf = int(100*(i+1)/float(N))
    st = ' '*(3-len(str(pf))) + str(pf) +'%|'
    
    if L==50:
        pb = '#'*int(pf//2)+' '*(L-int(pf//2))+'|'
    else: 
        L = 100
        pb = '#'*pf+' '*(L-pf)+'|'
    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='\r', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='\r', flush=True)
    if pf==100 and selfTerminate:
        print('\nDone..')
        
        
def ProgBar_float(i,N,title='',style=1,L=50,selfTerminate=True,delta=None):
    
    pf = np.around(100*(i+1)/float(N),2)
    st = ' '*(5-len(str(pf))) + str(pf) +'%|'
    
    if L==50:
        pb = '#'*int(pf//2)+' '*(L-int(pf//2))+'|'
    else: 
        L = 100
        pb = '#'*int(pf)+' '*(L-int(pf))+'|'
    if style==1:
        print(st+A[i%len(A)]+'|'+pb+title,end='\r', flush=True)
    elif style==2:
        print(st+pb+str(N)+'\\'+str(i+1)+'|'+title,end='\r', flush=True)
    if pf==100 and selfTerminate:
        print('\nDone..')