#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:49:37 2018

@author: samiksadhu
"""

'Some Functions for Feature Computation' 

import numpy as np
import scipy.linalg as lpc_solve


def getFrames(signal, srate, frate, flength, window):
    '''Generator of overlapping frames

    Args:
        signal (numpy.ndarray): Audio signal.
        srate (float): Sampling rate of the signal.
        frate (float): Frame rate in Hz.
        flength (float): Frame length in second.
        window (function): Window function (see numpy.hamming for instance).

    Yields:
        frame (numpy.ndarray): frame of length ``flength`` every ``frate``
            second.

    '''
    
    
    flength_samples = int(srate * flength)
    frate_samples = int(srate/frate)
    
    if flength_samples % 2 ==0:
        sp_b=int(flength_samples/2)-1
        sp_f=int(flength_samples/2)
        extend=int(flength_samples/2)
    else:
        sp_b=int((flength_samples-1)/2)
        sp_f=int((flength_samples-1)/2)
        extend=int((flength_samples-1)/2)
        
    sig_padded=np.pad(signal,extend,'reflect')
    win = window(flength_samples)
    idx=sp_b;
    
    while (idx+sp_f) < len(sig_padded):
        frame = sig_padded[idx-sp_b:idx + sp_f+1]
        yield frame * win
        idx += frate_samples
        

def createFbank(nfilters, nfft, srate):
    mel_max = 2595 * np.log10(1 + srate / 1400)
    fwarped = np.linspace(0, mel_max, nfilters + 2)

    mel_filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
    hz_points = (700 * (10 ** (fwarped / 2595) - 1))
    bin = np.floor((nfft + 1) * hz_points / srate)

    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            mel_filts[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            mel_filts[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])  

    return mel_filts

def computeLpcFast(signal,order):
    y=np.correlate(signal,signal,'full')
    y=y[(len(signal)-1):]
    xlpc=lpc_solve.solve_toeplitz(y[0:order],-y[1:order+1])
    xlpc=np.append(1,xlpc)
    gg=y[0]+np.sum(xlpc*y[1:order+2])
    return xlpc, gg

def computeModSpecFromLpc(gg,xlpc,lim):
    xlpc[1:]=-xlpc[1:]
    lpc_cep=np.zeros(lim)
    lpc_cep[0]=np.log(np.sqrt(gg))
    lpc_cep[1]=xlpc[1]
    
    for n in range(2,lim):
        aa=np.arange(1,n)/n
        bb=np.flipud(xlpc[1:n])
        cc=lpc_cep[1:n]
        acc=np.sum(aa*bb*cc)
        lpc_cep[n]=acc+xlpc[n]
    return lpc_cep

