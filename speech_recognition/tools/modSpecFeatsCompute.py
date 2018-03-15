#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:46:09 2018

@author: samiksadhu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:10:39 2018

@author: samiksadhu & lucasondel
"""

'Computing Modulation Spectral Features' 

import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis 
#import audiolazy as speechProcessor
import scipy.linalg as lpc_solve
from scipy.signal import freqz

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


#def computeLpc(signal,order):
#    lpc_coeffs=speechProcessor.lpc.nacorr(signal,order)
    #autoc=speechProcessor.acorr(signal,order+1)
    #lpc_coeffs=speechProcessor.levinson_durbin(autoc,order)
#    lpc_filt=lpc_coeffs.numerator
#    gg=np.sqrt(lpc_coeffs.error)
#    xlpc=np.asarray(lpc_filt)
#    return xlpc,gg

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
                    
                    
def extractModSpecFeatures(wavs, outdir, phone_map, phn_file_dir, get_phone_labels=True, only_center=True, around_center=1, ignore_edge=False, nmodulations=12, order=50, fduration=0.5, frate=100,
                            nfft=512, nfilters=15, srate=16000,
                            window=np.hanning):
    '''Extract the Modulation Spectral Features.

    Args:
        wavs (list): List of (uttid, 'filename or pipe-command').
        outdir (string): Output of an existing directory.
        phone_map(string): Map of the phonemes from Kaldi
        get_phone_labels(bool): Set True if you want to get the phoneme labels  
        fduration (float): Frame duration in seconds.
        frate (int): Frame rate in Hertz.
        hz2scale (function): Hz -> 'scale' conversion.
        nfft (int): Number of points to compute the FFT.
        nfilters (int): Number of filters.
        postproc (function): User defined post-processing function.
        srate (int): Expected sampling rate of the audio.
        scale2hz (function): 'scale' -> Hz conversion.
        srate (int): Expected sampling rate.
        window (function): Windowing function.

    Note:
        It is possible to use a Kaldi like style to read the audio
        using a "pipe-command" e.g.: "sph2pipe -f wav /path/file.wav |"

    '''
    
    
    if not only_center:
        fbank = createFbank(nfilters, int(2*fduration*srate), srate)
        
        # Get list of phonemes
        phn_list=[]
        phn_list_39=[]
        phn_list_39_uniq=[]
        
        with open(phone_map,'r') as fid2:
            for line2 in fid2:
                line2=line2.strip().split()
                
                if use_38_phones:
                    if len(line2)==3:
                        if 'sil' not in line2:
                            phn_list_39.append(line2[2])
                            phn_list.append(line2[0])
                    else:
                        phn_list_39.append(' ')
                        phn_list.append(line2[0])
                else:
                    if 'sil' not in line2:
                        phn_list.append(line2[0])
                        
        phn_list_39_uniq=list(set(phn_list_39))
        if ' ' in phn_list_39_uniq: phn_list_39_uniq.remove(' ')
        phn_list_39_uniq.sort()
        
        with open(wavs, 'r') as fid:
            
            save_name=wavs.replace('/','_')
            # Initialize matrix for all features
            if get_phone_labels:
                all_feats=np.empty(nmodulations*nfilters+1)
            else:
                all_feats=np.empty(nmodulations*nfilters)
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                
                if inwav[-1] == '|':
                    # Get phoneme file name
                    fname=tokens[4]
                    fname=os.path.basename(fname)
                    fname_phn=fname[0:-3]+'PHN'
                else:
                    fname=tokens[1]
                    fname=os.path.basename(fname)
                    fname_phn=fname[0:-3]+'PHN'
                
                if get_phone_labels:
                    
                    # Get phoneme file name
                    fname=tokens[4]
                    fname_phn=fname[0:-3]+'PHN'
                    # Get first line of phone file in the beginning 
                    phn_file=open(fname_phn)
                    phn_line=phn_file.readline()
                    phn_locs=phn_line.strip().split()
                    # Get phoneme information 
                    phone_now=phn_locs[2] 
                    phone_end=int(phn_locs[1])
                    
                    
                            
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                else:
                    sr, signal = read(inwav)
                assert sr == srate, 'Input file has different sampling rate.'
                # I want to work with numbers from 0 to 1 so.... 
                signal=signal/np.power(2,15)
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
                cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                [frame_num, ndct]=np.shape(cos_trans)
                if get_phone_labels:
                    feats=np.zeros([frame_num,nmodulations*nfilters+1])
                else:
                    feats=np.zeros([frame_num,nmodulations*nfilters])
                    print('Computing Features for file: %s' % uttid)
                for i in range(frame_num):
                    #print('Running for frame number %d \n' % (i+1))
                    each_feat=np.zeros([nfilters,nmodulations])
                    for j in range(nfilters):
                        filt=fbank[j,0:-1]
                        band_dct=filt*cos_trans[i,:]
                        #band_dct=band_dct[band_dct>0]
                        xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                        #xlpc=np.arange(1,order+3)
                        mod_spec=computeModSpecFromLpc(gg,xlpc,nmodulations)
                        each_feat[j,:]=mod_spec
                    each_feat=np.reshape(each_feat,(1,nfilters*nmodulations),order='F')
                    if get_phone_labels:
                    # Udates to current phoneme
                        if 100*i>phone_end:                
                            # Get new phone label
                            phn_line=phn_file.readline()
                            if phn_line:
                                phn_locs=phn_line.strip().split()
                                phone_now=phn_locs[2] 
                                phone_end=int(phn_locs[1])
                        
                        ind=phn_list.index(phone_now)
                        each_feat=np.append(each_feat,ind)
                    feats[i,:]=each_feat
                        
                
                all_feats=np.vstack([all_feats, feats])
            all_feats=all_feats[1:,:]
            # Save the final BIG feature file 
            np.save(os.path.join(outdir, save_name), all_feats)
    else:
        fbank = createFbank(nfilters, int(2*fduration*srate), srate)
        
        # Get list of phonemes
        phn_list=[]
        
        with open(phone_map,'r') as fid2:
            for line2 in fid2:
                line2=line2.strip().split()
                if len(line2)==2:
                    if 'sil' not in line2 and 'SIL' not in line2:
                        phn_list.append(line2[1])
                        
        phn_list=list(set(phn_list))
        phn_list.sort()            
                             
        with open(wavs, 'r') as fid:
            save_name=wavs.replace('/','_')
            # Initialize matrix for all features
            if get_phone_labels:
                all_feats=np.empty(nmodulations*nfilters+1)
            else:
                all_feats=np.empty(nmodulations*nfilters)
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                    
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                else:
                    sr, signal = read(inwav)
                assert sr == srate, 'Input file has different sampling rate.'
                # I want to work with numbers from 0 to 1 so.... 
                signal=signal/np.power(2,15)
                
                if inwav[-1] == '|':
                    # Get phoneme file name
                    fname=tokens[4]
                    fname=os.path.basename(fname)
                    fname_phn=fname[0:-3]+'PHN'
                else:
                    fname=tokens[1]
                    fname=os.path.basename(fname)
                    fname_phn=fname[0:-3]+'PHN'
                    
                # Get all phones and their center 
                
                phn_file=open(os.path.join(phn_file_dir,fname_phn))
                phone_mid=np.empty(0)
                phone_now=np.empty(0)
                for line2 in phn_file:

                    phn_locs=line2.strip().split() 
                    if phn_locs[2] in phn_list:
                        ind=phn_list.index(phn_locs[2])   
                        phone_now=np.append(phone_now, ind) 
                        phone_mid=np.append(phone_mid
                        ,int(int(phn_locs[0])+int(phn_locs[1]))/2)
                
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
    
                cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                
                [frame_num, ndct]=np.shape(cos_trans)
                
                                
                
                if ignore_edge:
                    phone_mid=phone_mid[1:-1]
                    phone_now=phone_now[1:-1]
                
                
                only_compute=len(phone_mid)
                #print(only_compute) 
                if get_phone_labels:
                    feats=np.zeros([only_compute*around_center,nmodulations*nfilters+1]) 
                else:
                    feats=np.zeros([only_compute*around_center,nmodulations*nfilters])
                
                print('Computing Features for file: %s' % uttid)
                for kk in range(only_compute):
                    i_mid=int(np.floor((phone_mid[kk])))
                    #print('Running for frame number %d \n' % (kk+1))
                    for cont in range(around_center):
                        i=i_mid+cont-int((around_center-1)/2)
                        each_feat=np.zeros([nfilters,nmodulations])
                        for j in range(nfilters):
                            filt=fbank[j,0:-1]
                            band_dct=filt*cos_trans[i,:]
                            #band_dct=band_dct[band_dct>0]
                            xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                            #xlpc=np.arange(1,order+3)
                            mod_spec=computeModSpecFromLpc(gg,xlpc,nmodulations)
                            each_feat[j,:]=mod_spec
                        each_feat=np.reshape(each_feat,(1,nfilters*nmodulations))
                        #print(i_mid)
                        if get_phone_labels:
                            feats[around_center*kk+cont,:]=np.append(each_feat,phone_now[kk])
                        else:
                            feats[around_center*kk+cont,:]=each_feat
                        
                #if get_phone_labels:
                #    feats=np.append(feats,np.reshape(phone_now,(len(phone_now),1)),axis=1)
                all_feats=np.vstack([all_feats, feats])
            all_feats=all_feats[1:,:]
            # Save the final BIG feature file 
            np.save(os.path.join(outdir, save_name), all_feats)
            np.save(os.path.join(outdir, 'phone_list'), phn_list)
        
        

def extractFDLPSpectrum(wavs, outdir, phone_map, phn_file_dir, get_phone_labels=False, only_center=False, order=50, fduration=0.5, frate=100,
                            nfft=512, nfilters=30, srate=16000,
                            window=np.hanning):
    
    if not only_center:
        fbank = createFbank(nfilters, int(2*fduration*srate), srate)
        
        # Get list of phonemes
        # Get list of phonemes
        phn_list=[]
        
        with open(phone_map,'r') as fid2:
            for line2 in fid2:
                line2=line2.strip().split()
                if len(line2)==2:
                    if 'sil' not in line2 and 'SIL' not in line2:
                        phn_list.append(line2[1])
                        
        phn_list=list(set(phn_list))
        phn_list.sort()              
                
        with open(wavs, 'r') as fid:
            
            save_name=wavs.replace('/','_')
            # Initialize matrix for all features
            if get_phone_labels:
                all_feats=np.empty(nfilters+1)
            else:
                all_feats=np.empty(nfilters)
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                    
        
                    
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                    
                    # Get phoneme file name
                    fname=tokens[4]
                    fname_phn=fname[0:-3]+'PHN'

                else:
                    sr, signal = read(inwav[1])
                                     
                    # Get phoneme file name
                    fname=tokens[1]
                    fname_phn=fname[0:-3]+'PHN'
    
                
                if get_phone_labels:
                    # Get first line of phone file in the beginning 
                    phn_file=open(fname_phn)
                    phn_line=phn_file.readline()
                    phn_locs=phn_line.strip().split()
                    # Get phoneme information 
                    phone_now=phn_locs[2] 
                    phone_end=int(phn_locs[1])
                    
                assert sr == srate, 'Input file has different sampling rate.'
                # I want to work with numbers from 0 to 1 so.... 
                signal=signal/np.power(2,15)
                
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
    
                cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                [frame_num, ndct]=np.shape(cos_trans)
                
                if get_phone_labels:
                    feats=np.zeros([frame_num,nfilters+1])
                else:
                    feats=np.zeros([frame_num,nfilters])
                    
                print('Computing Features for file: %s' % uttid)
                
                for i in range(frame_num):
                    
                    if get_phone_labels:
                    # Updates to current phoneme
                        if 100*i>phone_end:                
                            # Get new phone label
                            phn_line=phn_file.readline()
                            if phn_line:
                                phn_locs=phn_line.strip().split()
                                phone_now=phn_locs[2] 
                                phone_end=int(phn_locs[1])
                        
                        ind=phn_list.index(phone_now)
                    
                    each_feat=np.zeros(nfilters)
                    for j in range(nfilters):
                        filt=fbank[j,0:-1]
                        band_dct=filt*cos_trans[i,:]
                        #band_dct=band_dct[band_dct>0]
                        xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                        #xlpc=np.arange(1,order+3)
                        w, h=freqz(np.sqrt(gg),xlpc,ndct)
                        h_mid=np.log10(np.mean(np.abs(h[int(ndct/2-160):int(ndct/2+160)])))
                        each_feat[j]=h_mid
                     
                    if get_phone_labels:
                    # Updates to current phoneme
                        if 100*i>phone_end:                
                            # Get new phone label
                            phn_line=phn_file.readline()
                            if phn_line:
                                phn_locs=phn_line.strip().split()
                                phone_now=phn_locs[2] 
                                phone_end=int(phn_locs[1])
                        
                        ind=phn_list.index(phone_now)
                        each_feat=np.append(each_feat,ind)
                    feats[i,:]=each_feat
                        
                
                all_feats=np.vstack([all_feats, feats])
            all_feats=all_feats[1:,:]
            # Save the final BIG feature file 
            np.save(os.path.join(outdir, save_name), all_feats)
    else:
        fbank = createFbank(nfilters, int(2*fduration*srate), srate)
        
        # Get list of phonemes
        phn_list=[]
        phn_list_39=[]
        phn_list_39_uniq=[]
        
        with open(phone_map,'r') as fid2:
            for line2 in fid2:
                line2=line2.strip().split()
                
                if use_38_phones:
                    if len(line2)==3:
                        if 'sil' not in line2:
                            phn_list_39.append(line2[2])
                            phn_list.append(line2[0])
                    else:
                        phn_list_39.append(' ')
                        phn_list.append(line2[0])
                else:
                    if 'sil' not in line2:
                        phn_list.append(line2[0])
                        
                
        phn_list_39_uniq=list(set(phn_list_39))
        if ' ' in phn_list_39_uniq: phn_list_39_uniq.remove(' ')
        phn_list_39_uniq.sort()
        with open(wavs, 'r') as fid:
            save_name=wavs.replace('/','_')
            # Initialize matrix for all features
            if get_phone_labels:
                all_feats=np.empty(nfilters+1)
            else:
                all_feats=np.empty(nfilters)
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                
                if inwav[-1] == '|':
                    # Get phoneme file name
                    fname=tokens[4]
                    fname_phn=fname[0:-3]+'PHN'
                else:
                    fname=tokens[1]
                    fname_phn=fname[0:-3]+'PHN'
                # Get all phones and their center 
                phn_file=open(fname_phn)
                phone_mid=np.empty(0)
                phone_now=np.empty(0)
                for line2 in phn_file:

                    phn_locs=line2.strip().split() 
                    if phn_locs[2] in phn_list:
                        ind=phn_list.index(phn_locs[2])
                        if use_38_phones:
                            if ' ' in phn_list_39[ind]:
                                pass
                            else:
                                ind=phn_list_39_uniq.index(phn_list_39[ind])
                                phone_now=np.append(phone_now, ind) 
                                phone_mid=np.append(phone_mid
                                ,int(int(phn_locs[0])+int(phn_locs[1]))/2)
                        else:
                            phone_now=np.append(phone_now, ind) 
                            phone_mid=np.append(phone_mid
                            ,int(int(phn_locs[0])+int(phn_locs[1]))/2)
                
                            
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                else:
                    sr, signal = read(inwav)
                assert sr == srate, 'Input file has different sampling rate.'
                # I want to work with numbers from 0 to 1 so.... 
                signal=signal/np.power(2,15)
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
                cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                [frame_num, ndct]=np.shape(cos_trans)
                only_compute=len(phone_mid)
                #print(only_compute) 
                feats=np.zeros([only_compute,nfilters])      
                print('Computing Features for file: %s' % uttid)
                for kk in range(only_compute):
                    i=int(np.floor((phone_mid[kk]/160)))
                    #print('Running for frame number %d \n' % (kk+1))
                    each_feat=np.zeros(nfilters)
                    for j in range(nfilters):
                        filt=fbank[j,0:-1]
                        band_dct=filt*cos_trans[i,:]
                        #band_dct=band_dct[band_dct>0]
                        xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                        #xlpc=np.arange(1,order+3)
                        w, h=freqz(np.sqrt(gg),xlpc,ndct)
                        h_mid=np.log10(np.mean(np.abs(h[int(ndct/2-160):int(ndct/2+160)])))
                        each_feat[j]=h_mid
                    
                    feats[kk,:]=each_feat
                if get_phone_labels:
                    feats=np.append(feats,np.reshape(phone_now,(len(phone_now),1)),axis=1)
                all_feats=np.vstack([all_feats, feats])
                
                
            all_feats=all_feats[1:,:]
            # Save the final BIG feature file 
            np.save(os.path.join(outdir, save_name), all_feats)
            if use_38_phones:
                np.save(os.path.join(outdir, 'phone_list'), phn_list_39_uniq)
            else:
                np.save(os.path.join(outdir, 'phone_list'), phn_list)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Modulation Spectral features.')
    parser.add_argument('scp', help='"scp" list NOTE: Code assumes that the .PHN files have the same locations as in the scp file')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('phn_file', help='Phone map file')
    parser.add_argument('phn_file_dir', help='Location of all phone files')
    parser.add_argument('--nfilters', type=int, default=30,
                        help='number of filters (30)')
    parser.add_argument('--get_phone_labels', action='store_true',
                        help='get phone labels for each feature attached as another column to feature matrix(True)')
    parser.add_argument('--only_center', action='store_true',
                        help='get features from only the center of each phonemes for each utterance (True)') 
    parser.add_argument('--only_FDLP_spectrum', action='store_true',
                        help=' Compute only FDLP Spectrum and Not Modulation Spectrum (False)')
    args = parser.parse_args()
    
    print('Computing features from %s with output directory %s and phone map file %s and .PHN file directory %s' % (args.scp, args.outdir, args.phn_file, args.phn_file_dir))
    if args.only_FDLP_spectrum:
        extractFDLPSpectrum(args.scp, args.outdir, args.phn_file, args.get_phone_labels,args.only_center,args.use_38_phones)
    else:
        extractModSpecFeatures(args.scp, args.outdir, args.phn_file, args.phn_file_dir, args.get_phone_labels,args.only_center)
   
        
