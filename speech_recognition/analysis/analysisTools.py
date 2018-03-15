#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:03:22 2018

@author: samiksadhu
"""

' Some Analysis Tools'

import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

def phonewiseErrorPlots(class_count_file, err_count_file, phn_list_dir, plot_title, save_dir):
    # Load necessary files
    class_counts=np.load(class_count_file)
    err_counts=np.load(err_count_file)
    phn_list=np.ndarray.tolist(np.load(phn_list_dir))
    
    perc=np.divide(err_counts,class_counts)*100
    
    
    x=np.arange(0,len(perc))
    plt.rc('xtick', labelsize=5)
    plt.xticks(x,phn_list)
    plt.xlabel('Phonemes')
    plt.ylabel('Error %')
    plt.title(plot_title)
    plt.bar(x,perc)
    #plt.show()
    plt.savefig(os.path.join(save_dir,'phnwiseErrorPlot.jpg'), format='jpg', dpi=1000)
    
def comparePhonewiseErrorPlots2(class_count_file, err_count_files, legends, phn_list_dir, plot_title, save_dir):
    
    class_counts=np.load(class_count_file)
    err_counts1=np.load(err_count_files[0])
    err_counts2=np.load(err_count_files[1])
    perc1=np.divide(err_counts1,class_counts)*100
    perc2=np.divide(err_counts2,class_counts)*100
    phn_list=np.ndarray.tolist(np.load(phn_list_dir))
    
    N=len(class_counts)
    width = 0.35 
    ind = np.arange(N)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    

    rects1 = ax.bar(ind, perc1, width, color='r')
    rects2 = ax.bar(ind+width, perc2, width, color='y')
    
    ax.set_ylabel('Error %')
    ax.set_title(plot_title)
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels(phn_list)
    
    ax.legend( (rects1[0], rects2[0]), legends)
    #plt.show()
    plt.savefig(os.path.join(save_dir,'phnwiseErrorPlot.jpg'), format='jpg', dpi=1000)

def phoneDistributionAtFileEdge(scp_dir, phone_map, save_dir, edge_time=0.25, srate=1600,):
    
    allfiles = [f for f in listdir(scp_dir) if isfile(join(scp_dir, f))]
    

    phn_list=[]
    phn_list_39=[]
    phn_list_39_uniq=[]
    
    # Get my list of phonemes 
    
    with open(phone_map,'r') as fid2:
        for line2 in fid2:
            line2=line2.strip().split()
            
            if len(line2)==3:
                    if 'sil' not in line2:
                        phn_list_39.append(line2[2])
                        phn_list.append(line2[0])
            else:
                    phn_list_39.append(' ')
                    phn_list.append(line2[0])
    
        phn_list_39_uniq=list(set(phn_list_39))
        if ' ' in phn_list_39_uniq: phn_list_39_uniq.remove(' ')
        phn_list_39_uniq.sort() 
    phn_counts=np.zeros(38)
    for wavs in allfiles:
        with open(wavs, 'r') as fid:
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                print('Utt-Id: %s' % uttid)
                if inwav[-1] == '|':
                    # Get phoneme file name
                    fname=tokens[4]
                    fname_phn=fname[0:-3]+'PHN'
                else:
                    fname=tokens[1]
                    fname_phn=fname[0:-3]+'PHN'


                                
                phn_file=open(fname_phn)
                
                is_first=True;
                for line2 in phn_file:
                    phn_locs=line2.strip().split() 
                    if phn_locs[2] in phn_list:
                        ind=phn_list.index(phn_locs[2])
                        if ' ' in phn_list_39[ind]:
                            pass
                        else:
                            ind=phn_list_39_uniq.index(phn_list_39[ind]) 
                            if is_first:
                                phn_counts[int(ind)]+=1
                                is_first=False
                
                            
                phn_counts[int(ind)]+=1
                
                
    N=len(phn_counts)
    xaxis = np.arange(N)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width=0.35

    ax.bar(xaxis, phn_counts, width, color='r')
    
    ax.set_ylabel('Phoneme Counts')
    ax.set_title('Histogram of Edge Phonemes')
    ax.set_xticks(ind)
    ax.set_xticklabels(phn_list_39_uniq)
    
    #plt.show()
    plt.savefig(os.path.join(save_dir,'HistogramOfEdgePhoneme.jpg'), format='jpg', dpi=1000)

if __name__ == '__main__':

   # phonewiseErrorPlots('./class_counts.npy','./class_based_errors.npy','./outdir/phone_list.npy', 'For 0.5 seconds','./analysis_save')
   legends=['0.5 secs','0.25 secs']
   err_count_files=['./class_based_errors_0.5.npy','./class_based_errors_0.25.npy']
   comparePhonewiseErrorPlots2('./class_counts.npy',err_count_files, legends, './outdir/phone_list.npy','Phoneme-wise Error Comparison Ignoring Edge','./analysis_save/')