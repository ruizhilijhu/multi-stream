#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:16:44 2018

@author: samiksadhu
"""

'Get Phone Files from Alignments'

from os import listdir
import os 
from os.path import isfile, join
import argparse
import numpy as np


def get_phone_map(phone_file,save_loc):
    save_file=open(join(save_loc,'phone.map'),'w+')
    garbage=['0', '1', '2']
    with open(phone_file) as phn_file:       
        for line in phn_file:
            line=line.strip().split()
            phone=line[0]
            if phone=='<eps>':
                save_file.write("%s %s\n" % (phone, ' '))
            elif phone[0]=='#':
                save_file.write("%s %s\n" % (phone, ' '))
            else:                
                phone2=phone.split('_')
                phone2=phone2[0]
                if phone2[-1] in garbage:
                    phone2=phone2[0:-1]
                
                save_file.write("%s %s\n" % (phone, phone2))
    phn_file.close()
    save_file.close()    
                
                
        
        
def get_neat_alignments(ali_file,phone_map,save_dir):
    
    with open(phone_map) as phn_map:
        content=phn_map.readlines()
    content=[x.strip().split() for x in content]
    #print(content[1])
    ori_phone=[]
    map_phone=[]
    #print(len(content))
    for i in range(len(content)):
        x=content[i]
        if len(x)==1:
            ori_phone.append(x[0])
            map_phone.append(' ')
        else:
            ori_phone.append(x[0])
            map_phone.append(x[1])
        
    with open(ali_file) as file:
        for line in file:
            line=line.strip().split()
            fname=line[0];
            last_phone=line[1]
            last_phone2=map_phone[ori_phone.index(last_phone)]

            beg_phone=1
            count=0
            save_file=open(join(save_dir,fname +'.PHN'),'w+')
            for k in range(2,len(line)):
                count+=1
                if line[k]!=last_phone:
                    #print('%s' % last_phone)  
                    if last_phone2 not in [' ', 'SIL']:
                        save_file.write("%d %d %s\n" % (beg_phone,count,last_phone2))
                    last_phone=line[k]
                    last_phone2=map_phone[ori_phone.index(last_phone)]
                    beg_phone=count+1
            save_file.close()

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ali_dir',help='Directory containing all the alignment .txt files')
    parser.add_argument('nali',help='Number of alignment jobs run')
    parser.add_argument('phn_file_save_dir',help='Directory to Save the generated phone files')
    parser.add_argument('phone_file',help='File containing a list of all phonems in the database')
    
    
    args=parser.parse_args()
    
    # Get phone map
    
    get_phone_map(args.phone_file,'./conf')
    
    # Get all the phone files 
    
    print('Generating phone files....')
    for i in range(1,int(args.nali)+1):
        print('Generating phone files for alignment segment %d' % (i))
        ali_file=join(args.ali_dir,'ali.' + str(i) + '.txt')
        get_neat_alignments(ali_file,'./conf/phone.map',args.phn_file_save_dir)
    print('Phone file generation complete!')