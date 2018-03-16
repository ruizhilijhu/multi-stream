#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:55:42 2018

@author: samiksadhu
"""

'Program to clean the .PHN files in TIMIT'


from os.path import join
import argparse

def get_clean_PHN_files(scp_file,phone_map,save_dir):
    print('Obtaining the phone map file...')
    with open(phone_map) as phn_map:
        content=phn_map.readlines()
    content=[x.strip().split() for x in content]
    ori_phone=[]
    map_phone=[]
    for i in range(len(content)):
        x=content[i]
        if len(x)==1:
            ori_phone.append(x[0])
            map_phone.append(' ')
        else:
            ori_phone.append(x[0])
            map_phone.append(x[1])
    print('Cleaning phone files...')        
    with open(scp_file) as file:
        for line in file:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])
            if inwav[-1] == '|':             
                # Get phoneme file name
                fname=tokens[4]
                fname_phn=fname[0:-3]+'PHN'
            else:
                # Get phoneme file name
                fname=tokens[1]
                fname_phn=fname[0:-3]+'PHN'
                
            save_file=open(join(save_dir,uttid + '.PHN'),'w+')
            with open(fname_phn) as now_file:
                for line2 in now_file:
                    line2=line2.strip().split()
                    now_phone=line2[2]
                    now_phone2=map_phone[ori_phone.index(now_phone)]
                    if now_phone2 not in [' ', 'sil', 'SIL']:
                        save_file.write("%d %d %s\n" % (int(int(line2[0])/160),int(int(line2[1])/160),now_phone2))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scp_file',help='Scp files for all wav files')
    parser.add_argument('phone_map',help='Map file for phones')
    parser.add_argument('save_dir', help='Directory to save the generated phone files')
    
    args=parser.parse_args()
    get_clean_PHN_files(args.scp_file,args.phone_map,args.save_dir)
    