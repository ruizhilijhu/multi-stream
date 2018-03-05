#!/usr/bin/python3

from scipy.io import wavfile
import numpy as np
import argparse
import sys
import math
import subprocess
import io
import os

def read_scp(flist, mode='r'):
    '''return: fDict[uttid]-> {[frate, faudio]}
    '''
    fDict = {}
    for f in flist:
        f = f.rstrip()
        fL = f.split()
        uttid = fL.pop(0)
        if mode == 'r':
            if '|' in fL:
                output = subprocess.check_output(' '.join(fL)[:-1], shell=True)
                audio = io.BytesIO(output)
                fRate, faudio = wavfile.read(audio)
            else:
                fRate, faudio = wavfile.read(fL[-1])
            fDict[uttid] = [fRate, faudio]
        elif mode == 'w':
            fDict[uttid] = fL[-1]
    return fDict

def add_noise(iDict, oDict, noiseFile, snr):
    numUtt = len(iDict.keys())
    count = 0
    for i in iDict.keys():
        signal = iDict[i][1].astype(np.float64)
        oFile = oDict[i]
        iSmpRate = iDict[i][0]
        nSmpRate, noise = wavfile.read(noiseFile)
        if (iSmpRate != nSmpRate):
            sys.exit('Audio file smprate ' + str(iSmpRate) + ' not equal to noise')
        sig_len = signal.size
        if (sig_len > noise.size):
            sys.exit('Signal longer than noise, exiting...')
        else:
            noise_cut = noise[:sig_len]
            sig_power = np.sum(np.square(signal))
            noise_power = np.sum(np.square(noise_cut))
            B = math.pow(10, float(snr)/10)
            K_square = sig_power/(noise_power * B)
            K = math.sqrt(K_square)
            noisy = np.add(signal, K * noise_cut)
            noisy = np.around(noisy).astype(np.int16)
            if not os.path.exists(os.path.dirname(oFile)):
                os.makedirs(os.path.dirname(oFile))
            wavfile.write(oFile, iSmpRate, noisy)
            count = count + 1
            print ('Processed file ' + i + ' ' + str(count) + '/' + str(numUtt))

def main():
    parser = argparse.ArgumentParser(description=
    '''Adding noise from a noise file to a list of audios''')
    parser.add_argument('--i', help='input audio scp')
    parser.add_argument('--o', help='output audio list')
    parser.add_argument('--n', help='noise file')
    parser.add_argument('--snr')
    args = parser.parse_args()

    #noise_dir='/home/jyang/timit_phoneme_recognition/fant/noises/preMIRS/'
    #noise_dir=
    #nfile = noise_dir + args.n + '.wav'

    with open(args.o, 'r') as k:
        oL = k.readlines()
    oDict = read_scp(oL, mode='w')

    with open(args.i, 'r') as f:
        iL = f.readlines()
    iDict = read_scp(iL, mode='r')

    add_noise(iDict, oDict, args.n, args.snr)
    #    for j, iF in enumerate(iL):
    #        add_noise(iF.rstrip(), oL[j].rstrip(), args.n, float(args.snr))

if __name__ == '__main__':
    main()
else:
    raise ImportError('This script cannot be imported')

