#!/usr/bin/python3

import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

filename='./white_500_1500.wav'

fs = 16000
bandrange = [500, 1500]
t = np.arange(0,20,1 / fs)
white = np.random.random(size=len(t))

B, A = butter(4, np.asarray(bandrange) / (fs/2), btype='bandpass')
subnoise = 100 * lfilter(B, A, white)
subnoise = subnoise.astype(np.int16)

write(filename, fs, subnoise)
