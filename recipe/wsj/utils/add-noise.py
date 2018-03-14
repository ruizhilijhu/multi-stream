
'Add noise at a specific DB level.'


import argparse
import beer
import io
import numpy as np
import os
from scipy.io.wavfile import read, write
from scipy.signal import butter, lfilter
import subprocess


def generate_white_noise(nsamples, srate, lowfreq, highfreq, filt_order=4):
    if lowfreq <= 0 or highfreq >= srate / 2.:
        raise ValueError('cut-off frequencies outside the range (0, Fs/2)')
    noise_wideband = np.random.randn(nsamples)
    freqs = (2. * lowfreq / srate, 2. * highfreq / srate)
    B, A = butter(filt_order, freqs, btype='bandpass')
    return lfilter(B, A, noise_wideband)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('--low-freq', type=int, default=100)
    parser.add_argument('--high-freq', type=int, default=7500)
    parser.add_argument('--srate', type=int, default=16000,
                        help='expected sampling rate')
    parser.add_argument('--noise-level', type=float, default=0.,
                        help='noise level in db')
    args = parser.parse_args()


    with open(args.scp, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            uttid, inwav = tokens[0], ' '.join(tokens[1:])

            # If 'inwav' ends up with the '|' symbol, 'inwav' is
            # interpreted as a command otherwise we assume 'inwav' to
            # be a path to a wav file.
            if inwav[-1] == '|':
                proc = subprocess.run(inwav[:-1], shell=True,
                                      stdout=subprocess.PIPE)
                sr, signal = read(io.BytesIO(proc.stdout))
            else:
                sr, signal = read(inwav)
            assert sr == args.srate, 'Input file has different sampling rate.'

            # Generate white noise sample (0 mean and 1 variance).
            white_noise = generate_white_noise(len(signal), args.srate,
                args.low_freq, args.high_freq)

            # Measure the signal power.
            p_signal = np.sum(signal ** 2)

            # Measure the current noise power.
            p_noise0 = np.sum(white_noise ** 2)

            # Estimate the power of the noise to have the correct SNR.
            p_noise = p_signal / 10 ** (args.noise_level / 10.)

            # Scale the noise signal to have the requested power.
            white_noise *= np.sqrt(p_noise / p_noise0)

            # Save the corrupted signal.
            write(os.path.join(args.outdir, uttid + '.wav'), args.srate,
                  (white_noise + signal).astype(signal.dtype))


if __name__ == '__main__':
    main()

