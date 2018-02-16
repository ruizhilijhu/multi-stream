
'Filter Bank for speech analysis.'


import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess


def frames(signal, srate, frate, flength, window):
    '''Generator of overlapping frames

    Args:
        signal (numpy.ndarray): Audio signal.
        srate (int): Sampling rate of the signal.
        frate (float): Frame rate in second.
        flength (float): Frame length in second.
        window (function): Window function (see numpy.hamming for instance).

    Yields:
        frame (numpy.ndarray): frame of length ``flength`` every ``frate``
            second.

    '''
    idx = 0
    flength_samples = int(srate * flength)
    frate_samples = int(srate * frate)
    win = window(flength_samples)
    while idx < len(signal):
        frame = signal[idx:idx + flength_samples]
        pad_length = flength_samples - len(frame)
        frame = np.pad(frame, (0, pad_length), 'constant',
                               constant_values=(0, frame[-1]))
        yield frame * win
        idx += frate_samples


def power_spectrum(signals, srate, npoints):
    '''Compute the power spectrum for a set of signals.

    Args:
        signals (seq): Signals.
        srate (int): Sampling rate.
        npoints (int): Number of points of the FFT.

    Returns
        (numpy.ndarray): Power spectrum of the signal.
        (numpy.ndarray): Corresponding frequencies of the bins.

    '''
    #spectrum = np.fft.rfft(signals, npoints, norm='ortho')
    spectrum = np.fft.rfft(signals, npoints)
    power_spectrum = np.abs(spectrum) ** 2
    return power_spectrum


def hz2mel(hz):
    'Convert Hertz to Mel value(s).'
    return 2595 * np.log10(1+hz/700.)


def mel2hz(mel):
    'Convert Mel value(s) to Hertz.'
    return 700*(10**(mel/2595.0)-1)


def hz2bark(hz):
    'Convert Hertz to Bark value(s).'
    return (29.81 * hz) / (1960 + hz) - 0.53


def bark2hz(bark):
    'Convert Bark value(s) to Hertz.'
    return (1960 * (bark + .53)) / (29.81 - bark - .53)


def create_fbank(nfilters, npoints=512, srate=16000, lowfreq=0, highfreq=None,
                 hz2scale=hz2mel, scale2hz=mel2hz):
    '''Create a set of triangular filter.

    Args:
        nfilter (int): Number of filters.
        npoints (int): Number of points of the FFT transform.
        srate (int): Sampling rate of the signal to be filtered.
        lowfreq (float): Global cut off frequency (Hz).
        highfreq (float): Global cut off frequency (Hz).
        hz2scale (function): Conversion from Hertz to the 'perceptual' scale to use.
        scale2hz (function): Inversion function of ``hz2scale``.

    Returns
        (numpy.ndarray): The filters organized as a matrix.

    '''
    highfreq = highfreq or srate/2
    assert highfreq <= srate/2, "highfreq is greater than samplerate/2"

    low = hz2scale(lowfreq)
    high = hz2scale(highfreq)
    centers = np.linspace(low, high, nfilters + 2)

    # our points are in Hz, but we use fft bins, so we have to convert
    # from Hz to fft bin number
    bin = np.floor((npoints + 1) * scale2hz(centers) / srate)

    fbank = np.zeros([nfilters, npoints // 2 + 1])
    for j in range(0, nfilters):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[ j + 2] - bin[j + 1])

    return fbank


def extract_fbank_features(wavs, outdir, fduration=0.025, frate=100,
                           hz2scale=hz2mel, nfft=512, nfilters=16,
                           postproc=lambda x: x, scale2hz=mel2hz, srate=16000,
                           window=np.hamming):
    '''Extract the FBANK features.

    The features are extracted according to the following scheme:

                     ...
                     -> Filter
        x -> |FFT|^2 -> Filter -> log
                     -> Filter
                     ...

    The set of filters is composed of triangular filters equally spaced
    on some 'perceptual' scale (usually the Mel scale).

    Args:
        wavs (list): List of (uttid, 'filename or pipe-command').
        outdir (string): Output of an existing directory.
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
    fbank = create_fbank(
        nfilters,
        nfft,
        hz2scale=hz2scale,
        scale2hz=scale2hz
    )

    with open(wavs, 'r') as fid:
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
            time_frames = np.array([frame for frame in
                frames(signal, srate, 1. /frate, fduration, window)])
            powspec = power_spectrum(time_frames, srate, nfft)
            log_melspec = np.log(powspec @ fbank.T + 1e-30)
            np.save(os.path.join(outdir, uttid), postproc(log_melspec))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract Mel spectrum features.')
    parser.add_argument('scp', help='"scp" list')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('--nfilters', type=int, default=30,
                        help='number of filters (30)')
    parser.add_argument('--context', type=int, default=0,
                        help='number of context frame (one side) to stack (0)')
    parser.add_argument('--window', action='store_true',
                        help='apply a window when stacking')
    args = parser.parse_args()

    def stack_features(data):
        size = args.context
        win = np.hamming(2 * size + 1)
        if size == 0:
            return data
        padded = np.r_[np.repeat(data[0][None], size, axis=0), data,
                       np.repeat(data[-1][None], size, axis=0)]
        stacked_features = np.zeros((len(data), (2 * size + 1) * data.shape[1]))
        for i in range(size, len(data) + size):
            sfea = padded[i - size: i + size + 1]
            stacked_features[i - size] = sfea.reshape(-1)

        if args.window:
            for i in range(len(stacked_features)):
                stacked_features[i] = \
                    (stacked_features[i].reshape(2 * size + 1, data.shape[1]) \
                    * win[:, None]).reshape(-1)
        return stacked_features

    extract_fbank_features(args.scp, args.outdir, nfilters=args.nfilters,
                           postproc=stack_features)

