
'Prepare the TIMIT database.'

import argparse
import glob
import numpy as np
import os
import sys

sys.path.append('utils')
import asrio

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra', type=int, default=0)
    parser.add_argument('phones2id', help='phone id mapping')
    parser.add_argument('mlf', help='alignments in MLF format')
    parser.add_argument('feadir', help='features directory')
    parser.add_argument('outfea', help='output features')
    parser.add_argument('outlabels', help='output labels')
    parser.add_argument('keys', help='utterance ids')
    args = parser.parse_args()

    phones2id = {}
    with open(args.phones2id, 'r') as fid:
        for i, line in enumerate(fid):
            tokens = line.strip().split()
            phones2id[tokens[0]] = i

    with open(args.keys, 'r') as fid:
        keys = [line.strip() for line in fid]

    mlf = asrio.read_mlf(args.mlf)

    labels = []
    features = []
    for key in keys:
        fea = np.load(os.path.join(args.feadir, key + '.npy'))
        ali = mlf[key]
        for entry in ali:
            if entry[0] != 'sil':
                phoneid = phones2id[entry[0]]
                center = int(entry[1] + .5 * (entry[2] - entry[1]))
                features.append(fea[center]), labels.append(phoneid)
                for n in range(1, args.extra + 1):
                    try:
                        features.append(fea[center + n]), labels.append(phoneid)
                    except IndexError:
                        pass
                    try:
                        features.append(fea[center - n]), labels.append(phoneid)
                    except IndexError:
                        pass

    np.save(args.outfea, np.asarray(features))
    np.save(args.outlabels, np.asarray(labels))


if __name__ == '__main__':
    run()

