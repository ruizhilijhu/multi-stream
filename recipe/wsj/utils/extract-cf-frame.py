
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
    parser.add_argument('--exclude',
        help='comma separated phones exclude')
    parser.add_argument('phones', help='phone list')
    parser.add_argument('mlf', help='alignments in MLF format')
    parser.add_argument('keys', help='utterance ids')
    parser.add_argument('outdir', help='output features')
    parser.add_argument('infile', help='features directory')
    args = parser.parse_args()

    # List of phones to exclude.
    to_exclude = [phone for phone in args.exclude.split(',')]

    # Create the phone <-> id mapping.
    phones2id = {}
    count = 0
    with open(args.phones, 'r') as fid:
        for line in fid:
            tokens = line.strip().split()
            if tokens[0] not in to_exclude:
                phones2id[tokens[0]] = count
                count += 1

    # Keys to extracts.
    with open(args.keys, 'r') as fid:
        keys = [line.strip() for line in fid]

    # Load the alignments.
    mlf = asrio.read_mlf(args.mlf)

    # Load the features.
    arrays = np.load(args.infile)

    labels = []
    features = []
    for uttid in arrays:
        fea = arrays[uttid]
        try:
            ali = mlf[uttid]
        except KeyError:
            print('[warning]: no alignment for utterance:', uttid, file=sys.stderr)
            continue
        for entry in ali:
            if not entry[0] in to_exclude:
                phoneid = phones2id[entry[0]]
                center = int(entry[1] + .5 * (entry[2] - entry[1]))
                features.append(fea[center]), labels.append(phoneid)

    # Derive the name of the output file from the input file.
    bname = os.path.basename(args.infile)
    root, _ = os.path.splitext(bname)
    outpath = os.path.join(args.outdir, root)
    np.savez_compressed(outpath, features=features, labels=labels)


if __name__ == '__main__':
    run()

