
'Stack features.'

import argparse
import numpy as np
import os
import sys


def stack_features(data, context):
    if context == 0:
        return data
    padded = np.r_[np.repeat(data[0][None], context, axis=0), data,
                   np.repeat(data[-1][None], context, axis=0)]
    stacked_features = np.zeros((len(data), (2 * context + 1) * data.shape[1]))
    for i in range(context, len(data) + context):
        sfea = padded[i - context: i + context + 1]
        stacked_features[i - context] = sfea.reshape(-1)
    return stacked_features


def run():
    parser = argparse.ArgumentParser('Stack features.')
    parser.add_argument('indir', help='input directory')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('keys', help='"scp" list')
    parser.add_argument('--context', type=int, default=0,
                        help='number of context frame (one side) to stack')
    args = parser.parse_args()

    with open(args.keys, 'r') as fid:
        for line in fid:
            key = line.strip().split()[0]
            infname = os.path.join(args.indir, key + '.npy')
            outfname = os.path.join(args.outdir, key + '.npy')
            data = np.load(infname)
            np.save(outfname, stack_features(data, args.context))


if __name__ == '__main__':
    run()

