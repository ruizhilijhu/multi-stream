
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
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('infile', help='input "npz" file')
    parser.add_argument('--context', type=int, default=0,
                        help='number of context frame (one side) to stack')
    args = parser.parse_args()

    arrays = np.load(args.infile)
    out_arrays = {}
    for uttid in arrays:
        out_arrays[uttid] = stack_features(arrays[uttid], args.context)

    # Derive the name of the output file from the input file.
    bname = os.path.basename(args.infile)
    root, _ = os.path.splitext(bname)
    outpath = os.path.join(args.outdir, root)
    np.savez_compressed(outpath, **out_arrays)


if __name__ == '__main__':
    run()

