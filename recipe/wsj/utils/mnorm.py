
'Mean normalize a data set.'

import argparse
import glob
import numpy as np
import os
import sys

sys.path.append('utils')
import asrio

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', help='list of "npz" archive')
    args = parser.parse_args()

    mean = None
    var = None
    count = 0
    with open(args.archives, 'r') as fid:
        for line in fid:
            data = np.load(line.strip())
            fea = data['features']
            dmean = fea.sum(axis=0)
            dvar = (fea ** 2).sum(axis=0)
            count += len(fea)
            if mean is None:
                mean = dmean
                var = dvar
            else:
                mean += dmean
                var += dvar
    mean /= count
    var = (var - mean**2) / count

    with open(args.archives, 'r') as fid:
        for line in fid:
            data = np.load(line.strip())
            fea, labels = data['features'], data['labels']
            fea -= mean
            fea /= np.sqrt(var)
            np.savez_compressed(line.strip(), features=fea, labels=labels)


if __name__ == '__main__':
    run()

