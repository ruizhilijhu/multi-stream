
'Convert kaldi text alignments to a MLF file.'

import argparse
import glob
import numpy as np
import os
import sys

sys.path.append('utils')

import asrio

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out_mlf', help='output MLF file')
    args = parser.parse_args()

    mlf_data = {}
    for line in sys.stdin:
        tokens = line.strip().split()
        uttid = tokens[0]
        syms = tokens[1:]
        start = 0
        end = start + 1
        previous_sym = syms[0]
        data = []
        for sym in syms[1:]:
            if sym == previous_sym:
                end += 1
            else:
                outsym = previous_sym
                data.append((outsym, start, end, None, None))
                previous_sym = sym
                start = end
                end = start + 1
        outsym = sym
        data.append((outsym, start, end, None, None))
        mlf_data[uttid] = data
    asrio.write_mlf(args.out_mlf, mlf_data)

if __name__ == '__main__':
    run()

