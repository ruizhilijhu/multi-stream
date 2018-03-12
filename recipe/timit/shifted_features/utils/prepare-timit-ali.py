
'Prepare the TIMIT database.'

import argparse
import glob
import os
import sys

sys.path.append('utils')
import asrio


phone_map_61to39 = {
    'iy': 'iy',
    'ih': 'ih',
    'eh': 'eh',
    'ey': 'ey',
    'ae': 'ae',
    'aa': 'aa',
    'aw': 'aw',
    'ay': 'ay',
    'ah': 'ah',
    'ao': 'aa',
    'oy': 'oy',
    'ow': 'ow',
    'uh': 'uh',
    'uw': 'uw',
    'ux': 'uw',
    'er': 'er',
    'ax': 'ah',
    'ix': 'ih',
    'axr': 'er',
    'ax-h': 'ah',
    'jh': 'jh',
    'ch': 'ch',
    'b': 'b',
    'd': 'd',
    'g': 'g',
    'p': 'p',
    't': 't',
    'k': 'k',
    'dx': 'dx',
    's': 's',
    'sh': 'sh',
    'z': 'z',
    'zh': 'sh',
    'f': 'f',
    'th': 'th',
    'v': 'v',
    'dh': 'dh',
    'm': 'm',
    'n': 'n',
    'ng': 'ng',
    'em': 'm',
    'nx': 'n',
    'en': 'n',
    'eng': 'ng',
    'l': 'l',
    'r': 'r',
    'w': 'w',
    'y': 'y',
    'hh': 'hh',
    'hv': 'hh',
    'el': 'l',
    'pcl': 'sil',
    'tcl': 'sil',
    'kcl': 'sil',
    'bcl': 'sil',
    'dcl': 'sil',
    'gcl': 'sil',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil',
    # q is not map to anything.
}


def extract_key(path, depth=0):
    bname = os.path.basename(path)
    fname, _ = os.path.splitext(bname)
    prefixes = []
    dirname = os.path.dirname(path)
    for d in range(1, depth + 1):
        prefixes.append(os.path.basename(dirname))
        dirname = os.path.dirname(dirname)
    if len(prefixes) > 0:
        prefixes.insert(0, '')
    return ('_'.join(reversed(prefixes)) + fname).lower()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phns', help='glob pattern to get the PHN files')
    args = parser.parse_args()

    mlf_data = {}
    for path in glob.glob(args.phns):
        key = extract_key(path, depth=1)
        data = asrio.read_timit_labels(path)
        new_data = []
        for name, s, e, _, _ in data:
            try:
                new_data.append((phone_map_61to39[name], s, e, None, None))
            except KeyError:
                pass
        mlf_data[key] = new_data
    asrio.write_mlf(sys.stdout, mlf_data)


if __name__ == '__main__':
    run()

