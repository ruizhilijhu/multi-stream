
'Create keys from path.'

import argparse
import os
import sys


def key_from_path(path, depth):
    bname = os.path.basename(path)
    fname, ext = os.path.splitext(bname)
    tmp = path
    tokens = [fname]
    for i in range(depth):
        dirname = os.path.dirname(tmp)
        tokens.append(os.path.basename(dirname))
        tmp = dirname
    return '_'.join(reversed(tokens)).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', type=int, default=0,
                        help='number of dirname to append')
    parser.add_argument('--echo', action='store_true',
                        help='echo the input path after the key')
    args = parser.parse_args()

    for inpath in sys.stdin:
        key = key_from_path(inpath, args.depth)
        if args.echo:
            print(key, inpath.strip())
        else:
            print(key)


if __name__ == '__main__':
    main()
