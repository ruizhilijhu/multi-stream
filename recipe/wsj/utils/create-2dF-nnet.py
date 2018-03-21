
'Create MLP classifier with 2-Dimensional filters for the first layer.'

import argparse
import numpy as np
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feadim', type=int, help='input dimenstion')
    parser.add_argument('ntargets', type=int, help='number of targets')
    parser.add_argument('nfilters', type=int, help='number of 2D filters')
    parser.add_argument('nlayers', type=int, help='number of hidden layers')
    parser.add_argument('nunits', type=int, help='number of units per leayer')
    parser.add_argument('outmodel', help='output file')
    args = parser.parse_args()

    assert args.nlayers > 0

    # Build the MLP.
    structure = [nn.Linear(args.feadim, args.nfilters), nn.Tanh()]
    player_nunits = args.nfilters
    for i in range(args.nlayers - 1):
        structure += [nn.Linear(player_nunits, args.nunits), nn.Tanh()]
        player_nunits = args.nunits
    structure += [nn.Linear(args.nunits, args.ntargets)]
    model = nn.Sequential(*structure)

    with open(args.outmodel, 'wb') as fid:
        pickle.dump(model, fid)


if __name__ == '__main__':
    run()
