
'Plot 2^N 2D filters from a neural network.'

import argparse
import numpy as np
import os
import pickle

from bokeh.plotting import figure, gridplot, save, output_file
from bokeh.models import FixedTicker, LinearColorMapper, ColorBar
import colorcet as cc


XAXIS_NTICKS = 7


def figure_filter(filt, title, minimum, maximum, context, nfilters, frate=.01):
    filt_2d = filt.reshape(2 * context + 1, nfilters)
    width, height = (2 * context + 1), nfilters
    ticks = np.linspace(-frate * context, frate * context, XAXIS_NTICKS)

    fig = figure(
        title=title,
        x_range=(-frate * context, frate * context),
        y_range=(0, nfilters),
        x_axis_label='time (s)',
        y_axis_label='filter index'
    )
    fig.xaxis.ticker = FixedTicker(ticks=ticks)

    # Plot the image.
    cmap = LinearColorMapper(palette=cc.rainbow, low=minimum, high=maximum)
    fig.image(
        image=[filt_2d.T],
        x=-frate * context,
        y=0,
        dw=frate * width,
        dh=height,
        color_mapper=cmap
    )

    # Add a color bar to the figure.
    color_bar = ColorBar(color_mapper=cmap, label_standoff=7, location=(0,0))
    fig.add_layout(color_bar, 'right')

    return fig

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nnet', help='neural network')
    parser.add_argument('outhtml', help='output HTML file')
    parser.add_argument('--feadim', type=int, default=30,
                        help='features dimension per frame')
    args = parser.parse_args()

    output_file(args.outhtml, title='2D Filters')

    # Load the neural network.
    with open(args.nnet, 'rb') as fid:
        model = pickle.load(fid)

    # Extract the weight of the first layer.
    weights = list(model.parameters())[0]
    filters = weights.data.numpy()

    # Compute the size of the context.
    context = filters.shape[1] // (2 * args.feadim)

    # Estimate the min./max. value over all the filters. This is needed
    # to normalize the plotting of the filters.
    minimum, maximum = filters.min(), filters.max()

    # Number of figure per rows.
    nfig_per_row = int(np.log2(filters.shape[0]))

    figs = []
    for i, filt in enumerate(filters):
        if i % nfig_per_row == 0:
            figs.append([])
        fig = figure_filter(filt, 'Filter {:d}'.format(i+1), minimum,
            maximum, context, args.feadim)
        figs[-1].append(fig)

    # Save the figures as a single file.
    grid = gridplot(figs)
    save(grid)


if __name__ == '__main__':
    run()
