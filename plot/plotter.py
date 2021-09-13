#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
import plot_main


def get_parser():
    parser = argparse.ArgumentParser(
    description='Plot quantities post-processed from a netcdf trajectory.')

    #Positional arguments
    #--------------------
    parser.add_argument('skip', metavar='skip', action='store', type=int,
                    help='timesteps to skip')
    parser.add_argument('nChunks', metavar='nChunks', action='store', type=int,
                    help='Number of chunks of the horizontal/vertical grid')
    parser.add_argument('fluid', metavar='fluid', action='store', type=str,
                    help='The fluid to perform the analysis on.  \
                        Needed for the calculation of mass flux and density')
    parser.add_argument('qtty', metavar='qtty', nargs='+', action='store', type=str,
                    help='the quantity to plot')
    parser.add_argument('--format', metavar='format', action='store', type=str,
                    help='format of the saved figure. Default: PNG')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str,
                                help='datasets for plotting')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.fluid=='lj': mf = 39.948
    elif args.fluid=='propane': mf = 44.09
    elif args.fluid=='pentane': mf = 72.15
    elif args.fluid=='heptane': mf = 100.21

    datasets, txtfiles, arr = [], [], []
    for key, value in vars(args).items():
        if key.startswith('ds'):
            datasets.append(value)
        elif key.startswith('txt'):
            txtfiles.append(value)

    datasets_x, datasets_z = [], []
    for k in datasets:
        ds_dir = os.path.join(k, "data/out")
        for root, dirs, files in os.walk(ds_dir):
            for i in files:
                if i.endswith(f'{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(ds_dir, i))
                if i.endswith(f'1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(ds_dir, i))

    if len(datasets) > 0.:
        plot = plot_main.plot_from_ds(args.skip, datasets_x, datasets_z, mf)

        if 'length' in args.qtty[0]:
            # print(args.qtty)
            plot.qtty_len(args.qtty, lt='-', legend='y', opacity=1.0)
        if 'height' in args.qtty[0]:
            plot.qtty_height(args.qtty, lt='-', legend='y', opacity=0.9)
        if 'time' in args.qtty[0]:
            plot.qtty_time(args.qtty, lt='-', legend='y', opacity=0.3)
        if 'distrib' in args.qtty[0]:
            plot.v_distrib(legend='y', opacity=0.5)
        if 'acf' in args.qtty[0]:
            plot.acf(legend='y')

        plot.ax.set_rasterized(True)

        if args.format is not None:
            if 'eps' in args.format:
                plot.fig.savefig(args.qtty[0]+'.eps' , format='eps')
            if 'ps' in args.format:
                plot.fig.savefig(args.qtty[0]+'.ps' , format='ps')
            else:
                plot.fig.savefig(args.qtty[0]+'.png' , format='png')
        else:
            plot.fig.savefig(args.qtty[0]+'.png' , format='png')

    if len(txtfiles) > 0.:
        ptxt = plot_from_txt()
        ptxt.plot_from_txt(args.qtty)
