#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
from pathlib import Path
import yaml

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
    parser.add_argument('--pumpsize', metavar='pumpsize', action='store', type=float,
                    help='Pump size. Needed for the pressure gradient compuation.')
    parser.add_argument('--format', metavar='format', action='store', type=str,
                    help='format of the saved figure. Default: PNG')
    parser.add_argument('--config', metavar='config', action='store', type=str,
                    help='Config Yaml file to import plot settings from.')

    # Get the datasets
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-ds")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str,
                                help='datasets for plotting')
        if arg.startswith(("-txt")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str,
                                help='files for plotting')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # Set the fluid molecular mass
    if args.fluid=='lj': mf = 39.948
    elif args.fluid=='propane': mf = 44.09
    elif args.fluid=='pentane': mf = 72.15
    elif args.fluid=='heptane': mf = 100.21

    # Get the pump size to calculate the pressure gradient
    if not args.pumpsize:
        pumpsize = 0        # equilib (Equilibrium), sd (shear-driven)
        print('Pump size is set to zero.')
    else:
        pumpsize = args.pumpsize  # pd (pressure-driven), sp (superposed)
        print(f'Pump size is set to {pumpsize} Lx.')

    # Read the yaml file if given
    if args.config:
        import plot_main_220204 as pm
        with open(args.config, 'r') as f:
            file = yaml.safe_load(f)
    else:
        import plot_main_211018 as pm

    datasets, txtfiles = [], []
    for key, value in vars(args).items():
        if key.startswith('ds') and value!='all':
            datasets.append(value)
        if key.startswith('ds') and value=='all':
            for i in os.listdir(os.getcwd()):
                if os.path.isdir(i):
                    datasets.append(i)
        elif key.startswith('txt'):
            txtfiles.append(value)

    # Order as indexed on the FileSystem. Sorting is needed if all datasets are
    # points on the same curve e.g. EOS
    try:
        if args.ds == 'all': datasets.sort()
    except AttributeError:
        pass

    txts = []
    for k in txtfiles:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i.endswith('thermo.out') or i.endswith('.txt'):
                    txts.append(os.path.join(root, i))

    datasets_x, datasets_z = [], []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i.endswith(f'{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(root, i))

    if len(datasets) > 0.:
        if args.config:
            plot = pm.plot_from_ds(args.skip, datasets_x, datasets_z, mf, args.config, pumpsize)
        else:
            plot = pm.plot_from_ds(args.skip, datasets_x, datasets_z, mf, pumpsize)
        # Quantity profiles with the 0204 module
        if '_dim' in args.qtty[0]: plot.qtty_dim(args.qtty)
        # Velocity Distibution
        if 'distrib' in args.qtty: plot.v_distrib()
        # Transport Coefficients
        if 'pgrad_mflowrate' in args.qtty: plot.pgrad_mflowrate()
        if 'rate_stress' in args.qtty: plot.rate_stress()
        if 'rate_viscosity' in args.qtty: plot.rate_viscosity()
        if 'rate_slip' in args.qtty: plot.rate_slip()
        if 'rate_temp' in args.qtty: plot.rate_temp()
        # if 'pt_ratio' in args.qtty[0]: plot.pt_ratio()
        if 'eos' in args.qtty: plot.eos()
        if 'lambda' in args.qtty: plot.thermal_conduct()
        # ACFs
        if 'acf' in args.qtty: plot.acf()
        if 'transverse' in args.qtty: plot.transverse()
        if 'sk' in args.qtty: plot.struc_factor()
        if 'isf' in args.qtty: plot.isf()

        # Quantity profiles with the 1018 module
        if '_length' in args.qtty[0]: plot.qtty_len(args.qtty, draw_vlines='y', legend='y',opacity=0.8)
        if '_height' in args.qtty[0]: plot.qtty_height(args.qtty, legend='y',opacity=0.8)
        if '_time' in args.qtty[0]: plot.qtty_time(args.qtty, legend='y', opacity=0.3)

        if args.config:
            for ax in plot.axes_array:
                ax.set_rasterized(True)
        else:
            plot.ax.set_rasterized(True)

        if args.format is not None:
            if args.format == 'eps':
                plot.fig.savefig(args.qtty[0]+'.eps' , format='eps')
            if args.format == 'ps':
                plot.fig.savefig(args.qtty[0]+'.ps' , format='ps')
            if args.format == 'svg':
                plot.fig.savefig(args.qtty[0]+'.svg' , format='svg')
        else:
            plot.fig.savefig(args.qtty[0]+'.png' , format='png')

    if len(txtfiles) > 0.:
        ptxt = pm.plot_from_txt(args.skip, txts, args.config)
        p = ptxt.plot_txt(args.qtty)
        ptxt.fig.savefig(args.qtty[0]+'.png' , format='png')




#
