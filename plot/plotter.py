#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np

# Qplot = input('Quick plot yes(y) or no(n): ')
# if Qplot == 'y':
#     import plot_main_211018 as pm
# else:
#     import plot_main_220204 as pm

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
    parser.add_argument('--pumpsize', metavar='format', action='store', type=float,
                    help='Pump size. Needed for the pressure gradient compuation.')
    parser.add_argument('--format', metavar='format', action='store', type=str,
                    help='format of the saved figure. Default: PNG')
    parser.add_argument('--config', metavar='config', action='store', type=str,
                    help='Config Yaml file to import plot settings from.')

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

    datasets, txtfiles, arr = [], [], []
    for key, value in vars(args).items():
        if key.startswith('ds'):
            datasets.append(value)
        elif key.startswith('txt'):
            txtfiles.append(value)

    a = []
    for k in txtfiles:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i.endswith('thermo.out') or i.endswith('.txt'):
                    a.append(os.path.join(root, i))

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
        if '_dim' in args.qtty[0]:
            plot.qtty_dim(args.qtty)
        # Quantity profiles with the 1018 module
        if '_length' in args.qtty[0]:
            plot.qtty_len(args.qtty, draw_vlines='y', legend='y', opacity=0.8)
        if '_height' in args.qtty[0]:
            plot.qtty_height(args.qtty, pd='y', legend='y', opacity=1)
        if '_time' in args.qtty[0]:
            plot.qtty_time(args.qtty, lt='-', legend='y', opacity=0.3)

        # Velocity Distibution
        if 'distrib' in args.qtty[0]:
            plot.v_distrib(legend='y', opacity=0.5)

        # Transport Coefficients
        if 'pgrad_mflowrate' in args.qtty[0]:
            plot.pgrad_mflowrate(legend='y')
        if 'pgrad_viscosity' in args.qtty[0]:
            plot.pgrad_viscosity(legend='y')
        if 'rate_stress' in args.qtty[0]:
            plot.rate_stress(legend='y', couette=1)
        if 'rate_viscosity' in args.qtty[0]:
            plot.rate_viscosity(legend='y')
        if 'rate_slip' in args.qtty[0]:
            plot.rate_slip(legend='y')

        # Inlet-Outlet pump quantities
        if 'pt_ratio' in args.qtty[0]:
            plot.pt_ratio()

        if press_temp in args.qtty[0]:
            plot.press_temp()

        # ACFs
        if 'acf' in args.qtty[0]:
            plot.acf(legend='y')
        if 'transverse' in args.qtty[0]:
            plot.transverse(legend='y')
        if 'sk' in args.qtty[0]:
            plot.struc_factor(legend='y')

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
        ptxt = pm.plot_from_txt()
        ptxt.plot_from_txt(args.skip, a, args.qtty[0])
