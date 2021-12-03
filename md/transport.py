#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
# import get_variables_210921 as gv
import get_variables_211018 as gv


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
                    help='the quantity to print')

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

    datasets= []
    for key, value in vars(args).items():
        if key.startswith('ds'):
            datasets.append(value)

    datasets_x, datasets_z = [], []
    for k in datasets:
        ds_dir = os.path.join(k, "data/out")
        for root, dirs, files in os.walk(ds_dir):
            for i in files:
                if i.endswith(f'{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(ds_dir, i))
                if i.endswith(f'1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(ds_dir, i))

    # If the processed files are in the data dir
    if not datasets_x:
        for k in datasets:
            ds_dir = os.path.join(k, "data")
            for root, dirs, files in os.walk(ds_dir):
                for i in files:
                    if i.endswith(f'{args.nChunks}x1.nc'):
                        datasets_x.append(os.path.join(ds_dir, i))
                    if i.endswith(f'1x{args.nChunks}.nc'):
                        datasets_z.append(os.path.join(ds_dir, i))


    for i in range(len(datasets)):
        get = gv.derive_data(args.skip, datasets_x[i], datasets_z[i])

        if 'mflux' in args.qtty[0]:
            get.mflux(mf)
        if  'gk' in args.qtty[0]:
            get.green_kubo()
        if 'slip_length' in args.qtty[0]:
            params = get.slip_length(pd=1)
            print(f'Slip Length {params['Ls']} (nm) and velocity {params['Vs']} (m/s)')
        if 'transverse' in args.qtty[0]:
            get.trans()
        if 'pgrad' in args.qtty[0]:
            get.virial()
        if 'sigxz' in args.qtty[0]:
            get.sigwall()
        if 'skx' in args.qtty[0]:
            get.struc_factor()
        if 'transport' in args.qtty[0]:
            params = get.transport(pd=1)
            print(f'Viscosity is {params['mu']:.4f} mPa.s at Shear rate {params['shear_rate']:e} s^-1')
        if 'correlate' in args.qtty[0]:
            get.uncertainty_pDiff(pump_size=0.1)
