#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
# import get_variables_210921 as gv
import get_variables_211018 as gv


def get_parser():
    parser = argparse.ArgumentParser(
    description='Print quantities post-processed from a netcdf trajectory.')

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
    parser.add_argument('--pumpsize', metavar='format', action='store', type=float,
                    help='Pump size. Needed for the pressure gradient compuation.')

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

    # Get the pump size to calculate the pressure gradient
    if not args.pumpsize:
        pumpsize = 0        # equilib (Equilibrium), sd (shear-driven)
        print('Pump size is set to zero.')
    else:
        pumpsize = args.pumpsize  # pd (pressure-driven), sp (superposed)
        print(f'Pump size is set to {pumpsize} Lx.')

    datasets= []
    for key, value in vars(args).items():
        if key.startswith('ds') and value!='all':
            datasets.append(value)
        if key.startswith('ds') and value=='all':
            for i in os.listdir(os.getcwd()):
                if os.path.isdir(i):
                    datasets.append(i)

    # Order as indexed on the FileSystem. Sorting is needed if all datasets are
    # points on the same curve e.g. EOS
    try:
        if args.ds == 'all': datasets.sort()
    except AttributeError:
        pass

    datasets_x, datasets_z = [], []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i.endswith(f'{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(root, i))

    for i in range(len(datasets)):
        get = gv.derive_data(args.skip, datasets_x[i], datasets_z[i], mf, pumpsize)

        if 'mflowrate' in args.qtty[0]:
            params = get.mflux()
            print(f"mdot stable = {np.mean(params['mflowrate_stable']):e} g/ns") #\
                  #\nJx stable = {np.mean(params['jx_stable']):.4f} g/m2.ns \
                  #\nJx pump = {np.mean(params['jx_pump']):.4f} g/m2.ns \
                  #\nmdot pump = {np.mean(params['mflowrate_pump']):e} g/ns")
        if 'density' in args.qtty[0]:
            rho = np.mean(get.density()['den_X'])
            print(f'Bulk density (rho) = {rho:.5f} g/cm3')
        if 'gk_viscosity' in args.qtty[0]:
            mu = get.viscosity_gk()
            print(f'Dynamic Viscosity (mu) = {mu} mPa.s')
        if 'slip_length' in args.qtty[0]:
            params = get.slip_length()
            print(f"Slip Length {params['Ls']} (nm) and velocity {params['Vs']} m/s")
        if 'transverse' in args.qtty[0]:
            get.trans()
        if 'tgrad' in args.qtty[0]:
            temp_grad = get.temp()['temp_grad']
            print(f"Temp gradient is {temp_grad*1e9:e} K/m")
        if 'pgrad' in args.qtty[0]:
            vir = get.virial()
            print(f"Pressure gradient is {vir['pGrad']} MPa/nm")
            print(f"Pressure difference is {vir['pDiff']} MPa")
        if 'sigxz' in args.qtty[0]:
            print(f"Avg. sigma_xz {np.mean(get.sigwall()['sigxz_t'])} MPa")
        if 'gaph' in args.qtty[0]:
            print(f"Gap height {np.mean(get.h)} nm")
        if 'skx' in args.qtty[0]:
            get.struc_factor()
        if 'gk_lambda' in args.qtty[0]:
            lambda_tot = get.lambda_gk()['lambda_tot']
            print(f'Thermal conductivity (lambda) = {lambda_tot} W/mK')
        if 'lambda_nemd' in args.qtty[0]:
            thermo_out = datasets[i]+'/data/thermo.out'
            print(f"Thermal Conductivity: {get.lambda_nemd(thermo_out)['lambda_x']} W/mK")
        if 'transport' in args.qtty[0]:
            params = get.transport()
            print(f"Viscosity is {params['mu']:.4f} mPa.s at Shear rate {params['shear_rate']:e} s^-1")
            print(f"Sliding velocity {np.mean(get.h)*1e-9*params['shear_rate']}")
        if 'correlate' in args.qtty[0]:
            get.uncertainty_pDiff(pump_size=0.1)
