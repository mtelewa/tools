#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import numpy as np
import compute_thermo as ct

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
    parser.add_argument('--pumpsize', metavar='pumpsize', action='store', type=float,
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
    if 'all' in vars(args).values():
        datasets.sort()
    for i in vars(args).keys():
        if i.startswith('all'):
            datasets.sort()

    datasets_x, datasets_z = [], []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i.endswith(f'{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(root, i))

    txts = []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in sorted(files):
                if i.startswith('log.lammps'):
                    txts.append(os.path.join(root, i))

    # start and stop: the time frame for averaging the Green-Kubo viscosity
    # tcorr: correlation time for the evaluation of the acf and the Green-Kubo integral
    start, stop, t_corr = 2000, 8000, 20000         # frames
    eta_avg, lambda_avg = [], []
    eta = np.zeros([len(datasets), t_corr-1])
    lambda_time = np.zeros([len(datasets), t_corr-1])

    for i in range(len(datasets)):
        try:
            get = ct.ExtractFromTraj(args.skip, datasets_x[i], datasets_z[i], mf, pumpsize)
        except IndexError:
            print(f'Dataset is not processed or trajectory does not exist! \n\
            Run post-processing in the <dataset>/out directory and ensure that the trajectory is there.')
            exit()

        if 'mflowrate' in args.qtty:
            params = get.mflux()
            print(f"ṁ  stable = {np.mean(params['mflowrate_stable']):e} g/ns")
        if 'mflowrate_hp' in args.qtty:
            params = get.mflowrate_hp()
            print(f"ṁ  without slip = {np.mean(params['mflowrate_hp']):e} g/ns")
            print(f"ṁ  with slip = {np.mean(params['mflowrate_hp_slip']):e} g/ns")
        if 'density' in args.qtty:
            rho = np.mean(get.density()['den_X'])
            print(f'Bulk density (rho) = {rho:.5f} g/cm3')
        if 'temp' in args.qtty:
            temp = np.mean(get.temp()['temp_t'])
            print(f'Average temperature = {temp:.2f} K')
        if 'press' in args.qtty:
            pressure = np.mean(get.virial()['vir_t'])
            print(f'Average pressure = {pressure:.2f} MPa')

        if 'viscosity_gk' in args.qtty:
            # viscosity with correlation time plots for each replicate
            data = get.viscosity_gk_log(txts[i], t_corr)
            eta[i, :] = data['eta']
            time = np.arange(1, t_corr)
            np.savetxt(f'eta_{i:02d}.txt', np.c_[time*data['thermo_out'], eta[i]],  delimiter=' ',
                            header='time   viscosity')

        if 'conductivity_gk' in args.qtty:
            data = get.conductivity_gk_log(txts[i], t_corr)
            lambda_time[i, :] = data['lambda_time']
            time = np.arange(1, t_corr)
            np.savetxt(f'lambda_{i:02d}.txt', np.c_[time*data['thermo_out'], lambda_time[i]],  delimiter=' ',
                            header='time   conductivity')

        if 'slip_length' in args.qtty:
            params = get.slip_length()
            print(f"Slip Length {params['Ls']} (nm) and velocity {params['Vs']} m/s")
        if 'transverse' in args.qtty:
            get.trans()
        if 'tgrad' in args.qtty:
            temp_grad = get.temp()['temp_grad']
            print(f"Temp gradient is {temp_grad*1e9:e} K/m")
        if 'pgrad' in args.qtty:
            vir = get.virial()
            print(f"Pressure gradient is {vir['pGrad']:.2f} MPa/nm")
            print(f"Pressure difference is {vir['pDiff']:.2f} MPa")
        if 'sigxz' in args.qtty:
            print(f"Avg. σ_xz {np.mean(get.sigwall()['sigxz_t']):.4f} MPa")
        if 'gaph' in args.qtty:
            print(f"Gap height {np.mean(get.h):.2f} nm")
        if 'gaph_conv' in args.qtty:
            print(f"Gap height of converged region {np.mean(get.h_conv):.2f} nm")
        if 'gaph_div' in args.qtty:
            print(f"Gap height of diverged region {np.mean(get.h_div):.2f} nm")
        if 'skx' in args.qtty:
            get.struc_factor()

        if 'conductivity_ecouple' in args.qtty:
            log_file = datasets[i]+'/data/log.lammps'
            print(f"Thermal Conductivity = {get.conductivity_ecouple(log_file)['conductivity_z']} W/mK")
        if 'conductivity_IK' in args.qtty:
            print(f"Thermal Conductivity = {np.mean(get.conductivity_IK()['conductivity_z'])} W/mK")
        if 'viscosity_nemd' in args.qtty:
            params = get.viscosity_nemd()
            print(f"Viscosity is {params['eta']:.4f} mPa.s at Shear rate {params['shear_rate']:e} s^-1")
            print(f"Sliding velocity {np.mean(get.h)*1e-9*params['shear_rate']:.2f} m/s")
        if 'correlate' in args.qtty:
            get.uncertainty_pDiff(pump_size=0.1)
        if 'stension' in args.qtty:
            gamma = np.mean(get.surface_tension()['gamma'])
            print(f'Surface tension (gamma) = {gamma*1e3:.4f} mN/m')
        if 'contact_angle' in args.qtty:
            gamma_sv_gamma_ls = np.mean(get.surface_tension()['gamma'])*1e3
            gamma_lv = 12.1320 # mN/m (At 300K for pentane)
            contact_angle = np.degrees(np.arccos(-gamma_sv_gamma_ls/gamma_lv))
            print(f'Wetting angle (θ) = {contact_angle:.4f} degrees')
        if 'Re' in args.qtty:
            Reynolds = get.reynolds_num()
            print(f'Reynolds number: {Reynolds}')


    if 'viscosity_gk' in args.qtty:
        # viscosity with correlation time plots for average of all replicates
        eta_avg_time = np.mean(eta, axis=0)
        # average of all replicates
        np.savetxt(f'eta_{len(datasets):02d}.txt', np.c_[time*data['thermo_out'], eta_avg_time],  delimiter=' ',
                        header='time   viscosity')
        # average viscosity
        eta_avg = np.mean(eta_avg_time[start:stop])
        print(f'Dynamic Viscosity (η) = {eta_avg:.3f} mPa.s')


    if 'conductivity_gk' in args.qtty:
        # viscosity with correlation time plots for average of all replicates
        lambda_avg_time = np.mean(lambda_time, axis=0)
        # average of all replicates
        np.savetxt(f'lambda_{len(datasets):02d}.txt', np.c_[time*data['thermo_out'], lambda_avg_time],  delimiter=' ',
                        header='time   conductivity')
        # average viscosity
        lambda_avg = np.mean(lambda_avg_time[start:stop])
        print(f'Thermal conductivity (λ) = {lambda_avg:.3f} W/mK')

#
