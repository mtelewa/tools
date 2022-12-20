#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging
import argparse
import plot_general

# Logger Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    parser.add_argument('variables', metavar='variables', nargs='+', action='store', type=str,
                    help='the vaiable(s) to plot')
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
            # pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str,
                                help='datasets for plotting')
        if arg.startswith(("-all")):
            # plot all datasets in multiple directories
            parser.add_argument(arg.split('=')[0], type=str,
                                help='directory with the datasets for plotting')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    # Set the fluid molecular mass
    if args.fluid=='lj': mf = 39.948
    if args.fluid=='propane': mf = 44.09
    if args.fluid=='pentane': mf = 72.15
    if args.fluid=='heptane': mf = 100.21

    # Dimension to plot against
    if '_length' in args.variables[0]: dim='L'
    if '_height' in args.variables[0]: dim='H'
    if '_time' in args.variables[0]: dim='T'

    # Get the pump size to calculate the pressure gradient
    if not args.pumpsize:
        pumpsize = 0        # equilib (Equilibrium), sd (shear-driven)
        logging.info('Pump size is set to zero!')
    else:
        pumpsize = args.pumpsize  # pd (pressure-driven), sp (superposed)
        logging.info(f'Pump size is set to {pumpsize} Lx.')

    datasets = []
    for key, value in vars(args).items():
        if key.startswith('ds') and value!='all':
            datasets.append(os.path.abspath(value))
        # plot all datasets in a certain directory
        if key.startswith('ds') and value=='all':
            for i in os.listdir(os.getcwd()):
                if os.path.isdir(i):
                    datasets.append(f"{os.path.abspath(os.getcwd())+ '/' + i}")
        # plot all datasets in multiple directories
        if key.startswith('all'):
            for i in os.listdir(value):
                a = os.path.abspath(f"{value+ '/' +i}")
                if os.path.isdir(a):
                    datasets.append(a)

    if len(datasets) < 0.:
        logging.error("No Datasets Found! Make sure the Path to the datset is correct")
        quit()

    # Order as indexed on the FileSystem. Sorting is needed if all datasets are
    # points on the same curve e.g. EOS
    if 'all' in vars(args).values():
        datasets.sort()
    for i in vars(args).keys():
        if i.startswith('all'):
            datasets.sort()

    datasets_x, datasets_z = [], []
    for k in datasets:
        for root, dirs, files in sorted(os.walk(k)):
            for i in files:
                if i.endswith(f'{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(root, i))

    if not datasets_x:
        logging.error("No Files Found! Make sure the dataset is processed")
        quit()

    if args.config == None:
        logging.error('Insert Config file!')
        quit()

    pg = plot_general.PlotGeneral(args.skip, datasets_x, datasets_z, mf, args.config, pumpsize)

    # Velocity Distibution
    if 'distrib' in args.variables: pg.v_distrib()
    # Velocity evolution
    if 'evolution' in args.variables: pg.v_evolution()
    # Transport Coefficients
    if 'pgrad_mflowrate' in args.variables: pg.pgrad_mflowrate()
    if 'rate_viscosity' in args.variables: pg.rate_viscosity()
    if 'rate_stress' in args.variables: pg.rate_stress()
    if 'rate_slip' in args.variables: pg.rate_slip()
    if 'rate_temp' in args.variables: pg.rate_temp()
    if 'rate_qdot' in args.variables: pg.rate_qdot()
    if 'rate_lambda' in args.variables: pg.rate_lambda()
    # if 'pt_ratio' in args.variables[0]: pg.pt_ratio()
    if 'eos' in args.variables: pg.eos()
    if 'lambda' in args.variables: pg.thermal_conduct()
    # ACFs
    if 'acf' in args.variables: pg.acf()
    if 'transverse' in args.variables: pg.transverse()
    if 'sk' in args.variables: pg.struc_factor()
    if 'isf' in args.variables: pg.isf()
    # Coexistence curves
    if 'coexist' in args.variables: pg.coexistence_curve()
    if 'rc_gamma' in args.variables: pg.rc_gamma()
    if 'temp_gamma' in args.variables: pg.temp_gamma()

    if args.format is not None:
        if args.format == 'eps':
            pg.fig.savefig(args.variables[0]+'.eps' , format='eps', transparent=True)
        if args.format == 'ps':
            pg.fig.savefig(args.variables[0]+'.ps' , format='ps')
        if args.format == 'svg':
            pg.fig.savefig(args.variables[0]+'.svg' , format='svg')
        if args.format == 'pdf':
            pg.fig.savefig(args.variables[0]+'.pdf' , format='pdf')
    else:
        pg.fig.savefig(args.variables[0]+'.png' , format='png')
