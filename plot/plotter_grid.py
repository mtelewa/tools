#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging
import argparse

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
    if args.fluid=='squalane': mf = 422.83

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

    # Quick plotting is performed when no config file is provided
    if args.config:
        import plot_from_grid
    else:
        import plot_quick

    datasets = []
    for key, value in vars(args).items():
        if key.startswith('ds') and value!='allds':
            datasets.append(os.path.abspath(value))
        # plot all datasets in a certain directory
        if key.startswith('ds') and value=='allds':
            for i in os.listdir(os.getcwd()):
                if os.path.isdir(i):
                    datasets.append(f"{os.path.abspath(os.getcwd())+ '/' + i}")
        # plot all datasets in multiple directories
        if key.startswith('allds'):
            for i in os.listdir(value):
                a = os.path.abspath(f"{value+ '/' +i}")
                if os.path.isdir(a):
                    datasets.append(a)

    if len(datasets) < 0.:
        logging.error("No Datasets Found! Make sure the Path to the datset is correct")
        quit()

    # Order as indexed on the FileSystem. Sorting is needed if all datasets are
    # points on the same curve e.g. EOS
    if 'allds' in vars(args).values() or 'allds' in vars(args).keys():
        datasets.sort()

    datasets_x, datasets_z = [], []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i.endswith(f'_{args.nChunks}x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'_1x{args.nChunks}.nc'):
                    datasets_z.append(os.path.join(root, i))

    if not datasets_x:
        logging.error("No Files Found! Make sure the dataset is processed and the dataset name is correct.")
        quit()

    if args.config:
        pg = plot_from_grid.PlotFromGrid(args.skip, dim, datasets_x, datasets_z, mf, args.config, pumpsize)
        pg.extract_plot(args.variables)
        for ax in pg.axes_array:
            ax.set_rasterized(True)
    else:
        # Quantity profiles with the 'quick plot' module
        pg = plot_quick.plot_from_ds(args.skip, datasets_x, datasets_z, mf, pumpsize)
        if dim=='L': pg.variable_len(args.variables,legend='y', opacity=0.8)
        if dim=='H': pg.variable_height(args.variables, legend='y', opacity=0.8)
        if dim=='T': pg.variable_time(args.variables, legend='y', opacity=0.3)
        pg.ax.set_rasterized(True)

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
