#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging
import argparse
import plot_from_txt
import netCDF4
import numpy as np

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
    parser.add_argument('filename', metavar='filename', action='store', type=str,
                    help='first few characters of the filename')
    parser.add_argument('variables', metavar='variables', nargs='+', action='store', type=str,
                    help='the variable(s) to plot')
    parser.add_argument('--format', metavar='format', action='store', type=str,
                    help='format of the saved figure. Default: PNG')
    parser.add_argument('--config', metavar='config', action='store', type=str,
                    help='Config Yaml file to import plot settings from')

    # Get the datasets
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        # you can pass any arguments to add_argument
        if arg.startswith(("-ds")):
            parser.add_argument(arg.split('=')[0], type=str,
                                help='files for plotting')
        if arg.startswith(("-all")):
            # plot all datasets in multiple directories
            parser.add_argument(arg.split('=')[0], type=str,
                                help='directory with the datasets for plotting')
    return parser



if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

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

    txts = []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in sorted(files):
                if i.startswith(args.filename):
                    txts.append(os.path.join(root, i))

    if not txts:
        logging.error("No Files Found! Make sure the Filename is correct")
        quit()

    ptxt = plot_from_txt.PlotFromTxt(args.skip, args.filename, txts, args.config)

    datasets_list, datasets_x, datasets_z = [], [], []
    for g in datasets:
        for root, dirs, files in os.walk(g):
            for i in files:
                if i.endswith(f'144x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'1x144.nc'):
                    datasets_z.append(os.path.join(root, i))
                if i.endswith(f'50x50x1.nc'):
                    datasets_list.append(os.path.join(root, i))

    trajactories = []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i == 'flow.nc' or i == 'equilib.nc' or i == 'nvt.nc':
                    trajactories.append(os.path.join(root, i))

    if 'log' in args.filename and 'eos' not in args.variables: ptxt.extract_plot(args.variables)
    if 'log' in args.filename and 'nvt_eos' in args.variables: ptxt.nvt_eos()
    if 'eta' in args.filename: ptxt.GK()
    if 'pvisco' in args.filename or 'pconduct' in args.filename: ptxt.transport_press()
    if 'press-profile' in args.filename and 'press' in args.variables: ptxt.press_md_cont()
    if 'press-profile' in args.filename and 'virial' in args.variables: ptxt.virial()
    if 'virialChunkX' in args.filename and 'virial' in args.variables: ptxt.virial()
    if 'temp' in args.filename: ptxt.temp()
    if 'radii' in args.variables: ptxt.radii()
    if 'radius' in args.variables:
        method = 'static'
        if method == 'static':
            data = netCDF4.Dataset(trajactories[0])
            Time = data.variables["time"]
            time_arr = np.array(Time).astype(np.float32)
            # Time between output dumps
            dt = np.int32(time_arr[-1] - time_arr[-2])
            r0, v0, pl, start, stop = None, None, None, 0, None

            ptxt.radius(r0, v0, pl, datasets_list, start, stop, dt, method)

        elif method == 'dynamic':
            # iter-12-thick:
            # r0, v0, pl, start, stop = 33.6e-10, 0, -, 100, 180    35e-10, 405, 455, 'dynamic'
            # iter-12-thin:
            # r0, v0, pl, start, stop = 16e-10, 0, -, 115, 138
            # iter-16:
            if args.filename == 'radius_cavity_collapse':
                r0, v0, pl, start, stop = 19.8e-10, 0, 73.564e3, 0, None #651, 702 # cavity collapse
            if args.filename == 'radius_cavity_full':
                r0, v0, pl, start, stop = 0.1e-10, 5.0e3, 73.579e3, 0, None # cavity nucleation and collapse    73.579e3        73.564e3
            dt = None
            ptxt.radius(r0, v0, pl, datasets, start, stop, dt, method)
        else:
            r0, v0, pl, start, stop = None, None, None, None, None # cavity radius in whole trajectory
            dt = None

            ptxt.radius(r0, v0, pl, datasets, start, stop, dt, method)

    if args.format is not None:
        if args.format == 'eps':
            ptxt.fig.savefig(args.variables[0]+'.eps' , format='eps', transparent=True)
        if args.format == 'ps':
            ptxt.fig.savefig(args.variables[0]+'.ps' , format='ps')
        if args.format == 'svg':
            ptxt.fig.savefig(args.variables[0]+'.svg' , format='svg')
        if args.format == 'pdf':
            ptxt.fig.savefig(args.variables[0]+'.pdf' , format='pdf')
    else:
        ptxt.fig.savefig(args.variables[0]+'.png' , format='png')
