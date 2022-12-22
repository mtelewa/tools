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
    return parser



if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    datasets = []
    for key, val in vars(args).items():
        if key.startswith('ds'):
            datasets.append(val)

    if len(datasets) < 0.:
        logging.error("No Datasets Found! Make sure the Path to the datset is correct")
        quit()

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


    datasets_x, datasets_z = [], []
    for g in datasets:
        for root, dirs, files in os.walk(g):
            for i in files:
                if i.endswith(f'144x1.nc'):
                    datasets_x.append(os.path.join(root, i))
                if i.endswith(f'1x144.nc'):
                    datasets_z.append(os.path.join(root, i))

    trajactories = []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i == 'flow.nc' or i == 'equilib.nc' or i == 'nvt.nc':
                    trajactories.append(os.path.join(root, i))

    if 'log' in args.filename: ptxt.extract_plot(args.variables)
    if 'press-profile' in args.filename: ptxt.press_md_cont(args.variables)
    if 'radius' in args.variables:
        # iter-12-thick:
        # r0, v0, pl, start, stop, method = 33.6e-10, 0, -, 100, 180, 'dynamic'    35e-10, 405, 455, 'dynamic'
        # iter-12-thin:
        # r0, v0, pl, start, stop, method = 16e-10, 0, -, 115, 138, 'dynamic'
        # iter-16:
        # r0, v0, pl, start, stop, method = 19.8e-10, 0, 1.1e5, 651, 702, 'dynamic' # (collapse one cavity)
        r0, v0, pl, start, stop, method = 0.1e-10, 5e3, 1.01e5, 0, 702, 'dynamic' #(nucleation and collapse one cavity)
        if method == 'static':
            data = netCDF4.Dataset(trajactories[0])
            Time = data.variables["time"]
            time_arr = np.array(Time).astype(np.float32)
            # Time between output dumps
            dt = np.int32(time_arr[-1] - time_arr[-2])
        else:
            dt = None

        ptxt.radius(r0, v0, pl, datasets_x, datasets_z, start, stop, dt, method)

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
