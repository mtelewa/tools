#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, logging
import argparse
import plot_from_txt

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
    fig = ptxt.extract_plot(args.variables)

    if args.format is not None:
        if args.format == 'eps':
            fig.savefig(args.variables[0]+'.eps' , format='eps', transparent=True)
        if args.format == 'ps':
            fig.savefig(args.variables[0]+'.ps' , format='ps')
        if args.format == 'svg':
            fig.savefig(args.variables[0]+'.svg' , format='svg')
        if args.format == 'pdf':
            fig.savefig(args.variables[0]+'.pdf' , format='pdf')
    else:
        fig.savefig(args.variables[0]+'.png' , format='png')
