#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys, os
from cavitation_size import CavitySize
import progressbar

def get_parser():
    parser = argparse.ArgumentParser(
    description='Plot quantities post-processed from a netcdf trajectory.')

    #Positional arguments
    #--------------------
    parser.add_argument('nevery', metavar='nevery', action='store', type=int,
                    help='Ovito will process every nth frame')
    parser.add_argument('rprobe', metavar='rprobe', action='store', type=float,
                    help='The fluid to perform the analysis on.  \
                        Needed for the calculation of mass flux and density')
    # Get the datasets
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-ds")):
            # pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str,
                                help='datasets for plotting')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    datasets = []
    for key, value in vars(args).items():
        if key.startswith('ds'):
            datasets.append(os.path.abspath(value))

    trajactories = []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i == 'flow.nc':
                    trajactories.append(os.path.join(root, i))

    for i in range(len(trajactories)):
        dataset = CavitySize(trajactories[0], args.nevery, args.rprobe)

        time_list, radius_list = [], []
        # # Run the data pipleine
        for i in dataset.frames_to_process:
            # progressbar(i,len(dataset.frames_to_process))
            data = dataset.pipeline.compute(i)
            r = dataset.void_surface_area(i,data)['radius']
            # area = void_surface_area(i,data)['area']
            time_list.append(i*dataset.lammps_dump_every)
            if r.size==0:
                radius_list.append(0)
            else:
                radius_list.append(np.max(r))

        # TODO: Get the simulation nucleation rate
        # J_sim = tau*vol

        np.savetxt('radius.txt', np.c_[time_list, radius_list],  delimiter=' ',\
                                        header='Time (fs)              Radius (A)')
