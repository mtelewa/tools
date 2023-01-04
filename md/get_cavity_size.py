#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys, os
import numpy as np
from cavitation_size import CavitySize
from progressbar import progressbar
import netCDF4

# np.seterr(all='raise')

def get_parser():
    parser = argparse.ArgumentParser(
    description='Plot quantities post-processed from a netcdf trajectory.')

    #Positional arguments
    #--------------------
    parser.add_argument('start', metavar='start', action='store', type=int,
                    help='Start from this frame')
    parser.add_argument('stop', metavar='stop', action='store', type=int,
                    help='Discard after this frame')
    parser.add_argument('nevery', metavar='nevery', action='store', type=int,
                    help='Ovito will process every nth frame')
    parser.add_argument('rprobe', metavar='rprobe', action='store', type=float,
                    help='The fluid to perform the analysis on.  \
                        Needed for the calculation of mass flux and density')
    parser.add_argument('shape', metavar='shape', action='store', type=str,
                    help='Void shape (sphere/cylinder)')
    parser.add_argument('region', metavar='region', action='store', type=str,
                    help='Timeframe to measure the bubble radius (collapse/full)')
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
                if i == 'flow.nc' or i == 'equilib.nc' or i == 'nvt.nc':
                    trajactories.append(os.path.join(root, i))

    for i in range(len(trajactories)):
        data = netCDF4.Dataset(trajactories[i])
        Time = data.variables["time"]
        step_size = Time.scale_factor

        dataset = CavitySize(trajactories[i], args.start, args.stop, args.nevery, args.rprobe, args.shape)

        time_list, radius_list = [], []

        # # Run the data pipleine
        for idx,val in enumerate(dataset.frames_to_process):
            progressbar(idx, len(dataset.frames_to_process))
            data = dataset.pipeline.compute(val)
            r = dataset.void_surface_area(val, data)['radius']
            v = dataset.void_surface_area(val, data)['volume']
            # area = void_surface_area(i,data)['area']
            time_list.append(val * dataset.lammps_dump_every * step_size)
            if args.region == 'collapse':
                if r.size==0:
                    radius_list.append(0)
                elif r.size==1:
                     radius_list.append(r[0])
                else:
                    for i in r:
                        if np.isclose(i, radius_list[-1], rtol=0.1):
                            a = i
                    radius_list.append(a)

            if args.region == 'full':
                if r.size==0:
                    radius_list.append(0)
                elif r.size==1:
                     radius_list.append(r[0])
                else:
                    radius_list.append(np.max(r))
                    # else:
                    # if val*dataset.lammps_dump_every<1.5908e06: #5.1e5: # Time where the two bubble curves meet
                    # if len(r)>1:
                    #     radius_list.append(0)
                    # elif isclose(np.min(r), radius_list[-1], rel_tol=0.2, abs_tol=0.0):
                    #     # if len(r)>1:
                    #     radius_list.append(np.min(r))
                    # else:
                    #     radius_list.append(0)

                # print(radius_list)
        np.savetxt(f'radius_{idx}.txt', np.c_[time_list, radius_list],  delimiter=' ',\
                                        header='Time (fs)              Radius (A)')

        with open(f'rargs_{idx}.txt', 'w') as myfile:
            myfile.write(f'Start from frame:{args.start}, Discard after:{args.stop}, Print every {args.nevery} frames,\n\
            Probe radius:{args.rprobe} Angstrom \n\
            Shape: {args.shape} \n\
            Region: {args.region}')
