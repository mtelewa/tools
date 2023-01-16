#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input:
------
NetCDF tajectory
Output:
-------
NetCDF file(s) with AMBER convention. Each file represents a timeslice
Time slices are then merged with `cdo mergetime` or `cdo merge` commands.
"""
import sys
import argparse
import grid


def get_parser():
    parser = argparse.ArgumentParser(
    description='Extract data from equilibrium and non-equilibrium NetCDF trajectories \
     by spatial binning or "chunking" along a dimension')

    #Positional arguments
    #--------------------
    parser.add_argument('infile', metavar='infile', action='store', type=str,
                    help='MD trajectory file name')
    parser.add_argument('Nx', metavar='Nx', action='store', type=int,
                    help='Number of chunks in the x-direcion')
    parser.add_argument('Nz', metavar='Nz', action='store', type=int,
                    help='Number of chunks in the z-direcion')
    parser.add_argument('slice_size', metavar='slice_size', action='store', type=int,
                    help='Number of time steps in a single slice')
    parser.add_argument('fluid', metavar='Fluid', action='store', type=str,
                    help='The fluid in the simulation')
    parser.add_argument('stable_start', metavar='stable_start', action='store', type=float,
                    help='Multiple of Lx, where the stable region (for sampling) starts')
    parser.add_argument('stable_end', metavar='stable_end', action='store', type=float,
                    help='Multiple of Lx, where the stable region (for sampling) ends')
    parser.add_argument('pump_start', metavar='pump_start', action='store', type=float,
                    help='Multiple of Lx, where the pump region (pertrubation field) starts')
    parser.add_argument('pump_end', metavar='pump_end', action='store', type=float,
                    help='Multiple of Lx, where the pump region (pertrubation field) ends')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.fluid=='lj':
        mf, A_per_molecule = 39.948, 1
    if args.fluid=='propane':
        mf, A_per_molecule = 44.09, 3
    if args.fluid=='pentane':
        mf, A_per_molecule = 72.15, 5
    if args.fluid=='heptane':
        mf, A_per_molecule = 100.21, 7
    if args.fluid=='triacontane':
        mf, A_per_molecule = 422.83, 30
    if args.fluid=='squalane':
        mf, A_per_molecule = 422.83, 30

    grid.make_grid(args.infile, args.Nx, args.Nz, args.slice_size, mf, A_per_molecule, args.fluid,
                    args.stable_start, args.stable_end, args.pump_start, args.pump_end)
