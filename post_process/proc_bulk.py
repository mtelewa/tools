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
import grid_bulk as grid


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
    parser.add_argument('--Ny', metavar='Ny', action='store', type=int,
                    help='Number of chunks in the z-direcion, default is 1')
    parser.add_argument('--tessellate', metavar='tessellate', action='store', type=int,
                    help='Perform Delaunay tessellation, default is 0')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.fluid=='lj': A_per_molecule = 1
    if args.fluid=='propane': A_per_molecule = 3
    if args.fluid=='pentane': A_per_molecule = 5
    if args.fluid=='heptane': A_per_molecule = 7
    if args.fluid=='squalane': A_per_molecule = 30

    grid.make_grid(args.infile, args.Nx, args.Ny, args.Nz, args.slice_size, A_per_molecule, args.tessellate)
