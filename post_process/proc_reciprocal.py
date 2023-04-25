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
import grid_reciprocal


def get_parser():
    parser = argparse.ArgumentParser(
    description='Extract data from equilibrium and non-equilibrium NetCDF trajectories \
     by spatial binning or "chunking" along a dimension')

    #Positional arguments
    #--------------------
    parser.add_argument('infile', metavar='infile', action='store', type=str,
                    help='MD trajectory file name')
    parser.add_argument('nx', metavar='nx', action='store', type=int,
                    help='Number of wavevectors in the longitudnal direcion')
    parser.add_argument('ny', metavar='ny', action='store', type=int,
                    help='Number of wavevectors in the transverse direcion')
    parser.add_argument('slice_size', metavar='slice_size', action='store', type=int,
                    help='Number of time steps in a single slice')
    parser.add_argument('fluid', metavar='Fluid', action='store', type=str,
                    help='The fluid in the simulation')
    parser.add_argument('fluid_start', metavar='fluid_start', action='store', type=float,
                    help='Value of Lx, where the fluid region of interest starts')
    parser.add_argument('fluid_end', metavar='stable_end', action='store', type=float,
                    help='Value of Lx, where the fluid region of interest ends')
    parser.add_argument('solid_start', metavar='pump_start', action='store', type=float,
                    help='Value of Lx, where the solid region of interest starts')
    parser.add_argument('solid_end', metavar='pump_end', action='store', type=float,
                    help='Value of Lx, where the solid region of interest ends')
    parser.add_argument('--nz', metavar='nz', action='store', type=int,
                    help='Number of discrete wave vectors in the z-direcion, default is 1')
    parser.add_argument('--TW_interface', metavar='TW_interface', action='store', type=int,
                    help='Define the location of vibrating atoms in the upper wall \
                        Default = 1: thermostat applied on the wall layers in contact with the fluid')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.fluid=='lj': mf, A_per_molecule = 39.948, 1
    if args.fluid=='propane': mf, A_per_molecule = 44.09, 3
    if args.fluid=='pentane': mf, A_per_molecule = 72.15, 5
    if args.fluid=='heptane': mf, A_per_molecule = 100.21, 7
    if args.fluid=='squalane': mf, A_per_molecule = 422.83, 30

    grid_reciprocal.make_grid(args.infile, args.nx, args.ny, args.nz, args.slice_size, mf, A_per_molecule, args.fluid,
                    args.fluid_start, args.fluid_end, args.solid_start, args.solid_end, args.TW_interface)
