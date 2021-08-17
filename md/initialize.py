#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input:
------
Fluid topology: pentane.lt, Wall topology: gold.lt
Output:
-------
geometry.lt
"""

import argparse
import init_moltemp

def get_parser():
    parser = argparse.ArgumentParser(
    description='Initializes a simulation box with specified dimensions \
                and fluid mass density. The script also modifies LAMMPS header file.')

    #Positional arguments
    #--------------------
    parser.add_argument('nUnitsX', metavar='nUnitsX', action='store', type=int,
                    help='Unit cells in the x-direction')
    parser.add_argument('nUnitsY', metavar='nUnitsY', action='store', type=int,
                    help='Unit cells in the y-direction')
    parser.add_argument('nUnitsZ', metavar='nUnitsZ', action='store', type=int,
                    help='Unit cells in the z-direction for each wall')
    parser.add_argument('h', metavar='height', action='store', type=float,
                    help='Fluid height (A)')
    parser.add_argument('density', metavar='rho_fluid', action='store', type=float,
                    help='The fluid density in (g/cm^3)')
    parser.add_argument('name', metavar='fluid', action='store', help='The molecule name')

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.name == 'pentane':
        mFluid = 72.15
        tolX, tolY, tolZ = 10 , 4 , 3
    elif args.name == 'propane':
        mFluid = 44.09
        tolX, tolY, tolZ = 5 , 3 , 3
    elif args.name == 'heptane':
        mFluid = 100.21
        tolX, tolY, tolZ = 10 , 4 , 3

    init_moltemp.initialize_walls(args.nUnitsX, args.nUnitsY, args.nUnitsZ,
                                 args.h, args.density, args.name,
                                 mFluid, tolX, tolY, tolZ)
