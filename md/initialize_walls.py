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
import subprocess
import init_walls

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

    # sub-command
    subparsers = parser.add_subparsers(help='choose initialization code', dest='code')
    parser_lammps = subparsers.add_parser('lammps')
    parser_moltemp = subparsers.add_parser('moltemp')

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
    elif args.name == 'lj':
        mFluid = 39.948

    # Initialize either by modifying the LAMMPS init file or running moltemplate
    if args.code == 'moltemp':
        init_walls.init_moltemp(args.nUnitsX, args.nUnitsY, args.nUnitsZ,
                                     args.h, args.density, args.name,
                                     mFluid, tolX, tolY, tolZ)
        q = input('run moltemp: ')
        if q == 'y':
            subprocess.call(['cd moltemp ; ./setup.sh'], shell=True)

    else:
        init_walls.init_lammps(args.nUnitsX, args.nUnitsY, args.nUnitsZ,
                                     args.h, args.density, mFluid)
        q = input('run lammps: ')
        if q == 'y':
            subprocess.call(['mpirun -np 8 lmp_mpi -in $(pwd)/init.LAMMPS'], shell=True)


    file = open('args.txt', 'w')
    file.write(f'Parameters : {args}')
    file.close()
