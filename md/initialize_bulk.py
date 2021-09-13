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
import init_bulk

def get_parser():
    parser = argparse.ArgumentParser(
    description='Initializes a simulation box with a desired mass and\
                 number density by modifying the Lammps input file "init.LAMMPS" ')

    #Positional arguments
    parser.add_argument('density', metavar='density', action='store', type=float,
                    help='The mass density in (g/cm^3)')
    parser.add_argument('Np', metavar='Np', action='store', type=float,
                    help='The no. of particles to insert')
    parser.add_argument('name', metavar='fluid', action='store',
                    help='The molecule file name')
    # parser.add_argument('forcefield', metavar='Force field file', action='store',
    #                 help='The Force field file name')

    # sub-command
    subparsers = parser.add_subparsers(help='choose initialization code', dest='code')
    parser_lammps = subparsers.add_parser('lammps')
    parser_moltemp = subparsers.add_parser('moltemp')

    return parser


if __name__ == '__main__':

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

    if args.code == 'moltemp':
        init_bulk.init_moltemp(args.density, args.Np, args.name, mFluid,
                                                        tolX, tolY, tolZ)
        q = input('run moltemp: ')
        if q == 'y':
            subprocess.call(['./setup.sh'], shell=True)

    elif args.code == 'lammps':
        init_bulk.init_lammps(args.density, args.Np, args.name, mFluid)
        q = input('run lammps: ')
        if q == 'y':
            subprocess.call(['mpirun -np 8 lmp_mpi -in $(pwd)/init.LAMMPS'], shell=True)

    else:
        raise NameError('Provide the code')

    file = open('args.txt', 'a')
    file.write(f'Parameters : {args}')
    file.close()
