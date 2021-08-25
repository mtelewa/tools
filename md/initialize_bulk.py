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
import init_bulk

def get_parser():
    parser = argparse.ArgumentParser(
    description='Initializes a simulation box with a desired mass and\
                 number density by modifying the Lammps input file "init.LAMMPS" ')

    #Positional arguments
    parser.add_argument('mass', metavar='M', action='store', type=float,
                    help='Molecular Mass (g/mol)')
    parser.add_argument('density', metavar='rho', action='store', type=float,
                    help='The input density in (g/cm^3)')
    parser.add_argument('Np', metavar='N', action='store', type=float,
                    help='The no. of particles to insert')
    parser.add_argument('molecule', metavar='Molecule file', action='store',
                    help='The molecule file name')
    parser.add_argument('forcefield', metavar='Force field file', action='store',
                    help='The Force field file name')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()


    init_lammps.init_bulk(args.nUnitsX, args.nUnitsY, args.nUnitsZ,
                         args.h, args.density)


    file = open('args.txt', 'a')
    file.write(f'Parameters : {args}')
    file.close()
