#!/usr/bin/env python
import numpy as np
import os
import re
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import argparse
import sys
#import this

# Avogadro's no. (1/mol)
NA = 6.02214e23
rootdir = os.getcwd()

def main(args):
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

# Previous calls to add_argument() determine exactly what objects are created
# and how they are assigned
    args = parser.parse_args(args)
    print(args)

#def get_N():

    # Input Mass density
    density_si = args.density

    # Mass density convert to per A^3
    mass_density_real = float(density_si)*1e-24
    # Number density (#/A^3)
    num_density_real = mass_density_real*NA/args.mass
    # Volume (A^3)
    volume_real = args.Np*args.mass*1e24/(args.density*NA)

    #print (volume_real)

    cellX = volume_real**(1/3)
    cellY = volume_real**(1/3)
    cellZ = volume_real**(1/3)

    #print(cellX,cellY,cellZ)

    for line in open('init.LAMMPS','r').readlines():
        line = re.sub(r'region          box block.+',
                      r'region          box block %.2f %.2f %.2f %.2f %.2f %.2f units box' %(-cellX/2.,cellX/2.,-cellY/2.,cellY/2.,-cellZ/2.,cellZ/2.), line)
        line = re.sub(r'create_atoms    0 random.+',
                      r'create_atoms    0 random %i 206649 NULL mol pentane 175649' %args.Np, line)
        fout = open("init2.LAMMPS", "a")
        fout.write(line)

    os.rename("init2.LAMMPS", "init.LAMMPS")

#replace("%s/initial.in" %os.getcwd(), "create_atoms    0 random %i 206649 fluid
# mol pentane 175649" %N_int, "create_atoms    0 random x 206649 fluid mol pentane 175649")

if __name__ == '__main__':
    main(sys.argv[1:])
