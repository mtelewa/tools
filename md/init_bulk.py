#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, re, sys
import scipy.constants as sci

# Unit conversion & constants
#----------------------------
N_to_kcalpermolA= 1.4393263121*1e10
kcalpermolA_to_N= 6.947694845598684e-11
A2_to_m2= 1e-20
mpa_to_kcalpermolA3= 0.00014393
kcalpermolA3_to_mpa= 1/mpa_to_kcalpermolA3
atm_to_pa= 101325

# GEOMETRY
#---------
t= 7.0                    # Thickness of gold layer
offset= 5.0               # Initial offset bet. solid and liquid to avoid atoms/molecules overlapping
Boffset= 1.0			   # Initial offset bet. the block and the simulation box
ls= 4.08                  # Lattice spacing for fcc gold

def init_moltemp(Np, density, mass):

    # Input Mass density
    density_si = args.density

    # Number density (#/A^3)
    num_density_real = float(density_si) * 1e-24 * sci.N_A / args.mass
    # Volume (A^3)
    volume_real = args.Np * args.mass * 1e24 / (args.density*NA)

    cellX = volume_real**(1/3)
    cellY = volume_real**(1/3)
    cellZ = volume_real**(1/3)

    #return (N)

    bond_length=1.54  # between carbons
    offset_mol_mol = 2.
    carbons_in_mol=5         # pentane

    tolX = (bond_length*carbons_in_mol)+offset_mol_mol
    tolY = round(3.6)
    tolZ = round(2.8)

    # No. of atoms in each direction
    Npx=0
    i=1
    while i<cellX:
        i=Npx*tolX
        Npx+=1
    Npx=Npx-2

    Npy=0
    j=1
    while j<cellY:
        j=Npy*tolY
        Npy+=1
    Npy=Npy-2

    Npz=int(args.Np/(Npx*Npy))

    with open('system.lt', 'w+') as out:
        #Header
        out.write( '# system.lt' + '\n')
        # Box dimensions
        out.write('\n' + 'write_once("Data Boundary"){' + '\n' \
              '%.2f %.2f xlo xhi' %(int(-cellX/2.),int(cellX/2.)) + '\n' + \
              '%.2f %.2f ylo yhi' %(int(-cellY/2.),int(cellY/2.)) + '\n' + \
              '%.2f %.2f zlo zhi' %(int(-cellZ/2.),int(cellZ/2.)) + '\n' + '}' + '\n')
        # Box boundary and variable `cutoff` required by GROMOS_54A7_ATB.lt
        out.write('\n' + 'write_once("In Init"){' + '\n'
             'variable cutoff equal 14.0 # Angstroms' + '\n' +
             'boundary p p p' + '\n' + '}' + '\n')
        # Import the forcefield and the molecule building block files
        if args.forcefield:
            out.write('import "%s"' %args.forcefield + '\n' +'import "%s"' %args.molecule)
        else:
            out.write('import "%s"' %args.molecule)

        # Create the periodic structure
        out.write('\n' +
             'mol = new GROMOS_54A7_ATB/N5UF [{0}].move({1},0,0)'.format(Npx,tolX) + '\n'
             #'mol = new pentane [{1}].move({0},0,0)'.format(Npx,tolX) + '\n'
     	 		                             '[{0}].move(0,{1},0)'.format(Npy,tolY) + '\n'
                                             '[{0}].move(0,0,{1})'.format(Npz,tolZ) + '\n')

        # if (Npx*Npy*Npz)-args.Np!=0:
        #    Nadd=args.Np-(Npx*Npy*Npz)
        #    for i in range(int(Nadd)):
        #        out.write('mol_%g = new GROMOS_54A7_ATB/N5UF [1].move(0,0,%g)' %(i,(tolZ*12)) + '\n')

      # Move to the center
        molecule_center=(carbons_in_mol-1)*bond_length/2.
        offset_mol_box = 2.

        shift_x = -(cellX/2)+molecule_center+offset_mol_box
        shift_y = -(cellY/2.)+offset_mol_box
        shift_z = -(cellZ/2.)+offset_mol_box

        out.write('mol[*][*][*].move({0},{1},{2})'.format(shift_x,shift_y,shift_z) + '\n')

    out.close()


# TODO: Expand for LJ fluid
def init_lammps(Np, density, mass):

    # Number density (#/A^3)
    num_density_real = float(density)* 1e-24 * sci.N_A / mass
    # Volume (A^3)
    volume_real = Np * mass * 1e24 / (density * sci.N_A)

    cellX = volume_real**(1/3)
    cellY = volume_real**(1/3)
    cellZ = volume_real**(1/3)

    for line in open('init.LAMMPS','r').readlines():
        line = re.sub(r'region          box block.+',
                      r'region          box block %.2f %.2f %.2f %.2f %.2f %.2f units box' %(-cellX/2.,cellX/2.,-cellY/2.,cellY/2.,-cellZ/2.,cellZ/2.), line)
        line = re.sub(r'create_atoms    0 random.+',
                      r'create_atoms    0 random %i 206649 NULL mol pentane 175649' %Np, line)
        fout = open("init2.LAMMPS", "a")
        fout.write(line)

    os.rename("init2.LAMMPS", "init.LAMMPS")
