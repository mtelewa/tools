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


def init_moltemp(density, Np, name, mass, tolX, tolY, tolZ):

    # Number density (#/A^3)
    num_density = density * 1e-24 * sci.N_A / mass
    # Volume (A^3)
    volume = Np * mass * 1e24 / (density * sci.N_A)

    xlength = volume**(1/3)
    ylength = volume**(1/3)
    zlength = volume**(1/3)

    bond_length = 1.54  # between carbons
    offset_mol_mol = 2.
    carbons_in_mol = 5         # pentane

    # tolX = (bond_length * carbons_in_mol) + offset_mol_mol
    # tolY = round(3.6)
    # tolZ = round(2.8)

    # No. of atoms in each direction
    Nx, Ny = 0, 0

    i = 1
    while i < xlength:
        i = Nx * tolX
        Nx += 1
    Nx = Nx-2

    j = 1
    while j < ylength:
        j = Ny * tolY
        Ny += 1
    Ny = Ny - 2

    Nz = int(Np / (Nx * Ny))


    new_Np = Nx * Ny * Nz
    diff = Np - new_Np

    ## TODO: Modify Nx Ny Nz automatically based on re-evaluation of the total no.
    # if new_Nfluid/Nfluid <= 98:
    #     Nz+=1
    #     Nx-=14
    #
    Np_mod = Nx * Ny * Nz
    diff2 = Np - Np_mod

    print('Created %g molecules by moltemplate' %new_Np)
    print('Created %g molecules after modification' %Np_mod)

    # Move to the center
    molecule_center = (carbons_in_mol-1) * bond_length / 2.
    Boffset = 1


    in_script= f" # System moltemplate file\n\
    #------------------------\n\
    \n\
    # import molecule building block file \n\
    import '{name}.lt' \n\
    \n\
    # Replicate the pentane, the value in [] is the no. of replicas (a) \n\
    # the value in () is the offset (b) between replicas. \n\
    # To avoid atoms creation outside of the box, a*b < xhi. Same for y and z. \n\
    mol = new {name}   [{Nx}].move({tolX},0,0) \n\
                       [{Ny}].move(0,{tolY},0) \n\
                       [{Nz}].move(0,0,{tolZ}) \n\
    \n\
    mol[*][*][*].move({tolX/2},{tolY},{tolZ*2})\n\
    \n\
    write_once('Data Boundary'){{ \n\
     0    {xlength}   xlo xhi \n\
     0    {ylength}   ylo yhi \n\
     0    {zlength}   zlo zhi \n\
    }}\n"

    # Note: The boudnaries have to start from zero, otherwise problems in post-processing.

    in_f=open('geometry.lt','w')
    in_f.write(in_script)
    in_f.close()


def init_lammps(density, Np, name, mass):

    num_density = density * 1e-24 * sci.N_A / mass # Number density (#/A^3)
    volume = Np * mass * 1e24 / (density * sci.N_A)  #(A^3)

    cellX = volume**(1/3)
    cellY = volume**(1/3)
    cellZ = volume**(1/3)

    for line in open('init.LAMMPS','r').readlines():
        line = re.sub(r'region          box block.+',
                      r'region          box block 0.0 %.2f 0.0 %.2f 0.0 %.2f units box' \
                            %(cellX, cellY, cellZ), line)
        if name == 'pentane':
            line = re.sub(r'create_atoms    0 random.+',
                          r'create_atoms    0 random %i 206649 NULL mol pentane 175649' %Np, line)
        if name == 'lj':
            line = re.sub(r'create_atoms    1 random.+',
                          r'create_atoms    1 random %i 206649 fluid' %Np, line)
        fout = open("init2.LAMMPS", "a")
        fout.write(line)

    os.rename("init2.LAMMPS", "init.LAMMPS")
