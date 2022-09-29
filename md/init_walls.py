#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, re, sys
import warnings
import scipy.constants as sci
import logging
import fileinput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.set_printoptions(threshold=sys.maxsize)

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
offset = 5.0              # Initial offset bet. solid and liquid to avoid atoms/molecules overlapping
Boffset = 1.0			   # Initial offset bet. the block and the simulation box
ls = 4.08                  # Lattice spacing for fcc gold

def init_moltemp(nUnitsX, nUnitsY, nUnitsZ, h, density, name, mFluid, tolX, tolY, tolZ):
    """
    Initializes a simulation box with specified dimensions and fluid mass density
    The script writes the "geometry.lt" which is then imported by "system.lt" in moltemplate.
    """
    #derived dimensions
    unitlengthX = ls * np.sqrt(6) / 2.
    xlength = nUnitsX * unitlengthX				# Block length

    unitlengthY = ls * np.sqrt(2) / 2.
    ylength = nUnitsY * unitlengthY        	    # Block width

    unitlengthZ = round(ls * np.sqrt(3))
    zlength = h + 2*offset + Boffset + 2*nUnitsZ*unitlengthZ 	# Block height (starts from -Boffset)
    totalboxHeight = zlength + Boffset

    print(f'lx={xlength:.2f},ly={ylength:.2f},lz={totalboxHeight:.2f}')

    gapHeight = h + 2*offset
    Nfluid= round(sci.N_A * density * xlength * ylength * gapHeight * 1.e-24 / mFluid)  # No. of fluid atoms

    if name=='lj':
        print('At rho=%g g/cm^3, Nf=%g atoms shall be created' %(density,Nfluid/4.0))
    else:
        print('At rho=%g g/cm^3, Nf=%g molecules shall be created' %(density,Nfluid))

    # Regions
    #---------
    # Fluid region
    fluidStartZ = nUnitsZ*unitlengthZ + offset          # Lower bound in the Z-direction
    fluidEndZ = h + fluidStartZ       # Upper bound for initialization

    # Upper wall region
    surfUStartZ = fluidEndZ + offset
    surfUEndZ = surfUStartZ + nUnitsZ*unitlengthZ

    # No. of fluid atoms in each direction
    # TODO: Fix the Nfluid to be exactly whats obtained from the previous eq.
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

    Nz = int(Nfluid / (Nx * Ny))
    if Nz * tolZ > gapHeight:
        Nz = 0
        k = 1
        while k < gapHeight:
            k = Nz * tolZ
            Nz += 1
        Nz = Nz-3
        Nfluid = Nx * Ny * Nz
        density = Nfluid / (sci.N_A * xlength * ylength * gapHeight * 1.e-24 / mFluid)
        warnings.warn(f"For the chosen density, atoms will be outside the box otherwise there will be overlap. \n\
               Density was reduced to {density:.2f}")

    Nfluid_created = Nx * Ny * Nz
    print('Created %g molecules by moltemplate' %Nfluid_created)
    diff = Nfluid - Nfluid_created
    add_molecules,remove_molecules = 0,0

    if diff > 0:
        while add_molecules < diff:
            add_molecules+=1
        logger.warning(f" ===> Add {add_molecules} Molecules to reach the required density\
        A deviation of {100-(Nfluid_created*100/Nfluid):.3f}% from the desired value")

    elif diff < 0:
        while remove_molecules < abs(diff):
            remove_molecules+=1
        logger.warning(f" ===> Remove {remove_molecules} Molecules to reach the required density\
        A deviation of {100-(Nfluid_created*100/Nfluid):.3f}% from the desired value")

    else:
        print(f'Created {Nfluid_created} molecules corresponds to bulk density of {density} g/cm^3 successfully!')

    # ## TODO: Modify Nx Ny Nz automatically based on re-evaluation of the total no.
    # if Nfluid_created / Nfluid <= 0.98:   # If the error in the number of desired atoms deviates from the actual created by 2%
    #     print(f'Current Nx is {Nx} and Nz is {Nz}')
    #     Nx = Nx + np.int(input('add/subtract to/from Nx:'))
    #     Nz = Nz + np.int(input('add/subtract to/from Nz:'))
    #
    # Nfluid_mod = Nx * Ny * Nz
    # diff2 = Nfluid - Nfluid_mod
    # add_molecules_mod, remove_molecules_mod = 0,0
    #
    # if diff2 > 0:
    #     while add_molecules_mod < diff2:
    #         add_molecules_mod+=1
    #     logger.warning(f" ===> Added {Nfluid_mod-Nfluid_created} molecules. Still need to add {add_molecules_mod} molecules after modification")
    #
    # elif diff < 0:
    #     while remove_molecules < abs(diff2):
    #         remove_molecules+=1
    #     logger.warning(f" ===> Remove {remove_molecules_mod} Molecules to reach the required density")
    #
    # print(f'Created {Nfluid_mod:g} molecules after modification with a deviation\
    # of {100-(Nfluid_mod*100/Nfluid):.3f}% from the desired value')


    # if 'fluid_walls' in sys.argv:

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
    # delete mol[14][0-9][0-8] \n\
    \n\
    # import wall building block file \n\
    import 'gold.lt' \n\
    \n\
    solidU = new gold  [{nUnitsX}].move({unitlengthX},0,0) \n\
                       [{nUnitsY}].move(0,{unitlengthY},0) \n\
                       [{nUnitsZ}].move(0,0,{unitlengthZ}) \n\
    \n\
    solidL = new gold [{nUnitsX}].move({unitlengthX},0,0)\n\
                      [{nUnitsY}].move(0,{unitlengthY},0) \n\
                      [{nUnitsZ}].move(0,0,{unitlengthZ}) \n\
    \n\
    # Shift the Upper layer from the origin in the z-direction. \n\
    solidU[*][*][*].move(0.0,0.0,{(surfUStartZ+Boffset)}) \n\
    \n\
    # Shift the fluid atoms from the box center \n\
    mol[*][*][*].move({tolX},{Boffset},{(fluidStartZ+Boffset)}) \n\
    \n\
    # The lower layer is not shifted \n\
    solidL[*][*][*].move(0.0,0.0,{Boffset}) \n\
    \n\
    write_once('Data Boundary'){{ \n\
     0    {(nUnitsX*unitlengthX)}    xlo xhi \n\
     0    {(nUnitsY*unitlengthY)}    ylo yhi \n\
    {(Boffset*-1)}  {zlength}   zlo zhi \n\
    }}\n"


    in_script_liquid_only= f" # System moltemplate file\n\
    #------------------------\n\
    \n\
    # import molecule building block file\n\
    import '{name}.lt' \n\
    \n\
    mol = new {name}   [{Nx}].move({tolX},0,0) \n\
                       [{Ny}].move(0,{tolY},0) \n\
                       [{Nz}].move(0,0,{tolZ}) \n\
    \n\
    # delete mol[2-17][0-7][0-2] \n\
    \n\
    # import wall building block file\n\
    import 'gold_all.lt'  \n\
    \n\
    gold = new au   \n\
    \n\
    # Shift the fluid atoms from the box center \n\
    mol[*][*][*].move({tolX},{Boffset},{(fluidStartZ+Boffset)}) \n\
    \n\
    write_once('Data Boundary'){{ \n\
     0    {(nUnitsX*unitlengthX)}    xlo xhi \n\
     0    {(nUnitsY*unitlengthY)}    ylo yhi \n\
    {(Boffset*-1)}  {zlength}   zlo zhi \n\
    }}"

    in_f=open('geometry.lt','w')
    if os.path.exists('gold_all.lt'):   # The walls were already created with LAMMPS
        in_f.write(in_script_liquid_only)
    else:
        in_f.write(in_script)
    in_f.close()



def init_lammps(nUnitsX, nUnitsY, nUnitsZ, h, density, mFluid):
    """
    Initializes a simulation box with specified dimensions and fluid mass density
    The script modifies the Lammps input file "init.LAMMPS".
    """
    unitlengthZ = round(ls * np.sqrt(3))

    xlength= nUnitsX * ls * np.sqrt(6) /2.				    # Block length
    ylength= nUnitsY * ls * np.sqrt(2) /2.         	    # Block width
    zlength= h + 2*offset + Boffset + 2*unitlengthZ*nUnitsZ 	                # Block height (starts from -Boffset)

    gapHeight = h + 2*offset
    totalboxHeight = zlength + Boffset
    Nfluid = round(sci.N_A * density * xlength * ylength * gapHeight * 1.e-24 / mFluid)  # No. of fluid atoms

    # print('lx=%g,ly=%g,lz=%g' %(xlength,ylength,zlength))
    #print('Nf=%g' %Nfluid)

    # Regions
    #---------
    # Fluid region
    fluidStartZ = nUnitsZ*unitlengthZ + offset          # Lower bound in the Z-direction
    fluidEndZ = h + fluidStartZ       # Upper bound for initialization
    # Upper wall region
    surfLEndZ = nUnitsZ * unitlengthZ
    # Upper wall region
    surfUStartZ = fluidEndZ + offset
    surfUEndZ = surfUStartZ + nUnitsZ*unitlengthZ

    # Modify the 'init.LAMMPS' ----------
    for line in open('init.LAMMPS','r').readlines():
        line = re.sub(r'variable       xlength equal.+',
                      r'variable       xlength equal %.2f' %xlength, line)
        line = re.sub(r'variable       zlength equal.+',
                      r'variable       zlength equal %.2f' %zlength, line)
        line = re.sub(r'create_atoms    1 random.+',
                      r'create_atoms    1 random %g 206649 fluid' %Nfluid, line)
        line = re.sub(r'region          box block.+',
                      r'region          box block 0.0 %.2f 0.0 %.2f %.2f %.2f units box' %(xlength, ylength, -Boffset, zlength), line)
        line = re.sub(r'region          fluid block.+',
                      r'region          fluid block INF INF INF INF %.2f %.2f units box' %(fluidStartZ, fluidEndZ), line)
        line = re.sub(r'region          surfL block.+',
                      r'region          surfL block 1e-5 INF INF INF -1e-5 %.2f units box' %(surfLEndZ), line)
        line = re.sub(r'region          surfU block.+',
                      r'region          surfU block 1e-5 INF INF INF %.2f %.2f units box' %(surfUStartZ, surfUEndZ), line)
        # For the converging-diverging channel (to empty a region in the middle - the gap height)
        line = re.sub(r'variable        surfU_div_begin equal.+',
                      r'variable        surfU_div_begin equal %.2f' %surfUStartZ, line)
        line = re.sub(r'variable        th equal.+',
                      r'variable        th equal %.2f' %(unitlengthZ*nUnitsZ), line)


        fout = open("init2.LAMMPS", "a")
        fout.write(line)

    os.rename("init2.LAMMPS", "init.LAMMPS")
