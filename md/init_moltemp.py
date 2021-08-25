#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import re
import sys
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
t = 7.0                    # Thickness of gold layer
offset = 5.0               # Initial offset bet. solid and liquid to avoid atoms/molecules overlapping
Boffset = 1.0			   # Initial offset bet. the block and the simulation box
ls = 4.08                  # Lattice spacing for fcc gold

rootdir = os.getcwd()


def initialize_walls(nUnitsX, nUnitsY, nUnitsZ, h, density, name, mFluid, tolX, tolY, tolZ):

    #derived dimensions
    unitlengthX = ls * np.sqrt(6) / 2
    xlength = nUnitsX * unitlengthX				# Block length
    unitlengthY = ls * np.sqrt(2) / 2
    ylength = nUnitsY * unitlengthY        	    # Block width
    unitlengthZ = round(ls * np.sqrt(3))
    zlength = h + 2*offset + Boffset + 2*t 	# Block height (starts from -Boffset)
    # print('lx=%g,ly=%g,lz=%g' %(xlength,ylength,zlength))
    gapHeight = h + 2*offset
    # print(gapHeight)
    totalboxHeight = zlength + Boffset
    Nfluid= round(sci.N_A * density * xlength * ylength * gapHeight * 1.e-24 / mFluid)  # No. of fluid atoms
    print('At rho=%g g/cm^3, Nf=%g molecules shall be created' %(density,Nfluid))

    # Regions
    #---------
    # Fluid region
    fluidStartZ = t + offset          # Lower bound in the Z-direction
    fluidEndZ = h + fluidStartZ       # Upper bound for initialization

    # Upper wall region
    surfUStartZ = fluidEndZ + offset
    surfUEndZ = surfUStartZ + t

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
        warnings.warn("For the chosen density, atoms will be outside the box otherwise there will be overlap. \n\
               Density was reduced to %.2f" %density)

    new_Nfluid = Nx * Ny * Nz
    diff = Nfluid - new_Nfluid

    # Nx = 34 # heptane
    # Nx= 59  # propane
    # Nz = 8   # propane

    Nfluid_mod = Nx * Ny * Nz
    diff2 = Nfluid - Nfluid_mod

    ## TODO: Modify Nx Ny Nz automatically based on re-evaluation of the total no.

    print('Created %g molecules by moltemplate' %new_Nfluid)
    print('Created %g molecules after modification' %Nfluid_mod)

    add_molecules,remove_molecules=0,0
    add_molecules_mod = 0

    if diff > 0:
        while add_molecules < diff:
            add_molecules+=1
        logger.warning(" ===> Add {0} Molecules to reach the required density".format(add_molecules))

        while add_molecules_mod < diff2:
            add_molecules_mod+=1
        logger.warning(" ===> Add {0} Molecules after modification".format(add_molecules_mod))

    elif diff < 0:
        while remove_molecules < abs(diff):
            remove_molecules+=1
        logger.warning(" ===> Remove {0} Molecules to reach the required density".format(remove_molecules))

    else:
        print('Created {} molecules corresponds to bulk density of {} g/cm^3 successfully!'.format(new_Nfluid,density))

    # a = tolZ*Nz

    in_script= " # System moltemplate file\n\
    #------------------------\n\
    \n\
    # import molecule building block file \n\
    import '{19}.lt' \n\
    \n\
    # Replicate the pentane, the value in [] is the no. of replicas (a) \n\
    # the value in () is the offset (b) between replicas. \n\
    # To avoid atoms creation outside of the box, a*b < xhi. Same for y and z. \n\
    mol = new {19}   [{12}].move({13},0,0) \n\
                     [{14}].move(0,{15},0) \n\
                     [{16}].move(0,0,{17}) \n\
    #                    ^          ^ ^ ^ \n\
    #                    Nf      offsetX,offsetY,offsetZ \n\
    # delete mol[14][0-9][0-8] \n\
    # import wall building block file \n\
    import 'gold.lt' \n\
    \n\
    solidU = new gold  [{0}].move({1},0,0) \n\
                       [{2}].move(0,{3},0) \n\
                       [{4}].move(0,0,{5}) \n\
    #                    ^        ^ ^ ^ \n\
    #                    Ns      unitX,unitY,unitZ \n\
    \n\
    solidL = new gold [{0}].move({1},0,0)\n\
                      [{2}].move(0,{3},0) \n\
                      [{4}].move(0,0,{5}) \n\
    \n\
    # Shift the Upper layer from the origin in the z-direction. \n\
    solidU[*][*][*].move(0.0,0.0,{6}) \n\
    #                             ^ \n\
    #                        surfUStartZ+Boffset \n\
    \n\
    # Shift the fluid atoms from the box center \n\
    mol[*][*][*].move({13},{7},{18}) \n\
    #                         ^ \n\
    #                    fluidStartZ+Boffset \n\
    \n\
    # The lower layer is not shifted \n\
    solidL[*][*][*].move(0.0,0.0,{7}) \n\
    #                             ^ \n\
    #                           Boffset \n\
    \n\
    write_once('Data Boundary'){{ \n\
     0    {8}    xlo xhi \n\
     0    {9}    ylo yhi \n\
    {10}  {11}   zlo zhi \n\
    }}\n".format( #0            #1          #2          #3
                 nUnitsX,unitlengthX,nUnitsY,unitlengthY,
                   #4           #5              #6              #7
                nUnitsZ,unitlengthZ,(surfUStartZ+Boffset),Boffset,
                            #8                          #9
                 (nUnitsX*unitlengthX),(nUnitsY*unitlengthY),
                        #10     #11   #12  #13 #14 #15 #16 #17    #18
                 (Boffset*-1),zlength,Nx,tolX,Ny,tolY,Nz,tolZ,(fluidStartZ+Boffset),
                 #19
                name,Nz-1)
    in_f=open('geometry.lt','w')
    in_f.write(in_script)
    in_f.close()

    # Modify the header ----------

    fout = open("../blocks/header2.LAMMPS", "w+")

    for line in open('../blocks/header.LAMMPS','r').readlines():
      line = re.sub(r'h index.+',r'h index %g' %h, line)
      line = re.sub(r'nUnitsX index.+',r'nUnitsX index %g' %nUnitsX, line)
      line = re.sub(r'nUnitsY index.+',r'nUnitsY index %g' %nUnitsY, line)
      fout.write(line)

    os.rename("../blocks/header2.LAMMPS", "../blocks/header.LAMMPS")

if __name__ == '__main__':
    main(sys.argv[1:])

    file= open('args.txt', 'a')
    file.write('Parameters : {}'.format(sys.argv[1:]))
    file.close()

    # for line in fileinput.input("args.txt", inplace=True):
    #     if line.split()[0]=='Parameters':
    #         print(' ', end='')
    #     else:
    #         print('{}'.format(line), end='')
