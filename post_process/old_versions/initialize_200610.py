import numpy as np
import os
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import argparse
#import this

# Avogadro's no. (1/mol)
NA = 6.02214e23
rootdir = os.getcwd()

def main(args):
    parser = argparse.ArgumentParser(
    description='Initializes a simulation box with \
                 a desired mass and number density')

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
    #parser.add_argument('z', metavar='Z', action='store', type=float,
    #                help='Height (Angstrom)')

# Previous calls to add_argument() determine exactly what objects are created and how they are assigned
    args = parser.parse_args(args)
    print(args)

#def get_N():

    # Input Mass density
    density_si = args.density

    # Input Volume
    #cellX = args.x
    #cellY = args.y
    #cellZ = args.z

    # Mass density convert to per A^3
    mass_density_real = float(density_si)*1e-24
    # Number density (#/A^3)
    num_density_real = mass_density_real*NA/args.mass
    # Volume (A^3)
    volume_real = args.Np*args.mass*1e24/(args.density*NA)
    # No. of molecules
    #N = round(num_density_real*volume_real)
    # No. of moles
    #mol = mass_real/molecular_mass
    # no. of molecules
    #N = round(mol*NA)
    #print ('No of particles to create ', N)
    print (volume_real)

    cellX = volume_real**(1/3)
    cellY = volume_real**(1/3)
    cellZ = volume_real**(1/3)

    #return (N)


# # -------------Script 2---------------
#
#
# #molar_volume = input('Volume (cm^3/mol): ')
#
# #density_needed = molecular_mass/float(molar_volume)  # g/cm^3
#
# #Number_of_molecules = density_needed*2315/0.673         # At N=2315, rho=0.673 g/cm^3
#
# Number_of_molecules = density*NA/molecular_mass_argon         # At N=2315, rho=0.673 g/cm^3
#
# #print (Number_of_molecules)
#
# N_int = int(Number_of_molecules)
#
#
# def replace(file_path, pattern, subst):
#     #Create temp file
#     fh, abs_path = mkstemp()
#     with fdopen(fh,'w') as new_file:
#         with open(file_path) as old_file:
#             for line in old_file:
#                 new_file.write(line.replace(pattern, subst))
#     #Copy the file permissions from the old file to the new file
#     copymode(file_path, abs_path)
#     #Remove original file
#     remove(file_path)
#     #Move new file
#     move(abs_path, file_path)
#
# #os.chdir('data_files')
# #for subdir, dirs, files in os.walk('data_files'):
# #    for subdirs in subdir:
# #        print(os.path.join(subdir))
#
#
# replace("%s/initial.in" %os.getcwd(), "create_atoms    0 random x 206649 fluid mol pentane 175649", "create_atoms    0 random %i 206649 fluid mol pentane 175649" %N_int)
# os.system('mpirun -np 16 lmp_mpi -i initial.in -v nCHx 5 -v velo gauss -v walls 0')
#
# directory=input('Move data file to: ')
# os.system('mv data.init %s' %directory)
#
# replace("%s/initial.in" %os.getcwd(), "create_atoms    0 random %i 206649 fluid mol pentane 175649" %N_int, "create_atoms    0 random x 206649 fluid mol pentane 175649")

    bond_length=1.54  # between carbons
    offset_mol_mol = 2.
    carbons_in_mol=5         # pentane

    tolX = (bond_length*carbons_in_mol)+offset_mol_mol
    tolY = round(3.6)
    tolZ = round(2.8)

# Initialize with moltemplate

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

      # # Thermodynamic output
      #   out.write('write_once("In Output"){' + '\n'
      #             'thermo 1000' + '\n'
      #             'thermo_style custom step temp ke epair ebond eangle edihed eimp pe etotal density vol press' + '\n'
      #             'dump 1 all custom 1000 out/dump.lammpstrj id type x y z vx vy vz fx fy fz' + '\n' + '}' + '\n')
      # # The simulation
      #   out.write('write_once("In Run"){' + '\n'
      #             'fix nve all nve/limit 0.1' + '\n'
      #             'fix berendsen all temp/berendsen 313.15 313.15 100.0' + '\n' + 'run 10000' + '\n ' +
      #             'unfix nve' + '\n ' +  'unfix berendsen' + '\n ' + 'fix nvt all nvt temp 313.15 313.15 100.0' + '\n ' +
      #             'timestep 1.0' + '\n ' + 'run 500000' + '\n ' + 'thermo_modify flush yes' + '\n' + '}' + '\n' )

    out.close()

if __name__ == '__main__':
    main(sys.argv[1:])
