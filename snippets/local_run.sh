#!/bin/sh

# Clean up from old run
#shopt -s extglob
if [ -f "log.lammps" ]
then
    rm log.lammps
else
    :
fi

# Create out directory
if [ -d "out" ]
then
    echo "Directory out exists."
    rm out/*
else
    mkdir out
fi

mpirun -np 16 lmp_mpi -i init.LAMMPS -v get_virial 1

# Equilibrate (compute the virial if needed)
read -p "Get virial? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
     mpirun -np 16 lmp_mpi -i $1 -v get_virial 1
else
     mpirun -np 16 lmp_mpi -i $1 -v get_virial 0
fi
