#!/bin/sh

# Clean up from old run
shopt -s extglob
rm out/*
rm log.lammps

# Create out directory
if [ -d "out" ]
then
    echo "Directory out exists." 
else
    mkdir out
fi

# Compute the virial if needed
read -p "Get virial? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
     mpirun -np 16 lmp_mpi -i $1 -v minimize 0 -v get_virial 1
else
     mpirun -np 16 lmp_mpi -i $1 -v minimize 0 -v get_virial 0
fi


