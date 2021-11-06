## For Traditional make with machine MPI

# Link with python virtual environment
# Activate the venv

`source $HOME/fireworks/bin/activate`

# Because of the dynamic loading, it is required that LAMMPS is compiled in “shared” mode. 
# Go to lammps/src dir and run

`make mode= shlib mpi`

# I had the voronoi package installed and this created a conflict since the voro lib was 
# static in order to change that, I had to recompile the voro package with the option
# -fPIC in the CFLAGS

# Inside the voro++ dir, add `-fPIC` to the CFLAGS in the config file then run make again
