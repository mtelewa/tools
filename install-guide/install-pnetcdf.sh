PREFIX=$HOME/.local

# pnetcdf installation
#module load mpi/openmpi/3.1-gnu-7.3
wget https://parallel-netcdf.github.io/Release/pnetcdf-1.11.0.tar.gz
tar -pizxvf pnetcdf-1.11.0.tar.gz
cd pnetcdf-1.11.0
# Configure without C++ support
MPICC=mpicc MPICXX=mpicxx ./configure --disable-fortran --disable-cxx --enable-shared --enable-relax-coord-bound --prefix=$PREFIX/pnetcdf-1.11.0
make -j 2
make install

