# On NEMO
#PREFIX=/work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/local_modules/netcdf/4.6.3
# To install into the local home
PREFIX=$HOME/.local

# pnetcdf installation
#module load mpi/openmpi/3.1-gnu-7.3
wget https://parallel-netcdf.github.io/Release/pnetcdf-1.11.0.tar.gz
tar -pizxvf pnetcdf-1.11.0.tar.gz
cd pnetcdf-1.11.0
# Configure without shared library support
MPICC=mpicc MPICXX=mpicxx ./configure --disable-fortran --disable-cxx --enable-shared --prefix=$PREFIX
make -j 2
make install

cd ..

#hdf5 installation
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz
tar -zvxf hdf5-1.10.5.tar.gz
cd hdf5-1.10.5
CC=mpicc CXX=mpicxx ./configure --enable-parallel --prefix=$PREFIX
make -j 2
make install

cd ..

# netcdf installation
wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-4.6.3.tar.gz
tar -zvxf netcdf-c-4.6.3.tar.gz
# Note on building NetCDF: If a system installed HDF5 is present, it may link by default to this library.
# Specify the library path with the -L option explicitly (in LDFLAGS) to link to the correct library.
HDF5_LIB_PATH=$(which h5ls | sed 's,bin/h5ls,lib,')
cd netcdf-c-4.6.3
CC=mpicc CXX=mpicxx LDFLAGS=-L$HDF5_LIB_PATH ./configure --enable-pnetcdf --enable-netcdf-4 --enable-parallel4 --enable-parallel-tests --prefix=$PREFIX
make -j 2
make install

# python netCDF4 installation
cd ..
wget https://files.pythonhosted.org/packages/6b/c2/264ceea72738fdd51b32d772f5ef35cbde7f6d7eeb50edae28f751f68711/netCDF4-1.4.3.2.tar.gz
tar -zvxf netCDF4-1.4.3.2.tar.gz
cd netCDF4-1.4.3.2
python3 setup.py build --force
python3 setup.py install --user
