#!/usr/bin/env python3

import netCDF4
import numpy as np
import time
from mpi4py import MPI

# 1. Reading/Writing ----------------------

# Reading data
data_in = netCDF4.Dataset('nvt.nc')
# NETCDF3_64BIT_DATA data model
# allow for unsigned/64 bit integer data types and 64-bit dimension sizes.

# print(data_in.data_model)
# print(data_in)

# Wrting data
data_out = netCDF4.Dataset('test.nc','w',format='NETCDF4')
# NETCDF4 data model
# use the version 4 disk format (HDF5) and use the new features of the version 4 API

# print(data_out.data_model)
# print(data_out)

# 2. Groups -----------------------------

# Groups serve as containers for variables, dimensions and attributes, as well as other groups

# Only NETCDF4 formatted files support Groups, this will give an error
# grp = data_in.createGroup("forecasts")
# But this not
# grp = data_out.createGroup("forecasts")
#
# def walktree(top):
#     values = top.groups.values()
#     yield values
#     for value in top.groups.values():
#          for children in walktree(value):
#              yield children
#
# print(data_out)
#
# for children in walktree(data_out):
#     for child in children:
#          print(child)

# 3. Dimensions -----------------------------

# netCDF defines the sizes of all variables in terms of dimensions
# so before any variables can be created the dimensions they use must be created first
# only scalars are exemption
# All of the Dimension instances are stored in a python dictionary
# A dimension is created using the createDimension method of a Dataset or Group instance.
# A Python string is used to set the name of the dimension, and an integer value is used to set the size.


lat = data_out.createDimension("lat", None)

# dictionary
# print(data_out.dimensions)
# the variable length
# print(len(var1))
# check if variable dimension is unlimited
# print(var1.isunlimited())

# print a summary on the dimensions of variables in the netCDF data
# for dimobj in data_in.dimensions.values():
#     print(dimobj)

for dimobj in data_in.dimensions.items():
    print(dimobj)

# 4. Variables -----------------------------

# Behave much like python multidimensional array objects supplied by numpy
# netCDF4 variables can be appended to along one or more 'unlimited' dimensions
# The createVariable method has two mandatory arguments,
# the variable name (a Python string), and the variable datatype
# To create a scalar variable, simply leave out the dimensions keyword.

# Primitive datatypes correspond to the dtype attribute of a numpy array
# Valid datatype specifiers include:
# 'f4' (32-bit floating point),
# 'f8' (64-bit floating point),
# 'i4' (32-bit signed integer),
# 'i2' (16-bit signed integer),
# 'i8' (64-bit signed integer),
# 'i1' (8-bit signed integer),
# 'u1' (8-bit unsigned integer),
# 'u2' (16-bit unsigned integer),
# 'u4' (32-bit unsigned integer),
# 'u8' (64-bit unsigned integer),
# 'S1' (single-character string).

# Get a summary of variables
for varobj in data_in.variables.values():
    print(varobj)
# Python dictionary for variables
# print(data_in.variables)

# i:time step, j: atoms, k: cartesian coords
# print(varobj.shape)
# No. of atoms
# print(len(varobj[810]))
# print the variable values
vels = np.array(data_in.variables['velocities'])
# velocities in the first and last time steps
# print(vels[0],vels[810])

print(data_in.variables['cell_lengths'].units)

#variable dimensions
print(data_in.variables['velocities'].dimensions)
# and shape attribute
print(data_in.variables['velocities'].shape)

# get all variable names
# print(data_in.variables.keys())

# Variable names can be changed using the renameVariable method of a Dataset instance.

latitudes = data_out.createVariable("lat","f4",("lat",))

# 5. Attributes -----------------------------

# Global and variable attributes:
# Global attributes provide information about a group, or the entire dataset, as a whole.
# Variable attributes provide information about one of the variables in a group.
# Attributes can be strings, numbers or sequences

# str=data_out.history = "Created " + time.ctime(time.time())
# print(str)

# for name in data_in.ncattrs():
#     print("Global attr {} = {}".format(name, getattr(data_in, name)))

# print(data_in.__dict__)

# multi-dimensional array attributes not supported, this will generate an error
# data_in.setncattr("velocities", vels)

data_out.setncattr("lat",latitudes)

# for name in data_out.ncattrs():
    # print("Global attr {} = {}".format(name, getattr(data_out, name)))

# 6. Writing data to and retrieving data from a netCDF variable -------

# Unlike NumPy's array objects, netCDF Variable objects with unlimited dimensions will grow along
# those dimensions if you assign data outside the currently defined range of indices.

lats =  np.arange(-90,91,2.5)
latitudes[:] = lats

# 8. Reading data from a multi-file netCDF dataset ---------------------

# create a MFDataset instance with either a list of filenames, or a string with a wildcard

# from netCDF4 import MFDataset
# f = MFDataset("mftest*nc")

# 9.  Efficient compression of netCDF variables ----------------------------

# Data stored in netCDF 4 Variable objects can be compressed and decompressed on the fly.
# parameters for the compression are determined by the zlib, complevel and shuffle

#To turn on compression, set zlib=True.

#complevel regulates the speed and efficiency of the compression
#(1 being fastest, but lowest compression ratio, 9 being slowest but best compression ratio).

# If this line was used to create the variabe, less disk space will be used
# temp = rootgrp.createVariable("temp","f4",("time","level","lat","lon",),
# zlib=True,least_significant_digit=3)

# 13. Parallel IO.

# when a new dataset is created or an existing dataset is opened, use the parallel
# keyword to enable parallel access.

# Parallel mode requires MPI enabled netcdf-c
# nc = netCDF4.Dataset('parallel_test.nc','w',parallel=True)

#The optional comm keyword may be used to specify a particular MPI communicator
#(MPI_COMM_WORLD is used by default).
