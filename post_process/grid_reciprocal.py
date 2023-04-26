#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, warnings
import numpy as np
import scipy.constants as sci
import time as timer
from mpi4py import MPI
import netCDF4
import processor_reciprocal as pnc
from operator import itemgetter
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.seterr(all='raise')

# # Get the processor version
version = 'reciprocal'

def make_grid(infile, nx, ny, nz, slice_size, mf, A_per_molecule, fluid, fluid_start, fluid_end, solid_start, solid_end, TW_interface):

    infile = comm.bcast(infile, root=0)
    data = netCDF4.Dataset(infile)
    nx, ny = comm.bcast(nx, root=0), comm.bcast(ny, root=0)
    slice_size = comm.bcast(slice_size, root=0)

    # Timesteps
    Time = data.variables["time"]
    time_arr=np.array(Time).astype(np.float32)

    step_size = Time.scale_factor
    tSteps_tot = Time.shape[0]-1
    out_frequency = np.int( (time_arr[-1] - time_arr[-2]) / step_size)
    total_sim_time = tSteps_tot * out_frequency * step_size

    # If the dataset is more than 1000 tsteps, slice it to fit in memory
    if tSteps_tot <= 1000:
        Nslices = 1
    else:
        Nslices = tSteps_tot // slice_size

    Nslices = np.int(Nslices)
    slice1, slice2 = 1, slice_size+1

    if rank == 0:
        print('Total simualtion time: {} {}'.format(np.int(total_sim_time), Time.units))
        print('======> The dataset will be sliced to %g slices!! <======' %Nslices)

    t0 = timer.time()

    for slice in range(Nslices):

        if tSteps_tot <= 1000:
            # Exclude the first snapshot in the trajectory
            time = Time[slice1:].shape[0]
            time_array = np.array(Time[slice1:]).astype(np.float32)
        else:
            time = slice_size
            time_array = np.array(Time[slice1:slice2]).astype(np.float32)

        # Chunk the data: each processor has a time chunk of the data
        nlocal = time // size          # no. of tsteps each proc should handle
        start = (rank * nlocal) + (time*slice) + 1
        end = ((rank + 1) * nlocal) + (time*slice) + 1

        if rank == size - 1:
            nlocal += time % size
            start = time - nlocal + (time*slice) + 1
            end = time + (time*slice) + 1

        chunksize = end - start

        # Postproc class construct
        init = pnc.TrajtoGrid(data, start, end, nx, ny, nz, mf, A_per_molecule, fluid, TW_interface)

        # Get the data
        cell_lengths, kx, ky, kz, \
        gap_heights, com, sf, sf_solid, rho_k, Nf, Nm = itemgetter('cell_lengths',
                                                                  'kx', 'ky', 'kz',
                                                                  'gap_heights',
                                                                  'com',
                                                                  'sf',
                                                                  'sf_solid',
                                                                  'rho_k',
                                                                  'Nf', 'Nm')(init.get_chunks(fluid_start,
                                                                   fluid_end, solid_start, solid_end))

        # Number of elements in the send buffer
        sendcounts_time = np.array(comm.gather(chunksize, root=0))
        sendcounts_chunk_fluid_layer_3d = np.array(comm.gather(sf.size, root=0))
        sendcounts_chunk_solid_layer_3d = np.array(comm.gather(sf_solid.size, root=0))

        if rank == 0:

            print('Sampled time: {} {}'.format(np.int(time_array[-1]), Time.units))

            # Dimensions: (time,)
            # Gap Heights
            gap_height_global = np.zeros(time, dtype=np.float32)
            gap_height_conv_global = np.zeros_like(gap_height_global)
            gap_height_div_global = np.zeros_like(gap_height_global)
            # Center of Mass
            com_global = np.zeros_like(gap_height_global)

            # Dimensions: (time, nx, ny)
            # Density Fourier coefficients
            rho_k_global = np.zeros([time, nx, ny] , dtype=np.complex64)
            # Structure factor
            sf_global = np.zeros([time, nx, ny] , dtype=np.complex64)
            sf_solid_global = np.zeros([time, nx, ny] , dtype=np.complex64)

        else:
            gap_height_global = None
            gap_height_conv_global = None
            gap_height_div_global = None
            com_global = None
            sf_global = None
            sf_solid_global = None
            rho_k_global = None

        comm.Gatherv(sendbuf=gap_heights[0], recvbuf=(gap_height_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=gap_heights[1], recvbuf=(gap_height_conv_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=gap_heights[2], recvbuf=(gap_height_div_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=com, recvbuf=(com_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=sf, recvbuf=(sf_global, sendcounts_chunk_fluid_layer_3d), root=0)
        comm.Gatherv(sendbuf=sf_solid, recvbuf=(sf_solid_global, sendcounts_chunk_solid_layer_3d), root=0)
        comm.Gatherv(sendbuf=rho_k, recvbuf=(rho_k_global, sendcounts_chunk_fluid_layer_3d), root=0)

        if rank == 0:

            # Write to netCDF file  ------------------------------------------
            outfile = f"{infile.split('.')[0]}_{nx}x{ny}_{slice:0>3}.nc"
            out = netCDF4.Dataset(outfile, 'w', format='NETCDF3_64BIT_OFFSET')

            out.setncattr("Version", version)
            # Reciprocal lattice
            out.createDimension('nx', nx)
            out.createDimension('ny', ny)
            out.createDimension('time', time)

            # Real lattice
            out.setncattr("Lx", cell_lengths[0])
            out.setncattr("Ly", cell_lengths[1])
            out.setncattr("Lz", cell_lengths[2])

            time_var = out.createVariable('Time', 'f4', ('time'))
            gap_height_var =  out.createVariable('Height', 'f4', ('time'))
            gap_height_conv_var =  out.createVariable('Height_conv', 'f4', ('time'))
            gap_height_div_var =  out.createVariable('Height_div', 'f4', ('time'))
            com_var =  out.createVariable('COM', 'f4', ('time'))

            # Reciprocal lattice wave vectors
            kx_var = out.createVariable('kx', 'f4', ('nx'))
            ky_var = out.createVariable('ky', 'f4', ('ny'))
            sf_var = out.createVariable('sf', 'f4', ('time', 'nx', 'ny'))
            sf_im = out.createVariable('sf_im', 'f4', ('time', 'nx', 'ny'))
            sf_solid_var = out.createVariable('sf_solid', 'f4', ('time', 'nx', 'ny'))
            rho_k_var = out.createVariable('rho_k', 'f4', ('time', 'nx', 'ny'))

            # Fill the arrays with data
            time_var[:] = time_array
            gap_height_var[:] = gap_height_global
            gap_height_conv_var[:] = gap_height_conv_global
            gap_height_div_var[:] = gap_height_div_global
            com_var[:] = com_global

            kx_var[:] = kx
            ky_var[:] = ky

            rho_k_var[:] = rho_k_global.real
            sf_var[:] = sf_global.real
            sf_im[:] = sf_global.imag
            sf_solid_var[:] = sf_solid_global.real

            out.close()
            print('Dataset is closed!')

            t1 = timer.time()
            print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        slice1 = slice2
        slice2 += slice_size
