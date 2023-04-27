#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, warnings
import numpy as np
import scipy.constants as sci
import time as timer
from mpi4py import MPI
import netCDF4
import processor_bulk as pnc
from operator import itemgetter
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.seterr(all='raise')

# Get the processor version
version = 'bulk'

def make_grid(infile, Nx, Ny, Nz, slice_size, A_per_molecule, tessellate):
    """
    Parameters:
    -----------
    data: str, NetCDF trajectory file
    Nx, Ny, Nz: int, Number of chunks in the x-, y- and z-direcions
    slice_size: int, Number of frames to sample in the sliced trajectory
    A_per_molecule: int, no. of atoms per molecule
    tessellate: int, boolean to perform Delaunay tessellation
    """

    infile = comm.bcast(infile, root=0)
    data = netCDF4.Dataset(infile)
    Nx, Ny, Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)
    slice_size = comm.bcast(slice_size, root=0)
    A_per_molecule = comm.bcast(A_per_molecule, root=0)
    tessellate = comm.bcast(tessellate, root=0)

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
        print('Total simulation time: {} {}'.format(np.int(total_sim_time), Time.units))
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
        init = pnc.TrajtoGrid(data, start, end, Nx, Ny, Nz, A_per_molecule, tessellate)

        # Get the data
        cell_lengths, Nf, Nm, totVi, del_totVi, fluid_vol, je_x, je_y, je_z,  \
        fluid_vx_avg, fluid_vy_avg, fluid_vz_avg, vx_ch, den_ch, temp_ch, \
        Wxx_ch, Wyy_ch, Wzz_ch, Wxy_ch, Wxz_ch, Wyz_ch, vir_ch = itemgetter('cell_lengths',
                                                                  'Nf',
                                                                  'Nm',
                                                                  'totVi',
                                                                  'del_totVi',
                                                                  'fluid_vol',
                                                                  'je_x',
                                                                  'je_y',
                                                                  'je_z',
                                                                  'fluid_vx_avg',
                                                                  'fluid_vy_avg',
                                                                  'fluid_vz_avg',
                                                                  'vx_ch',
                                                                  'den_ch',
                                                                  'temp_ch',
                                                                  'Wxx_ch',
                                                                  'Wyy_ch',
                                                                  'Wzz_ch',
                                                                  'Wxy_ch',
                                                                  'Wxz_ch',
                                                                  'Wyz_ch',
                                                                  'vir_ch')(init.get_chunks())

        # Number of elements in the send buffer
        sendcounts_time = np.array(comm.gather(chunksize, root=0))
        sendcounts_chunk_fluid = np.array(comm.gather(vx_ch.size, root=0))

        fluid_vx_avg = np.array(comm.gather(fluid_vx_avg, root=0))
        fluid_vy_avg = np.array(comm.gather(fluid_vy_avg, root=0))
        fluid_vz_avg = np.array(comm.gather(fluid_vz_avg, root=0))

        if rank == 0:
            print('Sampled time: {} {}'.format(np.int(time_array[-1]), Time.units))
            # Dimensions: (Nf,) ------------------------------------------------
            # Average velocity of each atom over all the tsteps
            vx_global_avg = np.mean(fluid_vx_avg, axis=0)
            vy_global_avg = np.mean(fluid_vy_avg, axis=0)
            vz_global_avg = np.mean(fluid_vz_avg, axis=0)

            # ------------------------------------------------------------------
            # Initialize arrays ------------------------------------------------
            # ------------------------------------------------------------------

            # Dimensions: (time,) ----------------------------------------------
            # Heat flux
            je_x_global = np.zeros(time, dtype=np.float32)
            je_y_global = np.zeros_like(je_x_global)
            je_z_global = np.zeros_like(je_x_global)
            # Fluid vol
            totVi_global = np.zeros_like(je_x_global)
            del_totVi_global = np.zeros_like(je_x_global)
            fluid_vol_global = np.zeros_like(je_x_global)

            # Dimensions: (time, Nx, Nz) ---------------------------------------
            # Velocity
            vx_ch_global = np.zeros([time, Nx, Nz], dtype=np.float32)
            # Density
            den_ch_global = np.zeros_like(vx_ch_global)
            # Temperature
            temp_global = np.zeros_like(vx_ch_global)
            # Virial pressure tensor
            Wxx_ch_global = np.zeros_like(vx_ch_global)
            Wyy_ch_global = np.zeros_like(vx_ch_global)
            Wzz_ch_global = np.zeros_like(vx_ch_global)
            Wxy_ch_global = np.zeros_like(vx_ch_global)
            Wxz_ch_global = np.zeros_like(vx_ch_global)
            Wyz_ch_global = np.zeros_like(vx_ch_global)
            vir_ch_global = np.zeros_like(vx_ch_global)


        else:
            totVi_global = None
            del_totVi_global = None
            fluid_vol_global = None
            je_x_global = None
            je_y_global = None
            je_z_global = None

            vx_ch_global = None
            den_ch_global = None
            temp_global = None
            Wxx_ch_global = None
            Wyy_ch_global = None
            Wzz_ch_global = None
            Wxy_ch_global = None
            Wxz_ch_global = None
            Wyz_ch_global = None
            vir_ch_global = None

        # ----------------------------------------------------------------------
        # Collective Communication ---------------------------------------------
        # ----------------------------------------------------------------------

        # Arrays of Shape:(time, )-----------------------------------------------
        # Volumes
        comm.Gatherv(sendbuf=fluid_vol, recvbuf=(fluid_vol_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=totVi, recvbuf=(totVi_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=del_totVi, recvbuf=(del_totVi_global, sendcounts_time), root=0)
        # Heat flux
        comm.Gatherv(sendbuf=je_x, recvbuf=(je_x_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=je_y, recvbuf=(je_y_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=je_z, recvbuf=(je_z_global, sendcounts_time), root=0)

        # Arrays of Shape:(time, Nx, Nz)-----------------------------------------------
        # Velocity
        comm.Gatherv(sendbuf=vx_ch, recvbuf=(vx_ch_global, sendcounts_chunk_fluid), root=0)
        # Density
        comm.Gatherv(sendbuf=den_ch, recvbuf=(den_ch_global, sendcounts_chunk_fluid), root=0)
        # Temperature
        comm.Gatherv(sendbuf=temp_ch, recvbuf=(temp_global, sendcounts_chunk_fluid), root=0)
        # Pressure
        comm.Gatherv(sendbuf=Wxx_ch, recvbuf=(Wxx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wyy_ch, recvbuf=(Wyy_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wzz_ch, recvbuf=(Wzz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wxy_ch, recvbuf=(Wxy_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wxz_ch, recvbuf=(Wxz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wyz_ch, recvbuf=(Wyz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vir_ch, recvbuf=(vir_ch_global, sendcounts_chunk_fluid), root=0)

        # ----------------------------------------------------------------------
        # Write to netCDF file -------------------------------------------------
        # ----------------------------------------------------------------------

        if rank == 0:
            outfile = f"{infile.split('.')[0]}_{Nx}x{Nz}_{slice:0>3}.nc"
            out = netCDF4.Dataset(outfile, 'w', format='NETCDF3_64BIT_OFFSET')

            out.createDimension('x', Nx)
            out.createDimension('y', Ny)
            out.createDimension('z', Nz)
            out.createDimension('time', time)
            out.createDimension('Nf', Nf)
            out.createDimension('Nm', Nm)
            out.setncattr("Version", version)

            # Real lattice
            out.setncattr("Lx", cell_lengths[0])
            out.setncattr("Ly", cell_lengths[1])
            out.setncattr("Lz", cell_lengths[2])

            vx_all_var = out.createVariable('Fluid_Vx', 'f4', ('Nf'))
            vy_all_var = out.createVariable('Fluid_Vy', 'f4', ('Nf'))
            vz_all_var = out.createVariable('Fluid_Vz', 'f4', ('Nf'))

            time_var = out.createVariable('Time', 'f4', ('time'))
            voronoi_volumes = out.createVariable('Voronoi_volumes', 'f4', ('time'))
            voronoi_volumes_del = out.createVariable('Voronoi_volumes_del', 'f4', ('time'))
            fluid_vol_var = out.createVariable('Fluid_Vol', 'f4', ('time'))
            je_x_var =  out.createVariable('JeX', 'f4', ('time'))
            je_y_var =  out.createVariable('JeY', 'f4', ('time'))
            je_z_var =  out.createVariable('JeZ', 'f4', ('time'))

            vx_var =  out.createVariable('Vx', 'f4', ('time', 'x', 'z'))
            den_var = out.createVariable('Density', 'f4',  ('time', 'x', 'z'))
            temp_var =  out.createVariable('Temperature', 'f4', ('time', 'x', 'z'))
            vir_var = out.createVariable('Virial', 'f4', ('time', 'x', 'z'))
            Wxx_var = out.createVariable('Wxx', 'f4', ('time', 'x', 'z'))
            Wyy_var = out.createVariable('Wyy', 'f4', ('time', 'x', 'z'))
            Wzz_var = out.createVariable('Wzz', 'f4', ('time', 'x', 'z'))
            Wxy_var = out.createVariable('Wxy', 'f4', ('time', 'x', 'z'))
            Wxz_var = out.createVariable('Wxz', 'f4', ('time', 'x', 'z'))
            Wyz_var = out.createVariable('Wyz', 'f4', ('time', 'x', 'z'))

            # Fill the arrays with data
            vx_all_var[:] = vx_global_avg
            vy_all_var[:] = vy_global_avg
            vz_all_var[:] = vz_global_avg

            time_var[:] = time_array
            voronoi_volumes[:] = totVi_global
            voronoi_volumes_del[:] = del_totVi_global
            fluid_vol_var[:] = fluid_vol_global
            je_x_var[:] = je_x_global
            je_y_var[:] = je_y_global
            je_z_var[:] = je_z_global

            vx_var[:] = vx_ch_global
            den_var[:] = den_ch_global
            temp_var[:] = temp_global
            Wxx_var[:] = Wxx_ch_global
            Wyy_var[:] = Wyy_ch_global
            Wzz_var[:] = Wzz_ch_global
            Wxy_var[:] = Wxy_ch_global
            Wxz_var[:] = Wxz_ch_global
            Wyz_var[:] = Wyz_ch_global
            vir_var[:] = vir_ch_global

            out.close()
            print('Dataset is closed!')

            t1 = timer.time()
            print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        slice1 = slice2
        slice2 += slice_size
