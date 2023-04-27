#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, warnings
import numpy as np
import scipy.constants as sci
import time as timer
from mpi4py import MPI
import netCDF4
import processor_4d as pnc
from operator import itemgetter
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.seterr(all='raise')

# # Get the processor version
# for i in sys.modules.keys():
#     if i.startswith('processor_nc'):
#         version = re.split('(\d+)', i)[1]

version = '4D'

def make_grid(infile, Nx, Ny, Nz, slice_size, mf, A_per_molecule, fluid, stable_start,
                        stable_end, pump_start, pump_end, tessellate, TW_interface):
    """
    Parameters:
    -----------
    data: str, NetCDF trajectory file
    Nx, Ny, Nz: int, Number of chunks in the x-, y- and z-direcions
    slice_size: int, Number of frames to sample in the sliced trajectory
    mf: float, molecular mass of fluid (g/mol)
    A_per_molecule: int, no. of atoms per molecule
    fluid: str, fluid material name
    stable_start, stable_end, pump_start, pump_end: floats,
            regions boundaries in the streamwise (x-direction)
    tessellate: int, boolean to perform Delaunay tessellation
    TW_interface: int, Define the location of vibrating atoms in the upper wall
            Default = 1: thermostat is applied on the wall layers in contact with the fluid
            if TW_interface = 0: thermostat is applied on the wall layers 1/3 wall
            thickness (in z-direction) away
    """

    infile = comm.bcast(infile, root=0)
    data = netCDF4.Dataset(infile)
    Nx, Ny, Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)
    slice_size = comm.bcast(slice_size, root=0)
    mf, A_per_molecule = comm.bcast(mf, root=0), comm.bcast(A_per_molecule, root=0)
    fluid = comm.bcast(fluid, root=0)
    stable_start, stable_end = comm.bcast(stable_start, root=0), comm.bcast(stable_end, root=0)
    pump_start, pump_end = comm.bcast(pump_start, root=0), comm.bcast(pump_end, root=0)
    tessellate = comm.bcast(tessellate, root=0)
    TW_interface = comm.bcast(TW_interface, root=0)

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
        init = pnc.TrajtoGrid(data, start, end, Nx, Ny, Nz, mf, A_per_molecule, fluid, tessellate, TW_interface)

        # Get the data
        cell_lengths, gap_heights, bulkStartZ_time, bulkEndZ_time, com, Nf, Nm, totVi, del_totVi, \
        vx_ch_whole, den_ch, jx_ch, mflowrate_ch, tempx_ch, tempy_ch, tempz_ch, temp_ch, \
        Wxy_ch, Wxz_ch, Wyz_ch, Wxx_ch, Wyy_ch, Wzz_ch, vir_ch, den_bulk_ch, vx_ch, \
        temp_ch_solid = itemgetter('cell_lengths',
                                  'gap_heights',
                                  'bulkStartZ_time',
                                  'bulkEndZ_time',
                                  'com',
                                  'Nf', 'Nm',
                                  'totVi',
                                  'del_totVi',
                                  'vx_ch_whole',
                                  'den_ch',
                                  'jx_ch',
                                  'mflowrate_ch',
                                  'tempx_ch',
                                  'tempy_ch',
                                  'tempz_ch',
                                  'temp_ch',
                                  'Wxy_ch',
                                  'Wxz_ch',
                                  'Wyz_ch',
                                  'Wxx_ch',
                                  'Wyy_ch',
                                  'Wzz_ch',
                                  'vir_ch',
                                  'den_bulk_ch',
                                  'vx_ch',
                                  'temp_ch_solid')(init.get_chunks(stable_start,
                                    stable_end, pump_start, pump_end))

        # Number of elements in the send buffer
        sendcounts_time = np.array(comm.gather(chunksize, root=0))
        sendcounts_chunk_fluid = np.array(comm.gather(vx_ch.size, root=0))
        sendcounts_chunk_vib = np.array(comm.gather(temp_ch_solid.size, root=0))
        sendcounts_chunk_bulk = np.array(comm.gather(den_bulk_ch.size, root=0))

        if rank == 0:

            print('Sampled time: {} {}'.format(np.int(time_array[-1]), Time.units))
            # ------------------------------------------------------------------
            # Initialize arrays ------------------------------------------------
            # ------------------------------------------------------------------

            # Dimensions: (time,)  ---------------------------------------------
            # Gap Heights
            gap_height_global = np.zeros(time, dtype=np.float32)
            gap_height_conv_global = np.zeros_like(gap_height_global)
            gap_height_div_global = np.zeros_like(gap_height_global)
            bulkStartZ_time_global = np.zeros_like(gap_height_global)
            bulkEndZ_time_global = np.zeros_like(gap_height_global)
            # Center of Mass
            com_global = np.zeros_like(gap_height_global)
            # Fluid vol
            totVi_global = np.zeros_like(gap_height_global)
            del_totVi_global = np.zeros_like(gap_height_global)

            # Dimensions: (time, Nx, Nz) ---------------------------------------
            # Velocity  -- fluid
            vx_ch_whole_global = np.zeros([time, Nx, Ny, Nz], dtype=np.float32)
            # Density  -- fluid
            den_ch_global = np.zeros_like(vx_ch_whole_global)
            # Density -- bulk
            den_bulk_ch_global = np.zeros_like(vx_ch_whole_global)
            # Mass flux and flow rate  -- fluid
            jx_ch_global = np.zeros_like(vx_ch_whole_global)
            mflowrate_ch_global = np.zeros_like(vx_ch_whole_global)
            # Temperature -- fluid
            tempx_global = np.zeros_like(vx_ch_whole_global)
            tempy_global = np.zeros_like(vx_ch_whole_global)
            tempz_global = np.zeros_like(vx_ch_whole_global)
            temp_global = np.zeros_like(vx_ch_whole_global)
            # Virial pressure tensor -- fluid
            Wxy_ch_global = np.zeros_like(vx_ch_whole_global)
            Wxz_ch_global = np.zeros_like(vx_ch_whole_global)
            Wyz_ch_global = np.zeros_like(vx_ch_whole_global)
            # Virial pressure tensor -- bulk
            vir_ch_global = np.zeros_like(vx_ch_whole_global)
            Wxx_ch_global = np.zeros_like(vx_ch_whole_global)
            Wyy_ch_global = np.zeros_like(vx_ch_whole_global)
            Wzz_ch_global = np.zeros_like(vx_ch_whole_global)
            # Velocity  -- stable
            vx_ch_global = np.zeros_like(vx_ch_whole_global)
            # Temperature -- solid
            temp_solid_global = np.zeros_like(vx_ch_whole_global)

        else:
            gap_height_global = None
            gap_height_conv_global = None
            gap_height_div_global = None
            bulkStartZ_time_global = None
            bulkEndZ_time_global = None
            com_global = None
            totVi_global = None
            del_totVi_global = None

            vx_ch_whole_global = None
            den_ch_global = None
            den_bulk_ch_global = None
            jx_ch_global = None
            mflowrate_ch_global = None
            tempx_global = None
            tempy_global = None
            tempz_global = None
            temp_global = None
            Wxy_ch_global = None
            Wxz_ch_global = None
            Wyz_ch_global = None
            vir_ch_global = None
            Wxx_ch_global = None
            Wyy_ch_global = None
            Wzz_ch_global = None
            vx_ch_global = None
            temp_solid_global = None

        # ----------------------------------------------------------------------
        # Collective Communication ---------------------------------------------
        # ----------------------------------------------------------------------

        comm.Gatherv(sendbuf=gap_heights[0], recvbuf=(gap_height_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=gap_heights[1], recvbuf=(gap_height_conv_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=gap_heights[2], recvbuf=(gap_height_div_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=bulkStartZ_time, recvbuf=(bulkStartZ_time_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=bulkEndZ_time, recvbuf=(bulkEndZ_time_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=com, recvbuf=(com_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=totVi, recvbuf=(totVi_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=del_totVi, recvbuf=(del_totVi_global, sendcounts_time), root=0)

        comm.Gatherv(sendbuf=np.ma.masked_invalid(vx_ch_whole), recvbuf=(vx_ch_whole_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(den_ch), recvbuf=(den_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(den_bulk_ch), recvbuf=(den_bulk_ch_global, sendcounts_chunk_bulk), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(jx_ch), recvbuf=(jx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(mflowrate_ch), recvbuf=(mflowrate_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(tempx_ch), recvbuf=(tempx_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(tempy_ch), recvbuf=(tempy_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(tempz_ch), recvbuf=(tempz_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(temp_ch), recvbuf=(temp_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(Wxy_ch), recvbuf=(Wxy_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(Wxz_ch), recvbuf=(Wxz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(Wyz_ch), recvbuf=(Wyz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(vir_ch), recvbuf=(vir_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(Wxx_ch), recvbuf=(Wxx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(Wyy_ch), recvbuf=(Wyy_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(Wzz_ch), recvbuf=(Wzz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(vx_ch), recvbuf=(vx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=np.ma.masked_invalid(temp_ch_solid), recvbuf=(temp_solid_global, sendcounts_chunk_vib), root=0)

        # ----------------------------------------------------------------------
        # Write to netCDF file -------------------------------------------------
        # ----------------------------------------------------------------------

        if rank == 0:

            # Write to netCDF file  ------------------------------------------
            outfile = f"{infile.split('.')[0]}_{Nx}x{Ny}x{Nz}_{slice:0>3}.nc"
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

            time_var = out.createVariable('Time', 'f4', ('time'))
            gap_height_var =  out.createVariable('Height', 'f4', ('time'))
            gap_height_conv_var =  out.createVariable('Height_conv', 'f4', ('time'))
            gap_height_div_var =  out.createVariable('Height_div', 'f4', ('time'))
            bulkStartZ_time_var = out.createVariable('Bulk_Start', 'f4', ('time'))
            bulkEndZ_time_var = out.createVariable('Bulk_End', 'f4', ('time'))
            com_var =  out.createVariable('COM', 'f4', ('time'))
            voronoi_volumes = out.createVariable('Voronoi_volumes', 'f4', ('time'))
            voronoi_volumes_del = out.createVariable('Voronoi_volumes_del', 'f4', ('time'))

            vx_var =  out.createVariable('Vx', 'f4', ('time', 'x', 'y', 'z'))
            vx_whole_var = out.createVariable('Vx_whole', 'f4', ('time', 'x', 'y', 'z'))
            den_var = out.createVariable('Density', 'f4',  ('time', 'x', 'y', 'z'))
            density_bulk_var =  out.createVariable('Density_Bulk', 'f4',  ('time', 'x', 'y', 'z'))
            jx_var =  out.createVariable('Jx', 'f4', ('time', 'x', 'y', 'z'))
            mflowrate_var = out.createVariable('mdot', 'f4', ('time', 'x', 'y', 'z'))
            vir_var = out.createVariable('Virial', 'f4', ('time', 'x', 'y', 'z'))
            Wxx_var = out.createVariable('Wxx', 'f4', ('time', 'x', 'y', 'z'))
            Wyy_var = out.createVariable('Wyy', 'f4', ('time', 'x', 'y', 'z'))
            Wzz_var = out.createVariable('Wzz', 'f4', ('time', 'x', 'y', 'z'))
            Wxy_var = out.createVariable('Wxy', 'f4', ('time', 'x', 'y', 'z'))
            Wxz_var = out.createVariable('Wxz', 'f4', ('time', 'x', 'y', 'z'))
            Wyz_var = out.createVariable('Wyz', 'f4', ('time', 'x', 'y', 'z'))
            temp_var =  out.createVariable('Temperature', 'f4', ('time', 'x', 'y', 'z'))
            tempx_var =  out.createVariable('TemperatureX', 'f4', ('time', 'x', 'y', 'z'))
            tempy_var =  out.createVariable('TemperatureY', 'f4', ('time', 'x', 'y', 'z'))
            tempz_var =  out.createVariable('TemperatureZ', 'f4', ('time', 'x', 'y', 'z'))
            temp_solid_var = out.createVariable('Temperature_solid', 'f4', ('time', 'x', 'y', 'z'))

            fx_U_var =  out.createVariable('Fx_Upper', 'f4',  ('time', 'x'))
            fy_U_var =  out.createVariable('Fy_Upper', 'f4',  ('time', 'x'))
            fz_U_var =  out.createVariable('Fz_Upper', 'f4',  ('time', 'x'))
            fx_L_var =  out.createVariable('Fx_Lower', 'f4',  ('time', 'x'))
            fy_L_var =  out.createVariable('Fy_Lower', 'f4',  ('time', 'x'))
            fz_L_var =  out.createVariable('Fz_Lower', 'f4',  ('time', 'x'))

            # Fill the arrays with data
            time_var[:] = time_array
            gap_height_var[:] = gap_height_global
            gap_height_conv_var[:] = gap_height_conv_global
            gap_height_div_var[:] = gap_height_div_global
            bulkStartZ_time_var[:] = bulkStartZ_time_global
            bulkEndZ_time_var[:] = bulkEndZ_time_global
            com_var[:] = com_global
            voronoi_volumes[:] = totVi_global
            voronoi_volumes_del[:] = del_totVi_global

            vx_var[:] = vx_ch_global
            den_var[:] = den_ch_global
            density_bulk_var[:] = den_bulk_ch_global
            jx_var[:] = jx_ch_global
            mflowrate_var[:] = mflowrate_ch_global
            tempx_var[:] = tempx_global
            tempy_var[:] = tempy_global
            tempz_var[:] = tempz_global
            temp_var[:] = temp_global
            Wxy_var[:] = Wxy_ch_global
            Wxz_var[:] = Wxz_ch_global
            Wyz_var[:] = Wyz_ch_global
            Wxx_var[:] = Wxx_ch_global
            Wyy_var[:] = Wyy_ch_global
            Wzz_var[:] = Wzz_ch_global
            vir_var[:] = vir_ch_global
            vx_whole_var[:] = vx_ch_whole_global
            temp_solid_var[:] = temp_solid_global

            out.close()
            print('Dataset is closed!')

            t1 = timer.time()
            print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        slice1 = slice2
        slice2 += slice_size
