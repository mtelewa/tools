#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, warnings
import numpy as np
import scipy.constants as sci
import time as timer
from mpi4py import MPI
import netCDF4
import processor_nc_211018 as pnc
from operator import itemgetter
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the processor version
for i in sys.modules.keys():
    if i.startswith('processor_nc'):
        version = re.split('(\d+)', i)[1]

def make_grid(infile, Nx, Nz, slice_size, mf, A_per_molecule, stable_start, stable_end, pump_start, pump_end, Ny=1, nx=1, ny=1, nz=5):

    infile = comm.bcast(infile, root=0)
    data = netCDF4.Dataset(infile)
    Nx, Nz = comm.bcast(Nx, root=0), comm.bcast(Nz, root=0)
    slice_size = comm.bcast(slice_size, root=0)

    # Timesteps
    Time = data.variables["time"]
    time_arr=np.array(Time).astype(np.float32)

    tSteps_tot = Time.shape[0]-1
    out_frequency = np.int(time_arr[-1] - time_arr[-2])

    # If the dataset is more than 1M tsteps, slice it to fit in memory
    if tSteps_tot <= 1000:
        Nslices = 1
    else:
        Nslices = tSteps_tot // slice_size

    Nslices = np.int(Nslices)
    slice1, slice2 = 1, slice_size+1

    if rank == 0:
        print('Total simualtion time: {} {}'.format(np.int(tSteps_tot * out_frequency), Time.units))
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
        init = pnc.traj_to_grid(data, start, end, Nx, Nz, mf, A_per_molecule)

        # Get the data
        cell_lengths, kx, ky, kz, \
        gap_heights, bulkStartZ_time, bulkEndZ_time, com, fluxes, totVi,\
        fluid_vx_avg, fluid_vy_avg, \
        vx_ch, uCOMx, den_ch, sf, sf_x, sf_y, \
        jx_ch, vir_ch, Wxy_ch, Wxz_ch, Wyz_ch,\
        temp_ch,\
        surfU_fx_ch, surfU_fy_ch, surfU_fz_ch,\
        surfL_fx_ch, surfL_fy_ch, surfL_fz_ch,\
        den_bulk_ch, Nf, Nm, \
        fluid_vol = itemgetter('cell_lengths',
                                  'kx', 'ky', 'kz',
                                  'gap_heights',
                                  'bulkStartZ_time',
                                  'bulkEndZ_time',
                                  'com',
                                  'fluxes',
                                  'totVi',
                                  'fluid_vx_avg',
                                  'fluid_vy_avg',
                                  'vx_ch',
                                  'uCOMx',
                                  'den_ch',
                                  'sf',
                                  'sf_x',
                                  'sf_y',
                                  'jx_ch',
                                  'vir_ch',
                                  'Wxy_ch',
                                  'Wxz_ch',
                                  'Wyz_ch',
                                  'temp_ch',
                                  'surfU_fx_ch', 'surfU_fy_ch', 'surfU_fz_ch',
                                  'surfL_fx_ch', 'surfL_fy_ch', 'surfL_fz_ch',
                                  'den_bulk_ch',
                                  'Nf', 'Nm',
                                  'fluid_vol')(init.get_chunks(stable_start,
                                    stable_end, pump_start, pump_end, nx, ny, nz))


        # Number of elements in the send buffer
        sendcounts_time = np.array(comm.gather(chunksize, root=0))
        sendcounts_chunk_fluid = np.array(comm.gather(vx_ch.size, root=0))
        sendcounts_chunk_fluid_layer_3d = np.array(comm.gather(sf.size, root=0))
        sendcounts_chunk_fluid_layer_nx = np.array(comm.gather(sf_x.size, root=0))
        sendcounts_chunk_fluid_layer_ny = np.array(comm.gather(sf_y.size, root=0))
        sendcounts_chunk_solid = np.array(comm.gather(surfU_fx_ch.size, root=0))
        sendcounts_chunk_bulk = np.array(comm.gather(den_bulk_ch.size, root=0))

        fluid_vx_avg = np.array(comm.gather(fluid_vx_avg, root=0))
        fluid_vy_avg = np.array(comm.gather(fluid_vy_avg, root=0))

        if rank == 0:

            print('Sampled time: {} {}'.format(np.int(time_array[-1]), Time.units))
            # Dimensions: (Nf)
            # Average velocity of each atom over all the tsteps in the slice
            vx_global_avg = np.mean(fluid_vx_avg, axis=0)
            vy_global_avg = np.mean(fluid_vy_avg, axis=0)

            # Dimensions: (time)
            # Gap Heights
            gap_height_global = np.zeros(time, dtype=np.float32)
            gap_height_conv_global = np.zeros_like(gap_height_global)
            gap_height_div_global = np.zeros_like(gap_height_global)
            bulkStartZ_time_global = np.zeros_like(gap_height_global)
            bulkEndZ_time_global = np.zeros_like(gap_height_global)
            # Center of Mass
            com_global = np.zeros_like(gap_height_global)
            # Flow Rate and Mass flux
            mflux_pump_global = np.zeros_like(gap_height_global)
            mflowrate_pump_global = np.zeros_like(gap_height_global)
            mflux_stable_global = np.zeros_like(gap_height_global)
            mflowrate_stable_global = np.zeros_like(gap_height_global)
            # Voronoi vol
            totVi_global = np.zeros_like(gap_height_global)

            fluid_vol_global = np.zeros_like(gap_height_global)

            # Dimensions: (time, Nx, Nz)
            # Velocity (chunked in stable region)
            vx_ch_global = np.zeros([time, Nx, Nz], dtype=np.float32)
            uCOMx_global = np.zeros_like(vx_ch_global)
            # Density
            den_ch_global = np.zeros_like(vx_ch_global)
            # Density Fourier coefficients
            # rho_kx_ch_global = np.zeros([time, nmax] , dtype=np.complex64)
            sf_global = np.zeros([time, nx, ny] , dtype=np.complex64)
            sf_x_global = np.zeros([time, nx] , dtype=np.complex64)
            sf_y_global = np.zeros([time, ny] , dtype=np.complex64)

            # rho_kx_im_ch_global = np.zeros([time, nmax] , dtype=np.float32)
            # rho_ky_ch_global = np.zeros_like(rho_kx_ch_global)
            # Mass flux
            jx_ch_global = np.zeros_like(vx_ch_global)
            # Virial
            vir_ch_global = np.zeros_like(vx_ch_global)
            Wxy_ch_global = np.zeros_like(vx_ch_global)
            Wxz_ch_global = np.zeros_like(vx_ch_global)
            Wyz_ch_global = np.zeros_like(vx_ch_global)
            # Temperature
            temp_global = np.zeros_like(vx_ch_global)

            # Dimensions: (time, Nx)
            # Surface Forces
            surfU_fx_ch_global = np.zeros([time, Nx], dtype=np.float32)
            surfU_fy_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfU_fz_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfL_fx_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfL_fy_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfL_fz_ch_global = np.zeros_like(surfU_fx_ch_global)
            den_bulk_ch_global = np.zeros_like(surfU_fx_ch_global)

        else:
            gap_height_global = None
            gap_height_conv_global = None
            gap_height_div_global = None
            bulkStartZ_time_global = None
            bulkEndZ_time_global = None
            com_global = None
            mflowrate_stable_global = None
            mflux_stable_global = None
            mflux_pump_global = None
            mflowrate_pump_global = None
            totVi_global = None
            fluid_vol_global = None

            vx_ch_global = None
            uCOMx_global = None
            den_ch_global = None
            # rho_kx_ch_global = None
            sf_global = None
            sf_x_global = None
            sf_y_global = None
            # sf_ch_y_global = None
            # rho_kx_im_ch_global = None
            # rho_ky_ch_global = None
            jx_ch_global = None
            vir_ch_global = None
            Wxy_ch_global = None
            Wxz_ch_global = None
            Wyz_ch_global = None
            temp_global = None

            surfU_fx_ch_global = None
            surfU_fy_ch_global = None
            surfU_fz_ch_global = None
            surfL_fx_ch_global = None
            surfL_fy_ch_global = None
            surfL_fz_ch_global = None
            den_bulk_ch_global = None

        if gap_heights != None:     # If there are walls
            comm.Gatherv(sendbuf=gap_heights[0], recvbuf=(gap_height_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=gap_heights[1], recvbuf=(gap_height_conv_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=gap_heights[2], recvbuf=(gap_height_div_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=bulkStartZ_time, recvbuf=(bulkStartZ_time_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=bulkEndZ_time, recvbuf=(bulkEndZ_time_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=com, recvbuf=(com_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=fluxes[0], recvbuf=(mflux_pump_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=fluxes[1], recvbuf=(mflowrate_pump_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=fluxes[2], recvbuf=(mflux_stable_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=fluxes[3], recvbuf=(mflowrate_stable_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=surfU_fx_ch, recvbuf=(surfU_fx_ch_global, sendcounts_chunk_solid), root=0)
            comm.Gatherv(sendbuf=surfU_fy_ch, recvbuf=(surfU_fy_ch_global, sendcounts_chunk_solid), root=0)
            comm.Gatherv(sendbuf=surfU_fz_ch, recvbuf=(surfU_fz_ch_global, sendcounts_chunk_solid), root=0)
            comm.Gatherv(sendbuf=surfL_fx_ch, recvbuf=(surfL_fx_ch_global, sendcounts_chunk_solid), root=0)
            comm.Gatherv(sendbuf=surfL_fy_ch, recvbuf=(surfL_fy_ch_global, sendcounts_chunk_solid), root=0)
            comm.Gatherv(sendbuf=surfL_fz_ch, recvbuf=(surfL_fz_ch_global, sendcounts_chunk_solid), root=0)
            # comm.Gatherv(sendbuf=rho_kx_ch, recvbuf=(rho_kx_ch_global, sendcounts_chunk_fluid_layer), root=0)
            comm.Gatherv(sendbuf=sf, recvbuf=(sf_global, sendcounts_chunk_fluid_layer_3d), root=0)
            comm.Gatherv(sendbuf=sf_x, recvbuf=(sf_x_global, sendcounts_chunk_fluid_layer_nx), root=0)
            comm.Gatherv(sendbuf=sf_y, recvbuf=(sf_y_global, sendcounts_chunk_fluid_layer_ny), root=0)

            # If the virial was not computed, skip
            try:
                comm.Gatherv(sendbuf=totVi, recvbuf=(totVi_global, sendcounts_time), root=0)
            except TypeError:
                pass
        else:   # If only bulk
            comm.Gatherv(sendbuf=fluid_vol, recvbuf=(fluid_vol_global, sendcounts_time), root=0)
            try:
                comm.Gatherv(sendbuf=totVi, recvbuf=(totVi_global, sendcounts_time), root=0)
            except TypeError:
                pass

        # For both walls and bulk simulations
        comm.Gatherv(sendbuf=vx_ch, recvbuf=(vx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=uCOMx, recvbuf=(uCOMx_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=den_ch, recvbuf=(den_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=jx_ch, recvbuf=(jx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=temp_ch, recvbuf=(temp_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vir_ch, recvbuf=(vir_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wxy_ch, recvbuf=(Wxy_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wxz_ch, recvbuf=(Wxz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wyz_ch, recvbuf=(Wyz_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=den_bulk_ch, recvbuf=(den_bulk_ch_global, sendcounts_chunk_bulk), root=0)

        if rank == 0:

            # Write to netCDF file  ------------------------------------------
            outfile = f"{infile.split('.')[0]}_{Nx}x{Nz}_{slice:0>3}.nc"
            out = netCDF4.Dataset(outfile, 'w', format='NETCDF3_64BIT_OFFSET')

            out.createDimension('x', Nx)
            out.createDimension('y', Ny)
            out.createDimension('z', Nz)
            out.createDimension('nx', nx)
            out.createDimension('ny', ny)
            out.createDimension('time', time)
            out.createDimension('Nf', Nf)
            out.createDimension('Nm', Nm)

            # Real lattice
            out.setncattr("Lx", cell_lengths[0])
            out.setncattr("Ly", cell_lengths[1])
            out.setncattr("Lz", cell_lengths[2])
            out.setncattr("Version", version)

            vx_all_var = out.createVariable('Fluid_Vx', 'f4', ('Nf'))
            vy_all_var = out.createVariable('Fluid_Vy', 'f4', ('Nf'))

            time_var = out.createVariable('Time', 'f4', ('time'))
            gap_height_var =  out.createVariable('Height', 'f4', ('time'))
            gap_height_conv_var =  out.createVariable('Height_conv', 'f4', ('time'))
            gap_height_div_var =  out.createVariable('Height_div', 'f4', ('time'))
            bulkStartZ_time_var = out.createVariable('Bulk_Start', 'f4', ('time'))
            bulkEndZ_time_var = out.createVariable('Bulk_End', 'f4', ('time'))
            com_var =  out.createVariable('COM', 'f4', ('time'))
            mflux_pump = out.createVariable('mflux_pump', 'f4', ('time'))
            mflowrate_pump = out.createVariable('mflow_rate_pump', 'f4', ('time'))
            mflux_stable = out.createVariable('mflux_stable', 'f4', ('time'))
            mflowrate_stable = out.createVariable('mflow_rate_stable', 'f4', ('time'))
            voronoi_volumes = out.createVariable('Voronoi_volumes', 'f4', ('time'))
            fluid_vol_var = out.createVariable('Fluid_Vol', 'f4', ('time'))

            vx_var =  out.createVariable('Vx', 'f4', ('time', 'x', 'z'))
            den_var = out.createVariable('Density', 'f4',  ('time', 'x', 'z'))

            # Reciprocal lattice wave vectors
            kx_var = out.createVariable('kx', 'f4', ('nx'))
            ky_var = out.createVariable('ky', 'f4', ('ny'))
            sf_var = out.createVariable('sf', 'f4', ('time', 'nx', 'ny'))
            sf_x_var = out.createVariable('sf_x', 'f4', ('time', 'nx'))
            sf_y_var = out.createVariable('sf_y', 'f4', ('time', 'ny'))
            sf_im = out.createVariable('sf_im', 'f4', ('time', 'nx', 'ny'))

            jx_var =  out.createVariable('Jx', 'f4', ('time', 'x', 'z'))

            vir_var = out.createVariable('Virial', 'f4', ('time', 'x', 'z'))
            Wxy_var = out.createVariable('Wxy', 'f4', ('time', 'x', 'z'))
            Wxz_var = out.createVariable('Wxz', 'f4', ('time', 'x', 'z'))
            Wyz_var = out.createVariable('Wyz', 'f4', ('time', 'x', 'z'))
            temp_var =  out.createVariable('Temperature', 'f4', ('time', 'x', 'z'))

            fx_U_var =  out.createVariable('Fx_Upper', 'f4',  ('time', 'x'))
            fy_U_var =  out.createVariable('Fy_Upper', 'f4',  ('time', 'x'))
            fz_U_var =  out.createVariable('Fz_Upper', 'f4',  ('time', 'x'))
            fx_L_var =  out.createVariable('Fx_Lower', 'f4',  ('time', 'x'))
            fy_L_var =  out.createVariable('Fy_Lower', 'f4',  ('time', 'x'))
            fz_L_var =  out.createVariable('Fz_Lower', 'f4',  ('time', 'x'))
            density_bulk_var =  out.createVariable('Density_Bulk', 'f4',  ('time', 'x'))

            # Fill the arrays with data
            time_var[:] = time_array
            gap_height_var[:] = gap_height_global
            gap_height_conv_var[:] = gap_height_conv_global
            gap_height_div_var[:] = gap_height_div_global
            bulkStartZ_time_var[:] = bulkStartZ_time_global
            bulkEndZ_time_var[:] = bulkEndZ_time_global
            com_var[:] = com_global
            mflux_pump[:] = mflux_pump_global
            mflowrate_pump[:] = mflowrate_pump_global
            mflux_stable[:] = mflux_stable_global
            mflowrate_stable[:] = mflowrate_stable_global
            voronoi_volumes[:] = totVi_global
            fluid_vol_var[:] = fluid_vol_global

            vx_all_var[:] = vx_global_avg
            vy_all_var[:] = vy_global_avg

            vx_var[:] = vx_ch_global
            den_var[:] = den_ch_global

            kx_var[:] = kx
            ky_var[:] = ky
            sf_var[:] = sf_global.real
            sf_x_var[:] = sf_x_global.real
            sf_y_var[:] = sf_y_global.real
            sf_im[:] = sf_global.imag

            jx_var[:] = jx_ch_global
            vir_var[:] = vir_ch_global
            Wxy_var[:] = Wxy_ch_global
            Wxz_var[:] = Wxz_ch_global
            Wyz_var[:] = Wyz_ch_global
            temp_var[:] = temp_global

            fx_U_var[:] = surfU_fx_ch_global
            fy_U_var[:] = surfU_fy_ch_global
            fz_U_var[:] = surfU_fz_ch_global
            fx_L_var[:] = surfL_fx_ch_global
            fy_L_var[:] = surfL_fy_ch_global
            fz_L_var[:] = surfL_fz_ch_global
            density_bulk_var[:] = den_bulk_ch_global

            out.close()
            print('Dataset is closed!')

            t1 = timer.time()
            print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        slice1 = slice2
        slice2 += slice_size
