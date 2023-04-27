#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, warnings
import numpy as np
import scipy.constants as sci
import time as timer
from mpi4py import MPI
import netCDF4
import processor_nc_5_regions as pnc
from operator import itemgetter
import re

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.seterr(all='raise')

# Get the processor version
for i in sys.modules.keys():
    if i.startswith('processor_nc'):
        version = re.split('(\d+)', i)[1]

def make_grid(infile, Nx, Nz, slice_size, mf, A_per_molecule, fluid, stable_start, stable_end, pump_start, pump_end, Ny=1):

    infile = comm.bcast(infile, root=0)
    data = netCDF4.Dataset(infile)
    Nx, Nz = comm.bcast(Nx, root=0), comm.bcast(Nz, root=0)
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
        init = pnc.TrajtoGrid(data, start, end, Nx, Nz, mf, A_per_molecule, fluid)

        # Get the data
        cell_lengths, gap_heights, bulkStartZ_time, bulkEndZ_time, com, fluxes, totVi, mytotVi,\
        fluid_vx_avg, fluid_vy_avg, fluid_vz_avg, \
        fluid_vx_avg_lte, fluid_vy_avg_lte, fluid_vz_avg_lte, \
        vx_ch, vx_ch_whole, vx_R1, vx_R2, vx_R3, vx_R4, vx_R5, \
        uCOMx, den_ch, \
        jx_ch, mflowrate_ch, je_x, je_y, je_z, vir_ch, Wxx_ch, Wyy_ch, Wzz_ch, Wxy_ch, Wxz_ch, Wyz_ch,\
        temp_ch, tempx_ch, tempy_ch, tempz_ch, temp_ch_solid,\
        surfU_fx_ch, surfU_fy_ch, surfU_fz_ch,\
        surfL_fx_ch, surfL_fy_ch, surfL_fz_ch,\
        den_bulk_ch, Nf, Nm, \
        fluid_vol = itemgetter('cell_lengths',
                                  'gap_heights',
                                  'bulkStartZ_time',
                                  'bulkEndZ_time',
                                  'com',
                                  'fluxes',
                                  'totVi',
                                  'mytotVi',
                                  'fluid_vx_avg',
                                  'fluid_vy_avg',
                                  'fluid_vz_avg',
                                  'fluid_vx_avg_lte',
                                  'fluid_vy_avg_lte',
                                  'fluid_vz_avg_lte',
                                  'vx_ch',
                                  'vx_ch_whole',
                                  'vx_R1',
                                  'vx_R2',
                                  'vx_R3',
                                  'vx_R4',
                                  'vx_R5',
                                  'uCOMx',
                                  'den_ch',
                                  'jx_ch',
                                  'mflowrate_ch',
                                  'je_x',
                                  'je_y',
                                  'je_z',
                                  'vir_ch',
                                  'Wxx_ch',
                                  'Wyy_ch',
                                  'Wzz_ch',
                                  'Wxy_ch',
                                  'Wxz_ch',
                                  'Wyz_ch',
                                  'temp_ch',
                                  'tempx_ch',
                                  'tempy_ch',
                                  'tempz_ch',
                                  'temp_ch_solid',
                                  'surfU_fx_ch', 'surfU_fy_ch', 'surfU_fz_ch',
                                  'surfL_fx_ch', 'surfL_fy_ch', 'surfL_fz_ch',
                                  'den_bulk_ch',
                                  'Nf', 'Nm',
                                  'fluid_vol')(init.get_chunks(stable_start,
                                    stable_end, pump_start, pump_end))

        # Number of elements in the send buffer
        sendcounts_time = np.array(comm.gather(chunksize, root=0))
        sendcounts_chunk_fluid = np.array(comm.gather(vx_ch.size, root=0))
        sendcounts_chunk_solid = np.array(comm.gather(surfU_fx_ch.size, root=0))
        sendcounts_chunk_vib = np.array(comm.gather(temp_ch_solid.size, root=0))
        sendcounts_chunk_bulk = np.array(comm.gather(den_bulk_ch.size, root=0))

        fluid_vx_avg = np.array(comm.gather(fluid_vx_avg, root=0))
        fluid_vy_avg = np.array(comm.gather(fluid_vy_avg, root=0))
        fluid_vz_avg = np.array(comm.gather(fluid_vz_avg, root=0))
        fluid_vx_avg_lte = np.array(comm.gather(fluid_vx_avg_lte, root=0))
        fluid_vy_avg_lte = np.array(comm.gather(fluid_vy_avg_lte, root=0))
        fluid_vz_avg_lte = np.array(comm.gather(fluid_vz_avg_lte, root=0))

        if rank == 0:

            print('Sampled time: {} {}'.format(np.int(time_array[-1]), Time.units))
            # Dimensions: (Nf)
            # Average velocity of each atom over all the tsteps in the slice
            vx_global_avg = np.mean(fluid_vx_avg, axis=0)
            vy_global_avg = np.mean(fluid_vy_avg, axis=0)
            vz_global_avg = np.mean(fluid_vz_avg, axis=0)

            vx_global_avg_lte = np.mean(fluid_vx_avg_lte, axis=0)
            vy_global_avg_lte = np.mean(fluid_vy_avg_lte, axis=0)
            vz_global_avg_lte = np.mean(fluid_vz_avg_lte, axis=0)

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
            je_x_global = np.zeros_like(gap_height_global)
            je_y_global = np.zeros_like(gap_height_global)
            je_z_global = np.zeros_like(gap_height_global)
            # Voronoi vol
            totVi_global = np.zeros_like(gap_height_global)
            mytotVi_global = np.zeros_like(gap_height_global)
            fluid_vol_global = np.zeros_like(gap_height_global)

            # Dimensions: (time, Nx, Nz)
            # Velocity (chunked in stable region)
            vx_ch_global = np.zeros([time, Nx, Nz], dtype=np.float32)

            vx_ch_whole_global = np.zeros_like(vx_ch_global)
            vx_R1_global = np.zeros_like(vx_ch_global)
            vx_R2_global = np.zeros_like(vx_ch_global)
            vx_R3_global = np.zeros_like(vx_ch_global)
            vx_R4_global = np.zeros_like(vx_ch_global)
            vx_R5_global = np.zeros_like(vx_ch_global)
            uCOMx_global = np.zeros_like(vx_ch_global)
            # Density
            den_ch_global = np.zeros_like(vx_ch_global)
            # Mass flux
            jx_ch_global = np.zeros_like(vx_ch_global)
            mflowrate_ch_global = np.zeros_like(vx_ch_global)
            # Virial
            vir_ch_global = np.zeros_like(vx_ch_global)
            Wxx_ch_global = np.zeros_like(vx_ch_global)
            Wyy_ch_global = np.zeros_like(vx_ch_global)
            Wzz_ch_global = np.zeros_like(vx_ch_global)
            Wxy_ch_global = np.zeros_like(vx_ch_global)
            Wxz_ch_global = np.zeros_like(vx_ch_global)
            Wyz_ch_global = np.zeros_like(vx_ch_global)
            # Temperature
            temp_global = np.zeros_like(vx_ch_global)
            tempx_global = np.zeros_like(vx_ch_global)
            tempy_global = np.zeros_like(vx_ch_global)
            tempz_global = np.zeros_like(vx_ch_global)
            temp_solid_global = np.zeros_like(vx_ch_global)

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
            mytotVi_global = None
            fluid_vol_global = None

            vx_ch_global = None
            vx_ch_whole_global = None
            vx_R1_global = None
            vx_R2_global = None
            vx_R3_global = None
            vx_R4_global = None
            vx_R5_global = None

            uCOMx_global = None
            den_ch_global = None
            # rho_kx_ch_global = None
            sf_global = None
            sf_solid_global = None
            rho_k_global = None
            sf_x_global = None
            sf_y_global = None
            sf_x_solid_global = None
            sf_y_solid_global = None
            # sf_ch_y_global = None
            # rho_ky_ch_global = None
            jx_ch_global = None
            mflowrate_ch_global = None
            je_x_global = None
            je_y_global = None
            je_z_global = None
            vir_ch_global = None
            Wxx_ch_global = None
            Wyy_ch_global = None
            Wzz_ch_global = None
            Wxy_ch_global = None
            Wxz_ch_global = None
            Wyz_ch_global = None
            temp_global = None
            tempx_global = None
            tempy_global = None
            tempz_global = None
            temp_solid_global = None

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
            comm.Gatherv(sendbuf=temp_ch_solid, recvbuf=(temp_solid_global, sendcounts_chunk_vib), root=0)

            # If the virial was not computed, skip
            try:
                comm.Gatherv(sendbuf=totVi, recvbuf=(totVi_global, sendcounts_time), root=0)
                comm.Gatherv(sendbuf=mytotVi, recvbuf=(mytotVi_global, sendcounts_time), root=0)
            except TypeError:
                pass
        else:   # If only bulk
            comm.Gatherv(sendbuf=fluid_vol, recvbuf=(fluid_vol_global, sendcounts_time), root=0)
            try:
                comm.Gatherv(sendbuf=totVi, recvbuf=(totVi_global, sendcounts_time), root=0)
            except TypeError:
                pass

        # For both walls and bulk simulations ------

        # If the heat flux was not computed, skip
        try:
            comm.Gatherv(sendbuf=je_x, recvbuf=(je_x_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=je_y, recvbuf=(je_y_global, sendcounts_time), root=0)
            comm.Gatherv(sendbuf=je_z, recvbuf=(je_z_global, sendcounts_time), root=0)
        except TypeError:
            pass
        comm.Gatherv(sendbuf=vx_ch, recvbuf=(vx_ch_global, sendcounts_chunk_fluid), root=0)

        comm.Gatherv(sendbuf=vx_ch_whole, recvbuf=(vx_ch_whole_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vx_R1, recvbuf=(vx_R1_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vx_R2, recvbuf=(vx_R2_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vx_R3, recvbuf=(vx_R3_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vx_R4, recvbuf=(vx_R4_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vx_R5, recvbuf=(vx_R5_global, sendcounts_chunk_fluid), root=0)

        comm.Gatherv(sendbuf=uCOMx, recvbuf=(uCOMx_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=den_ch, recvbuf=(den_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=jx_ch, recvbuf=(jx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=mflowrate_ch, recvbuf=(mflowrate_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=temp_ch, recvbuf=(temp_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=tempx_ch, recvbuf=(tempx_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=tempy_ch, recvbuf=(tempy_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=tempz_ch, recvbuf=(tempz_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=vir_ch, recvbuf=(vir_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wxx_ch, recvbuf=(Wxx_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wyy_ch, recvbuf=(Wyy_ch_global, sendcounts_chunk_fluid), root=0)
        comm.Gatherv(sendbuf=Wzz_ch, recvbuf=(Wzz_ch_global, sendcounts_chunk_fluid), root=0)
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
            vz_all_var = out.createVariable('Fluid_Vz', 'f4', ('Nf'))
            vx_lte_var = out.createVariable('Fluid_lte_Vx', 'f4', ('Nf'))
            vy_lte_var = out.createVariable('Fluid_lte_Vy', 'f4', ('Nf'))
            vz_lte_var = out.createVariable('Fluid_lte_Vz', 'f4', ('Nf'))

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
            voronoi_volumes_new = out.createVariable('Voronoi_volumes_new', 'f4', ('time'))
            fluid_vol_var = out.createVariable('Fluid_Vol', 'f4', ('time'))

            vx_var =  out.createVariable('Vx', 'f4', ('time', 'x', 'z'))
            vx_whole_var = out.createVariable('Vx_whole', 'f4', ('time', 'x', 'z'))
            vx_R1_var =  out.createVariable('Vx_R1', 'f4', ('time', 'x', 'z'))
            vx_R2_var =  out.createVariable('Vx_R2', 'f4', ('time', 'x', 'z'))
            vx_R3_var =  out.createVariable('Vx_R3', 'f4', ('time', 'x', 'z'))
            vx_R4_var =  out.createVariable('Vx_R4', 'f4', ('time', 'x', 'z'))
            vx_R5_var =  out.createVariable('Vx_R5', 'f4', ('time', 'x', 'z'))
            den_var = out.createVariable('Density', 'f4',  ('time', 'x', 'z'))
            jx_var =  out.createVariable('Jx', 'f4', ('time', 'x', 'z'))
            mflowrate_var = out.createVariable('mdot', 'f4', ('time', 'x', 'z'))
            je_x_var =  out.createVariable('JeX', 'f4', ('time'))
            je_y_var =  out.createVariable('JeY', 'f4', ('time'))
            je_z_var =  out.createVariable('JeZ', 'f4', ('time'))
            vir_var = out.createVariable('Virial', 'f4', ('time', 'x', 'z'))
            Wxx_var = out.createVariable('Wxx', 'f4', ('time', 'x', 'z'))
            Wyy_var = out.createVariable('Wyy', 'f4', ('time', 'x', 'z'))
            Wzz_var = out.createVariable('Wzz', 'f4', ('time', 'x', 'z'))
            Wxy_var = out.createVariable('Wxy', 'f4', ('time', 'x', 'z'))
            Wxz_var = out.createVariable('Wxz', 'f4', ('time', 'x', 'z'))
            Wyz_var = out.createVariable('Wyz', 'f4', ('time', 'x', 'z'))
            temp_var =  out.createVariable('Temperature', 'f4', ('time', 'x', 'z'))
            tempx_var =  out.createVariable('TemperatureX', 'f4', ('time', 'x', 'z'))
            tempy_var =  out.createVariable('TemperatureY', 'f4', ('time', 'x', 'z'))
            tempz_var =  out.createVariable('TemperatureZ', 'f4', ('time', 'x', 'z'))
            temp_solid_var = out.createVariable('Temperature_solid', 'f4', ('time', 'x', 'z'))

            fx_U_var =  out.createVariable('Fx_Upper', 'f4',  ('time', 'x'))
            fy_U_var =  out.createVariable('Fy_Upper', 'f4',  ('time', 'x'))
            fz_U_var =  out.createVariable('Fz_Upper', 'f4',  ('time', 'x'))
            fx_L_var =  out.createVariable('Fx_Lower', 'f4',  ('time', 'x'))
            fy_L_var =  out.createVariable('Fy_Lower', 'f4',  ('time', 'x'))
            fz_L_var =  out.createVariable('Fz_Lower', 'f4',  ('time', 'x'))
            density_bulk_var =  out.createVariable('Density_Bulk', 'f4',  ('time', 'x'))

            # Fill the arrays with data

            vx_all_var[:] = vx_global_avg
            vy_all_var[:] = vy_global_avg
            vz_all_var[:] = vz_global_avg
            vx_lte_var[:] = vx_global_avg_lte
            vy_lte_var[:] = vy_global_avg_lte
            vz_lte_var[:] = vz_global_avg_lte

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
            voronoi_volumes_new[:] = mytotVi_global
            fluid_vol_var[:] = fluid_vol_global

            vx_var[:] = vx_ch_global
            vx_whole_var[:] = vx_ch_whole_global
            vx_R1_var[:] = vx_R1_global
            vx_R2_var[:] = vx_R2_global
            vx_R3_var[:] = vx_R3_global
            vx_R4_var[:] = vx_R4_global
            vx_R5_var[:] = vx_R5_global
            den_var[:] = den_ch_global
            jx_var[:] = jx_ch_global
            mflowrate_var[:] = mflowrate_ch_global
            je_x_var[:] = je_x_global
            je_y_var[:] = je_y_global
            je_z_var[:] = je_z_global
            vir_var[:] = vir_ch_global
            Wxx_var[:] = Wxx_ch_global
            Wyy_var[:] = Wyy_ch_global
            Wzz_var[:] = Wzz_ch_global
            Wxy_var[:] = Wxy_ch_global
            Wxz_var[:] = Wxz_ch_global
            Wyz_var[:] = Wyz_ch_global
            temp_var[:] = temp_global
            tempx_var[:] = tempx_global
            tempy_var[:] = tempy_global
            tempz_var[:] = tempz_global

            fx_U_var[:] = surfU_fx_ch_global
            fy_U_var[:] = surfU_fy_ch_global
            fz_U_var[:] = surfU_fz_ch_global
            fx_L_var[:] = surfL_fx_ch_global
            fy_L_var[:] = surfL_fy_ch_global
            fz_L_var[:] = surfL_fz_ch_global
            temp_solid_var[:] = temp_solid_global
            density_bulk_var[:] = den_bulk_ch_global

            out.close()
            print('Dataset is closed!')

            t1 = timer.time()
            print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        slice1 = slice2
        slice2 += slice_size
