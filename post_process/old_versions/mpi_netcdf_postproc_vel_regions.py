#!/usr/bin/env python

import netCDF4
import sys
import numpy as np
import scipy.constants as sci
import time as timer
from scipy.stats import norm
import matplotlib.pyplot as plt
import sample_quality as sq
# import tesellation as tes
import warnings
import logging
import spatial_bin as sb
# import xarray as xr

from mpi4py import MPI

# Warnings Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.set_printoptions(threshold=sys.maxsize)

# Constants
Kb = 1.38064852e-23 # m2 kg s-2 K-1
ang_to_cm = 1e-8
ang_to_m = 1e-10
fs_to_ns = 1e-6
A_per_fs_to_m_per_s = 1e5
kcalpermolA_to_N = 6.947694845598684e-11
atm_to_mpa = 0.101325
g_to_kg = 1e-3

t0 = timer.time()

def proc(infile, Nx, Nz, slice_size, Ny=1):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    infile = comm.bcast(infile, root=0)
    Nx, Ny, Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)

    data = netCDF4.Dataset(infile)

    # Query the dataset variables
    # for varobj in data.variables.keys():
    #     print(varobj)

    # Query the dataset attributes
    # for name in data.ncattrs():
    #     print("Global attr {} = {}".format(name, getattr(data, name)))

    Time = data.variables["time"]
    out_frequency = 1e3
    # tSteps_tot = (Time.shape[0]-1-3017) * out_frequency
    # time_tot = tSteps_tot * Time.scale_factor

    tSteps_tot = Time.shape[0]-1 #- 3017

    # If the dataset is big, slice it to fit in memory
    if tSteps_tot <= 1000:
        Nslices = 1
    else:
        Nslices = tSteps_tot // slice_size

    Nslices = np.int(Nslices)
    # print(Nslices)
    slice1, slice2 = 1, slice_size+1
    # print(slice1, slice2)

    if rank == 0:
        print('Total simualtion time: {} {}'.format(np.int(tSteps_tot*out_frequency*Time.scale_factor), Time.units))
        print('======> The dataset will be sliced to %g slices!! <======' %Nslices)

    #cell_lengths = np.array(cell_lengths[start:end]).astype(np.float32)
    cell_lengths = data.variables["cell_lengths"]
    cell_lengths = np.array(cell_lengths[0, :]).astype(np.float32)

    # Particle Types ------------------------------
    type = data.variables["type"]
    type = np.array(type[0, :]).astype(np.float32)

    if 'lj' in sys.argv:
        mf, A_per_molecule = 39.948, 1
    elif 'propane' in sys.argv:
        mf, A_per_molecule = 44.09, 3
    elif 'pentane' in sys.argv:
        mf, A_per_molecule = 72.15, 5
    elif 'heptane' in sys.argv:
        mf, A_per_molecule = 100.21, 7

    fluid_idx, solid_idx = [],[]

    # Lennard-Jones
    if np.max(type)==2:
        fluid_idx.append(np.where(type == 1))
        solid_idx.append(np.where(type == 2))

    # Hydrocarbons
    if np.max(type)==3:
        fluid_idx.append(np.where([type == 1, type == 2]))
        solid_idx.append(np.where(type == 3))

    Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/A_per_molecule, \
                            np.max(solid_idx)-np.max(fluid_idx)

    solid_start = np.min(solid_idx)

    if rank == 0:
        print('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,int(Nm),Ns))

    # Coordinates ------------------------------
    # coords_data_unavgd = data.variables["coordinates"]
    coords_data = data.variables["f_position"]

    # Velocities -----------------------------------
    vels_data_unavgd = data.variables["velocities"]     # Used for temp. calculation
    vels_data = data.variables["f_velocity"]

    # Forces on the walls
    forcesU = data.variables["f_fAtomU_avg"]
    forcesL = data.variables["f_fAtomL_avg"]
    # Force on all atoms
    # forces_data_unavgd = data.variables["forces"]
    # forces_data = data.variables["f_force"]

    for slice in range(Nslices):
        # Exclude the first snapshot since data are not stored
        if tSteps_tot <= 1000:
            steps = Time[slice1:].shape[0]
            steps_array = np.array(Time[slice1:]).astype(np.float32)
        else:
            steps = slice_size
            steps_array = np.array(Time[slice1:slice2]).astype(np.float32)
            # print(steps_array)

        # if rank == 0 :
        #     print('From Step %g to %g' %(slice1, slice2))

        # Chunk the data: each processor has a time chunk of the data
        nlocal = steps // size          # no. of tsteps each proc should handle
        start = (rank * nlocal) + (steps*slice) + 1
        end = ((rank + 1) * nlocal) + (steps*slice) + 1


        if rank == size - 1:
            nlocal += steps % size
            start = steps - nlocal + (steps*slice) + 1
            end = steps + (steps*slice) + 1

        chunksize = end - start

        # print(start, end)

        sim_time = np.array(Time[start:end]).astype(np.float32)
        time = comm.gather(sim_time, root=0)

        if rank == 0:
            print('Sampled time: {} {}'.format(np.int(time[-1][-1]), Time.units))

        # EXTRACT DATA  ------------------------------------------
        #---------------------------------------------------------
        # Array dimensions: (time, idx, dimesnion)

        # Coordinates ------------------------------
        coords = np.array(coords_data[start:end]).astype(np.float32)

        fluid_coords = coords[:, :Nf, :]
        fluid_xcoords,fluid_ycoords,fluid_zcoords=coords[:, :Nf, 0], \
                                                  coords[:, :Nf, 1], \
                                                  coords[:, :Nf, 2]

        solid_xcoords, solid_zcoords = coords[:, solid_start:, 0], \
                                        coords[:, solid_start:, 2]

        # Defin the fluid region
        # Extrema of fluid Z-coord in each timestep
        max_fluidX, min_fluidX = np.asarray(np.max(fluid_xcoords,axis=1)), \
                                 np.asarray(np.min(fluid_xcoords,axis=1))
        max_fluidY, min_fluidY = np.asarray(np.max(fluid_ycoords,axis=1)), \
                                 np.asarray(np.min(fluid_ycoords,axis=1))
        max_fluidZ, min_fluidZ = np.asarray(np.max(fluid_zcoords,axis=1)), \
                                 np.asarray(np.min(fluid_zcoords,axis=1))

        # Extrema of fluid coords in all timesteps across all the processors
        min_fluidX_global = comm.allreduce(np.min(min_fluidX), op=MPI.MIN)
        max_fluidX_global = comm.allreduce(np.max(max_fluidX), op=MPI.MAX)
        min_fluidY_global = comm.allreduce(np.min(min_fluidY), op=MPI.MIN)
        max_fluidY_global = comm.allreduce(np.max(max_fluidY), op=MPI.MAX)
        min_fluidZ_global = comm.allreduce(np.min(min_fluidZ), op=MPI.MIN)
        max_fluidZ_global = comm.allreduce(np.max(max_fluidZ), op=MPI.MAX)
        # Average Extrema in all timesteps

        avgmax_fluidZ_global = np.mean(np.array(comm.allgather(np.mean(max_fluidZ))))
        avgmin_fluidZ_global = np.mean(np.array(comm.allgather(np.mean(min_fluidZ))))

        # PBC correction (Z-axis is non-periodic)
        nonperiodic=(2,)
        pdim = [n for n in range(3) if not(n in nonperiodic)]
        hi = np.greater(fluid_coords[:, :, pdim], cell_lengths[pdim]).astype(int)
        lo = np.less(fluid_coords[:, :, pdim], 0.).astype(int)
        fluid_coords[:, :, pdim] += (lo - hi) * cell_lengths[pdim]
        fluid_posmin = np.array([min_fluidX_global, min_fluidY_global, min_fluidZ_global])
        fluid_posmax = np.array([max_fluidX_global, max_fluidY_global, max_fluidZ_global])
        lengths = fluid_posmax - fluid_posmin
        xlength,ylength = lengths[0],lengths[1]

        # Define the surfU and surfL regions

        surfL = np.less_equal(solid_zcoords, avgmax_fluidZ_global/2.)
        surfU = np.greater_equal(solid_zcoords, avgmax_fluidZ_global/2.)

        # Check if surfU and surfL are not equal
        N_surfL,N_surfU = np.sum(surfL[1]),np.sum(surfU[1])
        if N_surfU != N_surfL:
            logger.warning("No. of surfU atoms != No. of surfL atoms")

        surfL_xcoords, surfU_xcoords = solid_xcoords*surfL, solid_xcoords*surfU
        surfL_xcoords, surfU_xcoords = surfL_xcoords[:,N_surfU:], surfU_xcoords[:,:N_surfU]
        surfL_zcoords, surfU_zcoords = solid_zcoords*surfL, solid_zcoords*surfU
        surfL_zcoords, surfU_zcoords = surfL_zcoords[:,N_surfU:], surfU_zcoords[:,:N_surfU]

        # Min and Max SurfU zcoords in each timeStep
        surfU_begin,surfU_end = np.asarray(np.min(surfU_zcoords,axis=1)), \
                                np.asarray(np.max(surfU_zcoords,axis=1))
        surfL_begin,surfL_end = np.asarray(np.min(surfL_zcoords,axis=1)), \
                                np.asarray(np.max(surfL_zcoords,axis=1))

        # Extrema of solid z-coords in all timesteps across all the processors
        surfU_begin_global = comm.allreduce(np.min(surfU_begin), op=MPI.MIN)
        surfL_end_global = comm.allreduce(np.max(surfL_end), op=MPI.MAX)
        surfU_end_global = comm.allreduce(np.max(surfU_end), op=MPI.MAX)
        surfL_begin_global = comm.allreduce(np.min(surfL_begin), op=MPI.MIN)
        # Average Extrema in all timesteps
        avgsurfU_begin_global = np.mean(np.array(comm.allgather(np.mean(surfU_begin))))
        avgsurfL_end_global = np.mean(np.array(comm.allgather(np.mean(surfL_end))))
        avgsurfL_begin_global = np.mean(np.array(comm.allgather(np.mean(surfL_begin))))
        avgsurfU_end_global = np.mean(np.array(comm.allgather(np.mean(surfU_end))))

        # Gap height and COM (in Z-direction) at each timestep
        gap_height = (max_fluidZ - min_fluidZ + surfU_begin - surfL_end) / 2.
        # print(gap_height)
        comZ = (surfU_end - surfL_begin) / 2.

        # Update the lengths z-component
        fluid_posmin[2],fluid_posmax[2] = (avgmin_fluidZ_global + avgsurfL_end_global) / 2., \
                                          (avgmax_fluidZ_global + avgsurfU_begin_global)/2.
        lengths[2] = fluid_posmax[2] - fluid_posmin[2]

        avg_gap_height = (avgmax_fluidZ_global - avgmin_fluidZ_global \
                         + avgsurfU_begin_global - avgsurfL_end_global) / 2.

        if rank == 0:
            print('Average gap Height in the sampling time is {0:.3f}'.format(avg_gap_height))


        # Velocities ------------------------------
        vels = np.array(vels_data[start:end]).astype(np.float32)

        fluid_vx,fluid_vy,fluid_vz = vels[:, :Nf, 0], \
                                     vels[:, :Nf, 1], \
                                     vels[:, :Nf, 2]
        fluid_v = np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)

        # Forces ------------------------------
        # forces = np.array(forces_data[start:end]).astype(np.float32)
        #
        # fluid_fx,fluid_fy,fluid_fz = forces[:, :Nf, 0], \
        #                              forces[:, :Nf, 1], \
        #                              forces[:, :Nf, 2]
        # solid_fx,solid_fy,solid_fz = forces[:, solid_start:, 0], \
        #                              forces[:, solid_start:, 1], \
        #                              forces[:, solid_start:, 2]

        # Unavgd velocities--------------------------------------
        vels_t = np.array(vels_data_unavgd[start:end]).astype(np.float32)

        fluid_vx_t,fluid_vy_t,fluid_vz_t = vels_t[:, :Nf, 0], \
                                           vels_t[:, :Nf, 1], \
                                           vels_t[:, :Nf, 2]

        fluid_v_t = np.sqrt(fluid_vx_t**2+fluid_vy_t**2+fluid_vz_t**2)/A_per_molecule

        # REGIONS ------------------------------------------------
        #---------------------------------------------------------

        # Stresses in the wall ------------------------------
        forcesU_data = np.array(forcesU[start:end]).astype(np.float32)
        forcesL_data = np.array(forcesL[start:end]).astype(np.float32)

        surfU_fx,surfU_fy,surfU_fz = forcesU_data[:, solid_start:, 0][:,:N_surfU], \
                                     forcesU_data[:, solid_start:, 1][:,:N_surfU], \
                                     forcesU_data[:, solid_start:, 2][:,:N_surfU]

        surfL_fx,surfL_fy,surfL_fz = forcesL_data[:, solid_start:, 0][:,N_surfU:], \
                                     forcesL_data[:, solid_start:, 1][:,N_surfU:], \
                                     forcesL_data[:, solid_start:, 2][:,N_surfU:]

        # Bulk Region ----------------------------------
        total_box_Height = avgsurfU_end_global - avgsurfL_begin_global
        print(total_box_Height)
        bulkStartZ = 0.25 * total_box_Height
        bulkEndZ = 0.75 * total_box_Height
        bulkHeight = bulkEndZ - bulkStartZ
        bulk_hi = np.less_equal(fluid_zcoords, bulkEndZ)
        bulk_lo = np.greater_equal(fluid_zcoords, bulkStartZ)
        bulk_region = np.logical_and(bulk_lo, bulk_hi)
        bulk_vol = bulkHeight * xlength * ylength
        bulk_N = np.sum(bulk_region, axis=1)

        # Virial Pressure in the fluid -------------------------
        voronoi_vol_data = data.variables["f_Vi_avg"]
        voronoi_vol = np.array(voronoi_vol_data[start:end]).astype(np.float32)

        Vi = voronoi_vol[:, :Nf] * bulk_region
        totVi = np.sum(Vi, axis=1)

        virial_data = data.variables["f_Wi_avg"]
        virial = np.array(virial_data[start:end]).astype(np.float32)

        # W1 = virial[:, :Nf, 0] * bulk_region
        # W2 = virial[:, :Nf, 1] * bulk_region
        # W3 = virial[:, :Nf, 2] * bulk_region
        # totW1, totW2, totW3 = np.sum(W1, axis=1), np.sum(W2, axis=1), np.sum(W3, axis=1)
        # press = -(totW1+totW2+totW3)*atm_to_mpa / (3 * totVi)

        #Stable Region ---------------------------------
        stableStartX = 0.4 * xlength
        stableEndX = 0.8 * xlength
        stable_length = stableEndX - stableStartX
        stable_hi = np.less_equal(fluid_xcoords, stableEndX)
        stable_lo = np.greater_equal(fluid_xcoords, stableStartX)
        stable_region = np.logical_and(stable_lo, stable_hi)
        stable_vol = gap_height * stable_length * ylength
        stable_N = np.sum(stable_region, axis=1)

        # Get the mass flow rate and mass flux in the stable region
        vels_stable = (fluid_vx/A_per_molecule) * stable_region
        mflux_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * (mf/sci.N_A))/ (stable_vol * (ang_to_m)**3)      # g/m2.ns
        mflowrate_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * (mf/sci.N_A)) / (stable_length * ang_to_m)

        #Pump Region --------------------------------------
        pumpStartX = 0.0 * xlength
        pumpEndX = 0.2 * xlength
        pump_length = pumpEndX-pumpStartX
        pump_hi = np.less_equal(fluid_xcoords, pumpEndX)
        pump_lo = np.greater_equal(fluid_xcoords, pumpStartX)
        pump_region = np.logical_and(pump_lo, pump_hi)
        pump_vol = gap_height*pump_length*ylength
        pump_N = np.sum(pump_region, axis=1)

        # Measure mass flux in the pump region
        vels_pump = (fluid_vx/A_per_molecule) * pump_region
        mflux_pump = (np.sum(vels_pump, axis=1) * (ang_to_m/fs_to_ns) * (mf/sci.N_A)) / (pump_vol * (ang_to_m)**3)      # g/m2.ns

        # The Grid -------------------------------------------------------------
        dim = np.array([Nx, Ny, Nz])

        # Bounds Fluid
        bounds = [np.arange(dim[i] + 1) / dim[i] * lengths[i] + fluid_posmin[i]
                        for i in range(3)]

        xx, yy, zz = np.meshgrid(bounds[0], bounds[1], bounds[2])

        xx = np.transpose(xx, (1, 0, 2))
        yy = np.transpose(yy, (1, 0, 2))
        zz = np.transpose(zz, (1, 0, 2))

        dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
        dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
        dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]

        vol = dx * dy * dz

        # Bounds surfU
        surfURange = np.arange(dim[0] + 1) / dim[0] * xlength
        bounds_surfU = [surfURange, np.array([0,ylength]), np.array([surfU_begin_global,surfU_end_global])]

        xx_surfU, yy_surfU, zz_surfU = np.meshgrid(bounds_surfU[0], bounds_surfU[1], bounds_surfU[2])
        xx_surfU = np.transpose(xx_surfU, (1, 0, 2))
        yy_surfU = np.transpose(yy_surfU, (1, 0, 2))
        zz_surfU = np.transpose(zz_surfU, (1, 0, 2))

        # Bounds surfL
        bounds_surfL = [surfURange, np.array([0,ylength]), np.array([surfL_begin_global,surfL_end_global])]

        xx_surfL, yy_surfL, zz_surfL = np.meshgrid(bounds_surfL[0], bounds_surfL[1], bounds_surfL[2])
        xx_surfL = np.transpose(xx_surfL, (1, 0, 2))
        yy_surfL = np.transpose(yy_surfL, (1, 0, 2))
        zz_surfL = np.transpose(zz_surfL, (1, 0, 2))

        # Bounds Stable
        stableRange = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX
        bounds_stable = [stableRange, np.array([bounds[1]]), np.array([bounds[2]])]

        xx_stable, yy_stable, zz_stable = np.meshgrid(bounds_stable[0], bounds_stable[1], bounds_stable[2])
        xx_stable = np.transpose(xx_stable, (1, 0, 2))
        yy_stable = np.transpose(yy_stable, (1, 0, 2))
        zz_stable = np.transpose(zz_stable, (1, 0, 2))

        # Bounds Bulk
        bulkRange = np.arange(dim[2] + 1) / dim[2] * bulkHeight + bulkStartZ
        bounds_bulk = [np.array([bounds[0]]), np.array([bounds[1]]), bulkRange]

        xx_bulk, yy_bulk, zz_bulk = np.meshgrid(bounds_bulk[0], bounds_bulk[1], bounds_bulk[2])
        xx_bulk = np.transpose(xx_bulk, (1, 0, 2))
        yy_bulk = np.transpose(yy_bulk, (1, 0, 2))
        zz_bulk = np.transpose(zz_bulk, (1, 0, 2))
        dx_bulk = xx_bulk[1:, 1:, 1:] - xx_bulk[:-1, :-1, :-1]
        dy_bulk = yy_bulk[1:, 1:, 1:] - yy_bulk[:-1, :-1, :-1]
        dz_bulk = zz_bulk[1:, 1:, 1:] - zz_bulk[:-1, :-1, :-1]
        vol_bulk = dx_bulk * dy_bulk * dz_bulk


        # # Bounds R1
        # R1Range = np.arange(dim[0] + 1) / dim[0] * pump_length
        # bounds_R1 = [R1Range, np.array([bounds[1]]), np.array([bounds[2]])]
        #
        # xx_R1, yy_R1, zz_R1 = np.meshgrid(bounds_R1[0], bounds_R1[1], bounds_R1[2])
        # xx_R1 = np.transpose(xx_R1, (1, 0, 2))
        # yy_R1 = np.transpose(yy_R1, (1, 0, 2))
        # zz_R1 = np.transpose(zz_R1, (1, 0, 2))
        #
        # # Bounds R2
        # R2Range = np.arange(dim[0] + 1) / dim[0] * pump_length + bounds_R1[0][-1]
        # bounds_R2 = [R2Range, np.array([bounds[1]]), np.array([bounds[2]])]
        #
        # xx_R2, yy_R2, zz_R2 = np.meshgrid(bounds_R2[0], bounds_R2[1], bounds_R2[2])
        # xx_R2 = np.transpose(xx_R2, (1, 0, 2))
        # yy_R2 = np.transpose(yy_R2, (1, 0, 2))
        # zz_R2 = np.transpose(zz_R2, (1, 0, 2))
        #
        # # Bounds R3
        # R3Range = np.arange(dim[0] + 1) / dim[0] * pump_length + bounds_R2[0][-1]
        # bounds_R3 = [R3Range, np.array([bounds[1]]), np.array([bounds[2]])]
        #
        # xx_R3, yy_R3, zz_R3 = np.meshgrid(bounds_R3[0], bounds_R3[1], bounds_R3[2])
        # xx_R3 = np.transpose(xx_R3, (1, 0, 2))
        # yy_R3 = np.transpose(yy_R3, (1, 0, 2))
        # zz_R3 = np.transpose(zz_R3, (1, 0, 2))
        #
        # # Bounds R4
        # R4Range = np.arange(dim[0] + 1) / dim[0] * pump_length + bounds_R3[0][-1]
        # bounds_R4 = [R4Range, np.array([bounds[1]]), np.array([bounds[2]])]
        #
        # xx_R4, yy_R4, zz_R4 = np.meshgrid(bounds_R4[0], bounds_R4[1], bounds_R4[2])
        # xx_R4 = np.transpose(xx_R4, (1, 0, 2))
        # yy_R4 = np.transpose(yy_R4, (1, 0, 2))
        # zz_R4 = np.transpose(zz_R4, (1, 0, 2))
        #
        # # Bounds R5
        # R5Range = np.arange(dim[0] + 1) / dim[0] * pump_length + bounds_R4[0][-1]
        # bounds_R5 = [R5Range, np.array([bounds[1]]), np.array([bounds[2]])]
        #
        # xx_R5, yy_R5, zz_R5 = np.meshgrid(bounds_R5[0], bounds_R5[1], bounds_R5[2])
        # xx_R5 = np.transpose(xx_R5, (1, 0, 2))
        # yy_R5 = np.transpose(yy_R5, (1, 0, 2))
        # zz_R5 = np.transpose(zz_R5, (1, 0, 2))


        # Thermodynamic and Mechanical Quantities ---------------------------------
        # -------------------------------------------------------------------------

        # # local buffers
        N = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
        N_bulk = np.zeros_like(N)
        N_stable = np.zeros_like(N)

        N_R1 = np.zeros_like(N)
        N_R2 = np.zeros_like(N)
        N_R3 = np.zeros_like(N)
        N_R4 = np.zeros_like(N)
        N_R5 = np.zeros_like(N)

        v_t_ch = np.zeros_like(N)
        den_ch = np.zeros_like(N)
        vx_ch = np.zeros_like(N)

        # vx_R1 = np.zeros_like(N)
        # vx_R2 = np.zeros_like(N)
        # vx_R3 = np.zeros_like(N)
        # vx_R4 = np.zeros_like(N)
        # vx_R5 = np.zeros_like(N)

        jx_ch = np.zeros_like(N)
        vir_ch = np.zeros_like(N)

        surfU_fx_ch = np.zeros([chunksize, Nx], dtype=np.float32)
        surfU_fy_ch = np.zeros_like(surfU_fx_ch)
        surfU_fz_ch = np.zeros_like(surfU_fx_ch)
        surfL_fx_ch = np.zeros_like(surfU_fx_ch)
        surfL_fy_ch = np.zeros_like(surfU_fx_ch)
        surfL_fz_ch = np.zeros_like(surfU_fx_ch)
        den_ch_bulk = np.zeros_like(surfU_fx_ch)

        Ntotal = 0

        for i in range(Nx):
            for k in range(Nz):
                # create a mask to filter only particles in grid cell

                # Fluid -----------------------------------------
                xlo = np.less(fluid_xcoords, xx[i+1, 0, k])
                xhi = np.greater_equal(fluid_xcoords, xx[i, 0, k])
                cellx = np.logical_and(xlo, xhi)

                zlo = np.less(fluid_zcoords, zz[i, 0, k+1])
                zhi = np.greater_equal(fluid_zcoords, zz[i, 0, k])
                cellz = np.logical_and(zlo, zhi)

                cell = np.logical_and(cellx,cellz)

                # Stable --------------------------------------
                xlo_stable = np.less(fluid_xcoords, xx_stable[i+1, 0, k])
                xhi_stable = np.greater_equal(fluid_xcoords, xx_stable[i, 0, k])
                cell_stablex = np.logical_and(xlo_stable, xhi_stable)

                zlo_stable = np.less(fluid_zcoords, zz_stable[i, 0, k+1])
                zhi_stable = np.greater_equal(fluid_zcoords, zz_stable[i, 0, k])
                cell_stablez = np.logical_and(zlo_stable, zhi_stable)

                cell_stable = np.logical_and(cell_stablex,cell_stablez)
                #-------------------------------------------------

                # xlo_R1 = np.less(fluid_xcoords, xx_R1[i+1, 0, k])
                # xhi_R1 = np.greater_equal(fluid_xcoords, xx_R1[i, 0, k])
                # cell_R1x = np.logical_and(xlo_R1, xhi_R1)
                #
                # zlo_R1 = np.less(fluid_zcoords, zz_R1[i, 0, k+1])
                # zhi_R1 = np.greater_equal(fluid_zcoords, zz_R1[i, 0, k])
                # cell_R1z = np.logical_and(zlo_R1, zhi_R1)
                #
                # cell_R1 = np.logical_and(cell_R1x,cell_R1z)
                #
                #
                # xlo_R2 = np.less(fluid_xcoords, xx_R2[i+1, 0, k])
                # xhi_R2 = np.greater_equal(fluid_xcoords, xx_R2[i, 0, k])
                # cell_R2x = np.logical_and(xlo_R2, xhi_R2)
                #
                # zlo_R2 = np.less(fluid_zcoords, zz_R2[i, 0, k+1])
                # zhi_R2 = np.greater_equal(fluid_zcoords, zz_R2[i, 0, k])
                # cell_R2z = np.logical_and(zlo_R2, zhi_R2)
                #
                # cell_R2 = np.logical_and(cell_R2x,cell_R2z)
                #
                #
                # xlo_R3 = np.less(fluid_xcoords, xx_R3[i+1, 0, k])
                # xhi_R3 = np.greater_equal(fluid_xcoords, xx_R3[i, 0, k])
                # cell_R3x = np.logical_and(xlo_R3, xhi_R3)
                #
                # zlo_R3 = np.less(fluid_zcoords, zz_R3[i, 0, k+1])
                # zhi_R3 = np.greater_equal(fluid_zcoords, zz_R3[i, 0, k])
                # cell_R3z = np.logical_and(zlo_R3, zhi_R3)
                #
                # cell_R3 = np.logical_and(cell_R3x, cell_R3z)
                #
                #
                #
                # xlo_R4 = np.less(fluid_xcoords, xx_R4[i+1, 0, k])
                # xhi_R4 = np.greater_equal(fluid_xcoords, xx_R4[i, 0, k])
                # cell_R4x = np.logical_and(xlo_R4, xhi_R4)
                #
                # zlo_R4 = np.less(fluid_zcoords, zz_R4[i, 0, k+1])
                # zhi_R4 = np.greater_equal(fluid_zcoords, zz_R4[i, 0, k])
                # cell_R4z = np.logical_and(zlo_R4, zhi_R4)
                #
                # cell_R4 = np.logical_and(cell_R4x, cell_R4z)
                #
                #
                #
                # xlo_R5 = np.less(fluid_xcoords, xx_R5[i+1, 0, k])
                # xhi_R5 = np.greater_equal(fluid_xcoords, xx_R5[i, 0, k])
                # cell_R5x = np.logical_and(xlo_R5, xhi_R5)
                #
                # zlo_R5 = np.less(fluid_zcoords, zz_R5[i, 0, k+1])
                # zhi_R5 = np.greater_equal(fluid_zcoords, zz_R5[i, 0, k])
                # cell_R5z = np.logical_and(zlo_R5, zhi_R5)
                #
                # cell_R5 = np.logical_and(cell_R5x, cell_R5z)



                # Bulk -----------------------------------
                xlo_bulk = np.less(fluid_xcoords, xx_bulk[i+1, 0, k])
                xhi_bulk = np.greater_equal(fluid_xcoords, xx_bulk[i, 0, k])
                cellx_bulk = np.logical_and(xlo_bulk, xhi_bulk)

                zlo_bulk = np.less(fluid_zcoords, zz_bulk[i, 0, k+1])
                zhi_bulk = np.greater_equal(fluid_zcoords, zz_bulk[i, 0, k])
                cellz_bulk = np.logical_and(zlo_bulk, zhi_bulk)

                cell_bulk = np.logical_and(cellx_bulk,cellz_bulk)

                # Count particles in the fluid cell
                N[:, i, k] = np.sum(cell, axis=1)

                # Count particles in the fluid cell
                N_stable[:, i, k] = np.sum(cell_stable, axis=1)

                # N_R1[:, i, k] = np.sum(cell_R1, axis=1)
                # Nzero_R1 = np.less(N_R1[:, i, k], 1)
                # N_R1[Nzero_R1, i, k] = 1
                #
                # N_R2[:, i, k] = np.sum(cell_R2, axis=1)
                # Nzero_R2 = np.less(N_R2[:, i, k], 1)
                # N_R2[Nzero_R2, i, k] = 1
                #
                # N_R3[:, i, k] = np.sum(cell_R3, axis=1)
                # Nzero_R3 = np.less(N_R3[:, i, k], 1)
                # N_R3[Nzero_R3, i, k] = 1
                #
                # N_R4[:, i, k] = np.sum(cell_R4, axis=1)
                # Nzero_R4 = np.less(N_R4[:, i, k], 1)
                # N_R4[Nzero_R4, i, k] = 1
                #
                # N_R5[:, i, k] = np.sum(cell_R5, axis=1)
                # Nzero_R5 = np.less(N_R5[:, i, k], 1)
                # N_R5[Nzero_R5, i, k] = 1

                Nzero = np.less(N[:, i, k], 1)
                N[Nzero, i, k] = 1

                Nzero_stable = np.less(N_stable[:, i, k], 1)
                N_stable[Nzero_stable, i, k] = 1

                # Count particles in the bulk
                N_bulk[:, i, k] = np.sum(cell_bulk, axis=1)

                # Temperature
                v_t_ch[:, i, k] = np.sum((fluid_v_t**2) * cellx, axis=1) / (3*N[:, i, k]/A_per_molecule)

                # Density (bulk)
                den_ch_bulk[:, i] = (N_bulk[:, i, 0]/A_per_molecule) / vol_bulk[i, 0, 0]

                # Density (whole fluid)
                den_ch[:, i, k] = (N[:, i, k]/A_per_molecule) / vol[i, 0, k]

                # Vx (stable region)
                vx_ch[:, i, k] = np.sum(fluid_vx * cell_stable, axis=1) / (N_stable[:, i, k]/A_per_molecule)

                # vx_R1[:, i, k] = np.sum(fluid_vx * cell_R1, axis=1) / (N_R1[:, i, k]/A_per_molecule)
                # vx_R2[:, i, k] = np.sum(fluid_vx * cell_R2, axis=1) / (N_R2[:, i, k]/A_per_molecule)
                # vx_R3[:, i, k] = np.sum(fluid_vx * cell_R3, axis=1) / (N_R3[:, i, k]/A_per_molecule)
                # vx_R4[:, i, k] = np.sum(fluid_vx * cell_R4, axis=1) / (N_R4[:, i, k]/A_per_molecule)
                # vx_R5[:, i, k] = np.sum(fluid_vx * cell_R5, axis=1) / (N_R5[:, i, k]/A_per_molecule)

                # Mass flux (whole fluid)
                jx_ch[:, i, k] = vx_ch[:, i, k] * den_ch[:, i, k]

                # Virial Pressure
                W1_ch = np.sum(virial[:, :Nf, 0] * cell_bulk, axis=1)
                W2_ch = np.sum(virial[:, :Nf, 1] * cell_bulk, axis=1)
                W3_ch = np.sum(virial[:, :Nf, 2] * cell_bulk, axis=1)
                vir_ch[:, i, k] = -(W1_ch + W2_ch + W3_ch)

                # For Solid
                xloU = np.less(surfU_xcoords, xx_surfU[i+1, 0, 0])
                xhiU = np.greater_equal(surfU_xcoords, xx_surfU[i, 0, 0])
                cellxU = np.logical_and(xloU, xhiU)
                xloL = np.less(surfL_xcoords, xx_surfL[i+1, 0, 0])
                xhiL = np.greater_equal(surfL_xcoords, xx_surfL[i, 0, 0])
                cellxL = np.logical_and(xloL, xhiL)

                # Wall stresses
                # print(surfU_fx.shape)
                surfU_fx_ch[:, i] = np.sum(surfU_fx * cellxU, axis=1)
                surfU_fy_ch[:, i] = np.sum(surfU_fy * cellxU, axis=1)
                surfU_fz_ch[:, i] = np.sum(surfU_fz * cellxU, axis=1)

                surfL_fx_ch[:, i] = np.sum(surfL_fx * cellxL, axis=1)
                surfL_fy_ch[:, i] = np.sum(surfL_fy * cellxL, axis=1)
                surfL_fz_ch[:, i] = np.sum(surfL_fz * cellxL, axis=1)

        # Gather the data -------------------------------------

        if rank==0:

            Lx, Ly, h = xlength, ylength, avg_gap_height
            time = steps_array
            # Center of Mass in Z
            comZ_global = np.zeros(steps, dtype=np.float32)
            # Gap Height
            gap_height_global = np.zeros_like(time)
            # Flow Rate and Mass flux (In time)
            mflowrate_stable_global = np.zeros_like(time)
            mflux_stable_global = np.zeros_like(time)
            mflux_pump_global = np.zeros_like(time)

            # Voronoi vol
            vor_vol_global = np.zeros_like(time)
            surfL_begin_global = np.zeros_like(time)
            surfU_end_global = np.zeros_like(time)

            # Surface Forces
            surfU_fx_ch_global = np.zeros([steps, Nx], dtype=np.float32)
            surfU_fy_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfU_fz_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfL_fx_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfL_fy_ch_global = np.zeros_like(surfU_fx_ch_global)
            surfL_fz_ch_global = np.zeros_like(surfU_fx_ch_global)
            den_ch_bulk_global = np.zeros_like(surfU_fx_ch_global)

            # Temperature
            v_t_global = np.zeros([steps, Nx, Nz], dtype=np.float32)
            # Velocity (chunked in stable region)
            vx_ch_global = np.zeros_like(v_t_global)

            # vx_R1_global = np.zeros_like(v_t_global)
            # vx_R2_global = np.zeros_like(v_t_global)
            # vx_R3_global = np.zeros_like(v_t_global)
            # vx_R4_global = np.zeros_like(v_t_global)
            # vx_R5_global = np.zeros_like(v_t_global)

            # Density
            den_ch_global = np.zeros_like(v_t_global)
            # Mass flux
            jx_ch_global = np.zeros_like(v_t_global)
            # Virial
            vir_ch_global = np.zeros_like(v_t_global)

        else:
            comZ_global = None
            gap_height_global = None
            mflowrate_stable_global = None
            mflux_stable_global = None
            mflux_pump_global = None
            v_t_global = None

            # vx_R1_global = None
            # vx_R2_global = None
            # vx_R3_global = None
            # vx_R4_global = None
            # vx_R5_global = None

            surfU_fx_ch_global = None
            surfU_fy_ch_global = None
            surfU_fz_ch_global = None
            surfL_fx_ch_global = None
            surfL_fy_ch_global = None
            surfL_fz_ch_global = None
            vx_ch_global = None
            den_ch_global = None
            den_ch_bulk_global = None
            jx_ch_global = None
            vir_ch_global = None
            vor_vol_global = None

        sendcounts_time = np.array(comm.gather(comZ.size, root=0))
        sendcounts_chunk = np.array(comm.gather(v_t_ch.size, root=0))
        sendcounts_chunkS = np.array(comm.gather(surfU_fx_ch.size, root=0))
        sendcounts_chunk_bulk = np.array(comm.gather(den_ch_bulk.size, root=0))

        comm.Gatherv(sendbuf=comZ, recvbuf=(comZ_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=gap_height, recvbuf=(gap_height_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=mflowrate_stable, recvbuf=(mflowrate_stable_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=mflux_stable, recvbuf=(mflux_stable_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=mflux_pump, recvbuf=(mflux_pump_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=totVi, recvbuf=(vor_vol_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=surfU_end, recvbuf=(surfU_end_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=surfL_begin, recvbuf=(surfL_begin_global, sendcounts_time), root=0)

        comm.Gatherv(sendbuf=v_t_ch, recvbuf=(v_t_global, sendcounts_chunk), root=0)
        comm.Gatherv(sendbuf=vx_ch, recvbuf=(vx_ch_global, sendcounts_chunk), root=0)

        # vx_global = np.array(comm.gather(fluid_vx, root=0))
        # vy_global = np.array(comm.gather(fluid_vy, root=0))

        # comm.Gatherv(sendbuf=vx_R1, recvbuf=(vx_R1_global, sendcounts_chunk), root=0)
        # comm.Gatherv(sendbuf=vx_R2, recvbuf=(vx_R2_global, sendcounts_chunk), root=0)
        # comm.Gatherv(sendbuf=vx_R3, recvbuf=(vx_R3_global, sendcounts_chunk), root=0)
        # comm.Gatherv(sendbuf=vx_R4, recvbuf=(vx_R4_global, sendcounts_chunk), root=0)
        # comm.Gatherv(sendbuf=vx_R5, recvbuf=(vx_R5_global, sendcounts_chunk), root=0)

        comm.Gatherv(sendbuf=den_ch, recvbuf=(den_ch_global, sendcounts_chunk), root=0)
        comm.Gatherv(sendbuf=jx_ch, recvbuf=(jx_ch_global, sendcounts_chunk), root=0)
        comm.Gatherv(sendbuf=vir_ch, recvbuf=(vir_ch_global, sendcounts_chunk), root=0)

        comm.Gatherv(sendbuf=surfU_fx_ch, recvbuf=(surfU_fx_ch_global, sendcounts_chunkS), root=0)
        comm.Gatherv(sendbuf=surfU_fy_ch, recvbuf=(surfU_fy_ch_global, sendcounts_chunkS), root=0)
        comm.Gatherv(sendbuf=surfU_fz_ch, recvbuf=(surfU_fz_ch_global, sendcounts_chunkS), root=0)
        comm.Gatherv(sendbuf=surfL_fx_ch, recvbuf=(surfL_fx_ch_global, sendcounts_chunkS), root=0)
        comm.Gatherv(sendbuf=surfL_fy_ch, recvbuf=(surfL_fy_ch_global, sendcounts_chunkS), root=0)
        comm.Gatherv(sendbuf=surfL_fz_ch, recvbuf=(surfL_fz_ch_global, sendcounts_chunkS), root=0)
        comm.Gatherv(sendbuf=den_ch_bulk, recvbuf=(den_ch_bulk_global, sendcounts_chunk_bulk), root=0)


        if rank == 0:
            # Temp
            temp = ((mf*g_to_kg/sci.N_A) * v_t_global * A_per_fs_to_m_per_s**2) / Kb       # Kelvin

            # Velocity
            vx_ch_global *=  A_per_fs_to_m_per_s       # m/s

            # vx_global = np.reshape(vx_global, [steps, Nf])
            # vx_global = np.reshape(vy_global, [steps, Nf])
            # vx_global *=  A_per_fs_to_m_per_s       # m/s
            # vy_global *=  A_per_fs_to_m_per_s       # m/s

            # vx_R1_global *=  A_per_fs_to_m_per_s       # m/s
            # vx_R2_global *=  A_per_fs_to_m_per_s       # m/s
            # vx_R3_global *=  A_per_fs_to_m_per_s       # m/s
            # vx_R4_global *=  A_per_fs_to_m_per_s       # m/s
            # vx_R5_global *=  A_per_fs_to_m_per_s       # m/s

            # Density
            den_ch_global *= (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
            den_ch_bulk_global *= (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3

            # Mass flux
            jx_ch_global *= (ang_to_m/fs_to_ns) * (mf/sci.N_A) / (ang_to_m**3)

            # Virial Pressure (Mpa*A^3)
            vir_ch_global *= atm_to_mpa

            # Forces in the wall
            surfU_fx_ch_global *= kcalpermolA_to_N       # N
            surfU_fy_ch_global *= kcalpermolA_to_N
            surfU_fz_ch_global *= kcalpermolA_to_N

            surfL_fx_ch_global *= kcalpermolA_to_N
            surfL_fy_ch_global *= kcalpermolA_to_N
            surfL_fz_ch_global *= kcalpermolA_to_N

            outfile = f"{infile.split('.')[0]}_{Nx}x{Nz}_{slice}.nc"
            out = netCDF4.Dataset(outfile, 'w', format='NETCDF3_64BIT_OFFSET')

            out.createDimension('x', Nx)
            out.createDimension('z', Nz)
            # out.createDimension('Nf', Nf)
            out.createDimension('step', steps)

            out.setncattr("Lx", Lx)
            out.setncattr("Ly", Ly)
            out.setncattr("h", h)

            time_var = out.createVariable('Time', 'f4', ('step'))
            com_var =  out.createVariable('COM', 'f4', ('step'))
            gap_height_var =  out.createVariable('Height', 'f4', ('step'))
            voronoi_volumes = out.createVariable('Voronoi_volumes', 'f4', ('step'))
            mflux_stable = out.createVariable('mflux_stable', 'f4', ('step'))
            mflowrate_stable = out.createVariable('mflow_rate_stable', 'f4', ('step'))
            mflux_pump = out.createVariable('mflux_pump', 'f4', ('step'))

            temp_var =  out.createVariable('Temperature', 'f4', ('step', 'x', 'z'))

            vx_var =  out.createVariable('Vx', 'f4', ('step', 'x', 'z'))

            # vx_all_var = out.createVariable('Fluid_Vx', 'f4', ('step', 'Nf'))
            # vy_all_var = out.createVariable('Fluid_Vy', 'f4', ('step', 'Nf'))

            # vx_R1_var =  out.createVariable('Vx_R1', 'f4', ('step', 'x', 'z'))
            # vx_R2_var =  out.createVariable('Vx_R2', 'f4', ('step', 'x', 'z'))
            # vx_R3_var =  out.createVariable('Vx_R3', 'f4', ('step', 'x', 'z'))
            # vx_R4_var =  out.createVariable('Vx_R4', 'f4', ('step', 'x', 'z'))
            # vx_R5_var =  out.createVariable('Vx_R5', 'f4', ('step', 'x', 'z'))

            jx_var =  out.createVariable('Jx', 'f4', ('step', 'x', 'z'))
            vir_var = out.createVariable('Virial', 'f4', ('step', 'x', 'z'))
            den_var = out.createVariable('Density', 'f4',  ('step', 'x', 'z'))

            fx_U_var =  out.createVariable('Fx_Upper', 'f4',  ('step', 'x'))
            fy_U_var =  out.createVariable('Fy_Upper', 'f4',  ('step', 'x'))
            fz_U_var =  out.createVariable('Fz_Upper', 'f4',  ('step', 'x'))

            fx_L_var =  out.createVariable('Fx_Lower', 'f4',  ('step', 'x'))
            fy_L_var =  out.createVariable('Fy_Lower', 'f4',  ('step', 'x'))
            fz_L_var =  out.createVariable('Fz_Lower', 'f4',  ('step', 'x'))

            surfL_begin_var =  out.createVariable('SurfL_begin', 'f4',  ('step'))
            surfU_end_var =  out.createVariable('SurfU_end', 'f4',  ('step'))

            density_bulk_var =  out.createVariable('Density_Bulk', 'f4',  ('step', 'x'))

            time_var[:] = time #+ 25000000
            com_var[:] = comZ_global
            gap_height_var[:] = gap_height_global
            voronoi_volumes[:] = vor_vol_global
            mflux_pump[:] = mflux_pump_global
            mflowrate_stable[:] = mflowrate_stable_global
            mflux_stable[:] = mflux_stable_global
            surfU_end_var[:] = surfU_end_global
            surfL_begin_var[:] = surfL_begin_global

            # vx_all_var[:] = vx_global
            # vy_all_var[:] = vy_global

            # vx_R1_var[:] = vx_R1_global
            # vx_R2_var[:] = vx_R2_global
            # vx_R3_var[:] = vx_R3_global
            # vx_R4_var[:] = vx_R4_global
            # vx_R5_var[:] = vx_R5_global

            temp_var[:] = temp
            vx_var[:] = vx_ch_global
            jx_var[:] = jx_ch_global
            vir_var[:] = vir_ch_global
            den_var[:] = den_ch_global

            fx_U_var[:] = surfU_fx_ch_global
            fy_U_var[:] = surfU_fy_ch_global
            fz_U_var[:] = surfU_fz_ch_global
            fx_L_var[:] = surfL_fx_ch_global
            fy_L_var[:] = surfL_fy_ch_global
            fz_L_var[:] = surfL_fz_ch_global
            density_bulk_var[:] = den_ch_bulk_global

            out.close()
            print('Dataset is closed!')

            t1 = timer.time()
            print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        slice1 = slice2
        slice2 += slice_size

if __name__ == "__main__":
   proc(sys.argv[1],np.int(sys.argv[2]),np.int(sys.argv[3]),np.int(sys.argv[4]))
