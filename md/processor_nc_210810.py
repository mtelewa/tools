#!/usr/bin/env python

import netCDF4
import sys
import numpy as np
import scipy.constants as sci
import time as timer
# import tesellation as tes
import warnings
import logging
import utils
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
    tSteps_tot = Time.shape[0]-1
    out_frequency = 1e3

    # If the dataset is more than 1M tsteps, slice it to fit in memory
    if tSteps_tot <= 1000:
        Nslices = 1
    else:
        Nslices = tSteps_tot // slice_size

    Nslices = np.int(Nslices)
    slice1, slice2 = 1, slice_size+1

    if rank == 0:
        print('Total simualtion time: {} {}'.format(np.int(tSteps_tot*out_frequency*Time.scale_factor), Time.units))
        print('======> The dataset will be sliced to %g slices!! <======' %Nslices)

    cell_lengths = data.variables["cell_lengths"]
    cell_lengths = np.array(cell_lengths[0, :]).astype(np.float32)

    # Particle Types ------------------------------
    type = data.variables["type"]
    type = np.array(type[0, :]).astype(np.float32)

    mCH2, mCH3, mCH_avg = 12, 13, 12.5

    if 'lj' in sys.argv:
        mf, A_per_molecule = 39.948, 1
    if 'propane' in sys.argv:
        mf, A_per_molecule = 44.09, 3
    if 'pentane' in sys.argv:
        mf, A_per_molecule = 72.15, 5
    if 'heptane' in sys.argv:
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
    if 'no-temp' not in sys.argv:
        #print('No variable T defined!')
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

        # Chunk the data: each processor has a time chunk of the data
        nlocal = steps // size          # no. of tsteps each proc should handle
        start = (rank * nlocal) + (steps*slice) + 1
        end = ((rank + 1) * nlocal) + (steps*slice) + 1


        if rank == size - 1:
            nlocal += steps % size
            start = steps - nlocal + (steps*slice) + 1
            end = steps + (steps*slice) + 1

        chunksize = end - start

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

        # Define the fluid region
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
        # print(N_surfU, N_surfL)
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
            print('Average gap Height in the sampled time is {0:.3f}'.format(avg_gap_height))


        # Velocities ------------------------------
        vels = np.array(vels_data[start:end]).astype(np.float32)

        fluid_vx,fluid_vy,fluid_vz = vels[:, :Nf, 0], \
                                     vels[:, :Nf, 1], \
                                     vels[:, :Nf, 2]
        fluid_v = np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)/A_per_molecule

        fluid_vx_avg = np.mean(fluid_vx, axis=0)/A_per_molecule
        fluid_vy_avg = np.mean(fluid_vy, axis=0)/A_per_molecule

        # Unavgd velocities--------------------------------------
        if 'no-temp' not in sys.argv:
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
        # total_box_Height = avgsurfU_end_global - avgsurfL_begin_global
        bulkStartZ = (0.4 * avg_gap_height) + surfL_end_global #+ 1
        bulkEndZ = (0.6 * avg_gap_height) + surfL_end_global #+ 1
        # print(bulkStartZ, bulkEndZ)
        bulkHeight = bulkEndZ - bulkStartZ
        bulk_hi = np.less_equal(fluid_zcoords, bulkEndZ)
        bulk_lo = np.greater_equal(fluid_zcoords, bulkStartZ)
        bulk_region = np.logical_and(bulk_lo, bulk_hi)
        bulk_vol = bulkHeight * xlength * ylength
        bulk_N = np.sum(bulk_region, axis=1)

        # Virial Pressure in the fluid -------------------------
        if 'no-virial' not in sys.argv:
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
        if 'smooth' in sys.argv:
            stableStartX = 0.4 * xlength
            stableEndX = 0.6 * xlength
        stable_length = stableEndX - stableStartX
        stable_hi = np.less_equal(fluid_xcoords, stableEndX)
        stable_lo = np.greater_equal(fluid_xcoords, stableStartX)
        stable_region = np.logical_and(stable_lo, stable_hi)
        # USe the avg_gap_height instead of the inst gap height
        stable_vol = gap_height * stable_length * ylength
        stable_N = np.sum(stable_region, axis=1)

        # Get the mass flow rate and mass flux in the stable region
        vels_stable = fluid_vx * stable_region
        mflux_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * (mCH_avg/sci.N_A))/ (stable_vol * (ang_to_m)**3)      # g/m2.ns
        mflowrate_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * (mCH_avg/sci.N_A)) / (stable_length * ang_to_m)

        #Pump Region --------------------------------------
        pumpStartX = 0.0 * xlength
        pumpEndX = 0.2 * xlength
        if 'smooth' in sys.argv:
            pumpEndX = pumpEndX * 15/8
        pump_length = pumpEndX-pumpStartX
        pump_hi = np.less_equal(fluid_xcoords, pumpEndX)
        pump_lo = np.greater_equal(fluid_xcoords, pumpStartX)
        pump_region = np.logical_and(pump_lo, pump_hi)
        pump_vol = gap_height*pump_length*ylength
        pump_N = np.sum(pump_region, axis=1)

        # Measure mass flux in the pump region
        vels_pump = fluid_vx * pump_region
        mflux_pump = (np.sum(vels_pump, axis=1) * (ang_to_m/fs_to_ns) * (mCH_avg/sci.N_A)) / (pump_vol * (ang_to_m)**3)      # g/m2.ns
        mflowrate_pump = (np.sum(vels_pump, axis=1) * (ang_to_m/fs_to_ns) * (mCH_avg/sci.N_A)) / (pump_length * ang_to_m)       # g/ns

        # The Grid -------------------------------------------------------------
        dim = np.array([Nx, Ny, Nz])

        # Bounds Fluid
        bounds = [np.arange(dim[i] + 1) / dim[i] * lengths[i] + fluid_posmin[i]
                        for i in range(3)]

        xx, yy, zz, vol = utils.bounds(bounds[0], bounds[1], bounds[2])

        # Bounds surfU
        bounds_surfU = [bounds[0], np.array([0,ylength]), np.array([surfU_begin_global,surfU_end_global])]
        xx_surfU, yy_surfU, zz_surfU, vol_surfU = utils.bounds(bounds_surfU[0], bounds_surfU[1], bounds_surfU[2])

        # Bounds surfL
        bounds_surfL = [bounds[0], np.array([0,ylength]), np.array([surfL_begin_global,surfL_end_global])]
        xx_surfL, yy_surfL, zz_surfL, vol_surfL = utils.bounds(bounds_surfL[0], bounds_surfL[1], bounds_surfL[2])

        # Bounds Stable
        stableRange = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX
        bounds_stable = [stableRange, np.array([bounds[1]]), np.array([bounds[2]])]
        xx_stable, yy_stable, zz_stable, vol_stable = utils.bounds(bounds_stable[0], bounds_stable[1], bounds_stable[2])

        # Bounds Bulk
        bulkRange = np.arange(dim[2] + 1) / dim[2] * bulkHeight + bulkStartZ
        bounds_bulk = [np.array([bounds[0]]), np.array([bounds[1]]), bulkRange]
        xx_bulk, yy_bulk, zz_bulk, vol_bulk = utils.bounds(bounds_bulk[0], bounds_bulk[1], bounds_bulk[2])


        # Thermodynamic and Mechanical Quantities ---------------------------------
        # -------------------------------------------------------------------------

        # # local buffers
        N = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
        N_bulk = np.zeros_like(N)
        N_stable = np.zeros_like(N)

        v_t_ch = np.zeros_like(N)
        den_ch = np.zeros_like(N)
        vx_ch = np.zeros_like(N)

        jx_ch = np.zeros_like(N)
        if 'no-virial' not in sys.argv:
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
                cellx = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx[i, 0, k], xx[i+1, 0, k])['mask']

                cellz = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz[i, 0, k], zz[i, 0, k+1])['mask']

                cell = np.logical_and(cellx,cellz)

                # Count particles in the fluid cell
                N[:, i, k] = np.sum(cell, axis=1)
                Nzero = np.less(N[:, i, k], 1)
                N[Nzero, i, k] = 1

                # Stable --------------------------------------
                cellx_stable = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_stable[i, 0, k], xx_stable[i+1, 0, k])['mask']

                cellz_stable = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_stable[i, 0, k], zz_stable[i, 0, k+1])['mask']

                cell_stable = np.logical_and(cellx_stable,cellz_stable)

                N_stable[:, i, k] = np.sum(cell_stable, axis=1)
                Nzero_stable = np.less(N_stable[:, i, k], 1)
                N_stable[Nzero_stable, i, k] = 1

                # Bulk -----------------------------------
                cellx_bulk = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_bulk[i, 0, k], xx_bulk[i+1, 0, k])['mask']

                cellz_bulk = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_bulk[i, 0, k], zz_bulk[i, 0, k+1])['mask']

                cell_bulk = np.logical_and(cellx_bulk,cellz_bulk)
                # Count particles in the bulk
                N_bulk[:, i, k] = np.sum(cell_bulk, axis=1)

                # For Solid
                cellxU = utils.region(surfU_xcoords, surfU_xcoords,
                                        xx_surfU[i, 0, 0], xx_surfU[i+1, 0, 0])['mask']

                cellxL = utils.region(surfL_xcoords, surfL_xcoords,
                                        xx_surfL[i, 0, 0], xx_surfL[i+1, 0, 0])['mask']

                # print(np.sum(cellxU, axis=1))

                # Temperature (correct this)
                if 'no-temp' not in sys.argv:
                    v_t_ch[:, i, k] = np.sum((fluid_v_t**2) * cellx, axis=1) / (3*N[:, i, k]/A_per_molecule)

                # Density (bulk)
                den_ch_bulk[:, i] = (N_bulk[:, i, 0]/A_per_molecule) / vol_bulk[i, 0, 0]
                # den_ch_bulk[:, i] = (N[:, i, 0]/A_per_molecule) / vol[i, 0, 0]

                # Density (whole fluid)
                den_ch[:, i, k] = (N[:, i, k]/A_per_molecule) / vol[i, 0, k]

                # Vx (stable region)
                vx_ch[:, i, k] = np.sum(fluid_vx * cell_stable, axis=1) / N_stable[:, i, k]

                # Mass flux (whole fluid)
                jx_ch[:, i, k] = vx_ch[:, i, k] * den_ch[:, i, k]

                # Virial Pressure
                if 'no-virial' not in sys.argv:
                    W1_ch = np.sum(virial[:, :Nf, 0] * cell_bulk, axis=1)
                    W2_ch = np.sum(virial[:, :Nf, 1] * cell_bulk, axis=1)
                    W3_ch = np.sum(virial[:, :Nf, 2] * cell_bulk, axis=1)
                    vir_ch[:, i, k] = -(W1_ch + W2_ch + W3_ch)

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
            surfL_end_global = np.zeros_like(time)

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
        sendcounts_chunk = np.array(comm.gather(den_ch.size, root=0))
        sendcounts_chunkS = np.array(comm.gather(surfU_fx_ch.size, root=0))
        sendcounts_chunk_bulk = np.array(comm.gather(den_ch_bulk.size, root=0))

        comm.Gatherv(sendbuf=comZ, recvbuf=(comZ_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=gap_height, recvbuf=(gap_height_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=mflowrate_stable, recvbuf=(mflowrate_stable_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=mflux_stable, recvbuf=(mflux_stable_global, sendcounts_time), root=0)
        comm.Gatherv(sendbuf=mflux_pump, recvbuf=(mflux_pump_global, sendcounts_time), root=0)

        if 'no-temp' not in sys.argv:
            comm.Gatherv(sendbuf=v_t_ch, recvbuf=(v_t_global, sendcounts_chunk), root=0)

        comm.Gatherv(sendbuf=vx_ch, recvbuf=(vx_ch_global, sendcounts_chunk), root=0)

        vx_global = np.array(comm.gather(fluid_vx_avg, root=0))
        vy_global = np.array(comm.gather(fluid_vy_avg, root=0))

        comm.Gatherv(sendbuf=den_ch, recvbuf=(den_ch_global, sendcounts_chunk), root=0)
        comm.Gatherv(sendbuf=jx_ch, recvbuf=(jx_ch_global, sendcounts_chunk), root=0)
        if 'no-virial' not in sys.argv:
            comm.Gatherv(sendbuf=totVi, recvbuf=(vor_vol_global, sendcounts_time), root=0)
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
            if 'no-temp' not in sys.argv:
                temp = ((mf*g_to_kg/sci.N_A) * v_t_global * A_per_fs_to_m_per_s**2) / Kb       # Kelvin

            # Velocity
            vx_ch_global *=  A_per_fs_to_m_per_s       # m/s

            vx_global_avg = np.mean(vx_global, axis=0)
            vy_global_avg = np.mean(vy_global, axis=0)

            vx_global_avg *=  A_per_fs_to_m_per_s       # m/s
            vy_global_avg *=  A_per_fs_to_m_per_s       # m/s

            # Density
            den_ch_global *= (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
            den_ch_bulk_global *= (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3

            # Mass flux
            jx_ch_global *= (ang_to_m/fs_to_ns) * (mf/sci.N_A) / (ang_to_m**3)

            if 'no-virial' not in sys.argv:
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
            out.createDimension('Nf', Nf)
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

            if 'no-temp' not in sys.argv:
                temp_var =  out.createVariable('Temperature', 'f4', ('step', 'x', 'z'))

            vx_var =  out.createVariable('Vx', 'f4', ('step', 'x', 'z'))

            vy_all_var = out.createVariable('Fluid_Vy', 'f4', ('Nf'))
            vx_all_var = out.createVariable('Fluid_Vx', 'f4', ('Nf'))

            jx_var =  out.createVariable('Jx', 'f4', ('step', 'x', 'z'))
            if 'no-virial' not in sys.argv:
                vir_var = out.createVariable('Virial', 'f4', ('step', 'x', 'z'))
            den_var = out.createVariable('Density', 'f4',  ('step', 'x', 'z'))

            fx_U_var =  out.createVariable('Fx_Upper', 'f4',  ('step', 'x'))
            fy_U_var =  out.createVariable('Fy_Upper', 'f4',  ('step', 'x'))
            fz_U_var =  out.createVariable('Fz_Upper', 'f4',  ('step', 'x'))

            fx_L_var =  out.createVariable('Fx_Lower', 'f4',  ('step', 'x'))
            fy_L_var =  out.createVariable('Fy_Lower', 'f4',  ('step', 'x'))
            fz_L_var =  out.createVariable('Fz_Lower', 'f4',  ('step', 'x'))

            density_bulk_var =  out.createVariable('Density_Bulk', 'f4',  ('step', 'x'))

            time_var[:] = time
            com_var[:] = comZ_global
            gap_height_var[:] = gap_height_global
            voronoi_volumes[:] = vor_vol_global
            mflux_pump[:] = mflux_pump_global
            mflowrate_stable[:] = mflowrate_stable_global
            mflux_stable[:] = mflux_stable_global

            vx_all_var[:] = vx_global_avg
            vy_all_var[:] = vy_global_avg

            if 'no-temp' not in sys.argv:
                temp_var[:] = temp
            vx_var[:] = vx_ch_global
            jx_var[:] = jx_ch_global
            if 'no-virial' not in sys.argv:
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
