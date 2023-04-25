#!/usr/bin/env python

import netCDF4
import sys
import numpy as np
import scipy.constants as sci
import time as timer
import utils
# import tesellation as tes
import warnings
import logging
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

if 'lj' in sys.argv:
    mf, A_per_molecule = 39.948, 1
if 'propane' in sys.argv:
    mf, A_per_molecule = 44.09, 3
if 'pentane' in sys.argv:
    mf, A_per_molecule = 72.15, 5
if 'heptane' in sys.argv:
    mf, A_per_molecule = 100.21, 7


t0 = timer.time()


slice_start, slice_end = [], []

def proc(infile, Nx, Nz, slice_size, Ny=1):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    infile = comm.bcast(infile, root=0)
    Nx, Ny, Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)

    global data
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
        print('Total simualtion time: {} {}'.format(np.int(tSteps_tot * out_frequency * Time.scale_factor), Time.units))
        print('======> The dataset will be sliced to %g slices!! <======' %Nslices)

    cell_lengths = data.variables["cell_lengths"]
    cell_lengths = np.array(cell_lengths[0, :]).astype(np.float32)

    # Particle Types ------------------------------
    type = data.variables["type"]
    type = np.array(type[0, :]).astype(np.float32)

    global fluid_idx
    global solid_idx
    fluid_idx, solid_idx = [],[]
    # Lennard-Jones
    if np.max(type)==2:
        fluid_idx.append(np.where(type == 1))
        solid_idx.append(np.where(type == 2))

    # Hydrocarbons
    if np.max(type)==3:
        fluid_idx.append(np.where([type == 1, type == 2]))
        solid_idx.append(np.where(type == 3))

    fluid_idx = fluid_idx[0][1]
    solid_idx = solid_idx[0][0]

    global Nf
    Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/A_per_molecule, \
                            np.max(solid_idx)-np.max(fluid_idx)

    global solid_start
    solid_start = np.min(solid_idx)

    if rank == 0:
        print('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,int(Nm),Ns))

    # Coordinates ------------------------------

    # Forces on the walls
    forcesU = data.variables["f_fAtomU_avg"]
    forcesL = data.variables["f_fAtomL_avg"]
    # Velocities -----------------------------------
    if 'no-temp' not in sys.argv:
        vels_data_unavgd = data.variables["velocities"]     # Used for temp. calculation
        vels_data = data.variables["f_velocity"]
    # Virial ---------------------------------------
    if 'no-virial' not in sys.argv:
        voronoi_vol_data = data.variables["f_Vi_avg"]
        virial_data = data.variables["f_Wi_avg"]


    for slice in range(Nslices):

        global slice_start
        global slice_end
        slice_start.append(slice1)
        slice_end.append(slice2)

        # Exclude the first snapshot since data are not stored
        if tSteps_tot <= 1000:
            steps = Time[slice1:].shape[0]
            steps_array = np.array(Time[slice1:]).astype(np.float32)
        else:
            steps = slice_size
            steps_array = np.array(Time[slice1:slice2]).astype(np.float32)

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

        slice1 = slice2
        slice2 += slice_size

    slice_start.append(slice1)
    slice_end.append(slice2)


    return start, end, Nslices, slice1, slice2




start_chunk, end_chunk, Nslices_chunk, s1, s2 = proc(sys.argv[1],np.int(sys.argv[2]),np.int(sys.argv[3]),np.int(sys.argv[4]))

print( slice_start, slice_end)

    #---------------------------------------------------------
    # Coordinates -------------------------------------------
    #---------------------------------------------------------
    # Array dimensions: (time, idx, dimesnion)
def get_coords(start, end, Nslices):

    for i in range(Nslices):
        # coords_data_unavgd = data.variables["coordinates"]
        coords_data = data.variables["f_position"]
        coords = np.array(coords_data[start[i]:end[i]]).astype(np.float32)

        fluid_xcoords,fluid_ycoords,fluid_zcoords = coords[:, fluid_idx, 0], \
                                                    coords[:, fluid_idx, 1], \
                                                    coords[:, fluid_idx, 2]

        solid_xcoords, solid_zcoords = coords[:, solid_start:, 0], \
                                       coords[:, solid_start:, 2]

        # To avoid problems with logical-and later
        solid_xcoords[solid_xcoords==0] = 1e-5

        # Define the fluid region ---------------------------------------------

        # Instanteneous extrema (array Needed for the gap height at each timestep)
        min_fluidZ, max_fluidZ = utils.extrema(fluid_zcoords)['global_min'], \
                                 utils.extrema(fluid_zcoords)['global_max']
        # Average Extrema over all timesteps ( integer Needed for the avergage gap height)
        # avgmin_fluidZ = utils.cmin(fluid_zcoords)[2]
        # avgmax_fluidZ = utils.cmax(fluid_zcoords)[2]

        # Define the surfU and surfL regions  ---------------------------------
        surfL = np.ma.masked_less(solid_zcoords, avgmax_fluidZ/2.)
        surfU = np.ma.masked_greater(solid_zcoords, avgmax_fluidZ/2.)
        # Indices of the the lower and upper
        surfL_indices = np.where(surfL[0].mask)[0]
        surfU_indices = np.where(surfU[0].mask)[0]

        surfL_xcoords, surfL_zcoords = solid_xcoords[:,surfL_indices],\
                                       solid_zcoords[:,surfL_indices]
        surfU_xcoords, surfU_zcoords = solid_xcoords[:,surfU_indices], \
                                       solid_zcoords[:,surfU_indices]

        # Check if surfU and surfL are not equal
        N_surfL, N_surfU = len(surfL_indices), len(surfU_indices)
        if N_surfU != N_surfL and rank == 0:
            logger.warning("No. of surfU atoms != No. of surfL atoms")

        Lx, Ly = lengths[0], lengths[1]

    return fluid_xcoords



    a= get_coords(slice_start, slice_end, Nslices_chunk)
    print(a.shape)


        #
        # # The converged and diverged regions fluid, surfU and surfL coordinates
        # fluid_zcoords_div = utils.region(fluid_zcoords, fluid_xcoords, 0, 0.2*Lx)[2]
        # surfU_zcoords_div = utils.region(surfU_zcoords, surfU_xcoords, 0, 0.2*Lx)[2]
        # surfL_zcoords_div = utils.region(surfL_zcoords, surfL_xcoords, 0, 0.2*Lx)[2]
        # # Narrow range because of outliers in the fluid coordinates
        # fluid_zcoords_conv = utils.region(fluid_zcoords, fluid_xcoords, 0.49*Lx, 0.5*Lx)[2]
        # surfU_zcoords_conv = utils.region(surfU_zcoords, surfU_xcoords, 0.49*Lx, 0.5*Lx)[2]
        # surfL_zcoords_conv = utils.region(surfL_zcoords, surfL_xcoords, 0.49*Lx, 0.5*Lx)[2]
        #
        # # For the diverging part
        # avg_surfL_begin_div = utils.cmin(surfL_zcoords_div)[2]
        # avg_surfL_end_div = utils.cmax(surfL_zcoords_div)[2]
        # avg_surfU_begin_div = utils.cnonzero_min(surfU_zcoords_div)[2]
        # avg_surfU_end_div = utils.cmax(surfU_zcoords_div)[2]
        # avg_fluid_begin_div = utils.cnonzero_min(fluid_zcoords_div)[2]
        # avg_fluid_end_div = utils.cmax(fluid_zcoords_div)[2]
        #
        # # For the converging part
        # avg_surfL_end_conv = utils.cmax(surfL_zcoords_conv)[2]
        # avg_surfU_begin_conv = utils.cnonzero_min(surfU_zcoords_conv)[2]
        # avg_fluid_begin_conv = utils.cnonzero_min(fluid_zcoords_conv)[2]
        # avg_fluid_end_conv = utils.cmax(fluid_zcoords_conv)[2]
        #
        # # Average gap height(s)
        # avg_gap_height_conv = (avg_fluid_end_conv - avg_fluid_begin_conv \
        #                  + avg_surfU_begin_conv - avg_surfL_end_conv) / 2.
        #
        # avg_gap_height_div = (avg_fluid_end_div - avg_fluid_begin_div \
        #                  + avg_surfU_begin_div - avg_surfL_end_div) / 2.
        #
        # avg_gap_height = (avg_gap_height_conv + avg_gap_height_div)/2.
        #
        # if rank == 0 :
        #     print('Average gap Height (converged region) in the sampled time is {0:.3f}'.format(avg_gap_height_conv))
        #     print('Average gap Height (diverged region) in the sampled time is {0:.3f}'.format(avg_gap_height_div))
        #     print('Average gap Height in the sampled time is {0:.3f}'.format(avg_gap_height))
        #
        # # Instanteneous coordinates
        # surfU_begin_conv = utils.cnonzero_min(surfU_zcoords_conv)[0]
        # surfU_end_conv = utils.cmax(surfU_zcoords_conv)[0]
        # surfL_begin_conv = utils.cmin(surfL_zcoords_conv)[0]
        # surfL_end_conv = utils.cmax(surfL_zcoords_conv)[0]
        #
        # surfU_begin_div = utils.cnonzero_min(surfU_zcoords_div)[0]
        # surfU_end_div = utils.cmax(surfU_zcoords_div)[0]
        # surfL_begin_div = utils.cmin(surfL_zcoords_div)[0]
        # surfL_end_div = utils.cmax(surfL_zcoords_div)[0]
        #
        # # Gap height and COM (in Z-direction) at each timestep
        # gap_height = (max_fluidZ - min_fluidZ + surfU_begin_conv - surfL_end_conv) / 2.
        # # Center of mass in the z-direction
        # comZ = (surfU_end_conv - surfL_begin_conv) / 2.
        #
        # # Update the lengths z-component
        # fluid_posmin[2],fluid_posmax[2] = (avgmin_fluidZ + avg_surfL_end_div) / 2., \
        #                                   (avgmax_fluidZ + avg_surfU_begin_div)/2.
        # lengths[2] = fluid_posmax[2] - fluid_posmin[2]
        #
        #
        # # REGIONS within the fluid------------------------------------------------
        # #-------------------------------------------------------------------------
        #
        # # Pump --------------------------------------------
        # pumpStartX, pumpEndX = 0.4 * Lx, 0.6 * Lx
        # pump_length, pump_region, pump_xcoords, pump_vol, pump_N = \
        #     utils.region(fluid_xcoords, fluid_xcoords, pumpStartX, pumpEndX,
        #                                     Ly=Ly, length=avg_gap_height)
        #
        # # Bulk --------------------------------------------
        # bulkStartZ = (0.4 * avg_gap_height_conv) + avg_surfL_end_conv
        # bulkEndZ = (0.6 * avg_gap_height_conv) + avg_surfL_end_conv
        # bulk_height, bulk_region, bulk_zcoords, bulk_vol, bulk_N = \
        #     utils.region(fluid_zcoords, fluid_zcoords, bulkStartZ, bulkEndZ,
        #                                     Ly=Ly, length=Lx)
        #
        # # Stable ---------------------------------
        # stableStartX, stableEndX = 0.9 * Lx, 1 * Lx
        # stableStartX2, stableEndX2 = 0 * Lx, 0.1 * Lx
        #
        # stable_length, stable_region, stable_xcoords, stable_vol, stable_N = \
        #     utils.region(fluid_xcoords, fluid_xcoords, stableStartX, stableEndX,
        #                                     Ly=Ly, length=avg_gap_height_conv)
        #
        # # The Grid -------------------------------------------------------------
        # # ----------------------------------------------------------------------
        # dim = np.array([Nx, Ny, Nz])
        #
        # # Fluid --------------------------------------------
        # bounds_fluid = [np.arange(dim[i] + 1) / dim[i] * lengths[i] + fluid_posmin[i]
        #                 for i in range(3)]
        # xx_fluid, yy_fluid, zz_fluid, vol_fluid = utils.bounds(bounds_fluid[0], bounds_fluid[1], bounds_fluid[2])
        #
        # # Bulk --------------------------------------------
        # bulkRangeZ = np.arange(dim[2] + 1) / dim[2] * bulk_height + bulkStartZ
        # bounds_bulk = [np.array([bounds_fluid[0]]), np.array([bounds_fluid[1]]), bulkRangeZ]
        # xx_bulk, yy_bulk, zz_bulk, vol_bulk = utils.bounds(bounds_bulk[0], bounds_bulk[1], bounds_bulk[2])
        #
        # # surfU --------------------------------------------
        # bounds_surfU = [bounds_fluid[0], bounds_fluid[1], np.array([avg_surfU_begin_div, avg_surfU_end_div])]
        # xx_surfU, yy_surfU, zz_surfU, vol_surfU = utils.bounds(bounds_surfU[0], bounds_surfU[1], bounds_surfU[2])
        #
        # # surfL --------------------------------------------
        # bounds_surfL = [bounds_fluid[0], bounds_fluid[1], np.array([avg_surfL_begin_div, avg_surfL_end_div])]
        # xx_surfL, yy_surfL, zz_surfL, vol_surfL = utils.bounds(bounds_surfL[0], bounds_surfL[1], bounds_surfL[2])
        #
        # # Stable --------------------------------------------
        # stableRangeX = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX
        # bounds_stable = [stableRangeX, np.array([bounds_fluid[1]]), np.array([bounds_fluid[2]])]
        # xx_stable, yy_sytable, zz_stable, vol_stable = utils.bounds(bounds_stable[0], bounds_stable[1], bounds_stable[2])
        #
        #
        # #---------------------------------------------------------
        # # Velocities --------------------------------------------
        # #---------------------------------------------------------
        # vels = np.array(vels_data[start:end]).astype(np.float32)
        #
        # fluid_vx,fluid_vy,fluid_vz = vels[:, :Nf, 0], \
        #                              vels[:, :Nf, 1], \
        #                              vels[:, :Nf, 2]
        #
        # fluid_vx_avg = np.mean(fluid_vx, axis=0) / A_per_molecule
        # fluid_vy_avg = np.mean(fluid_vy, axis=0) / A_per_molecule
        # fluid_v = np.sqrt(fluid_vx**2 + fluid_vy**2 + fluid_vz**2) / A_per_molecule
        #
        # # Vx in the stable region
        # vx_ch = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
        #
        # # The unaveraged velocities
        # if 'no-temp' not in sys.argv:
        #     vels_t = np.array(vels_data_unavgd[start:end]).astype(np.float32)
        #
        #     fluid_vx_t,fluid_vy_t,fluid_vz_t = vels_t[:, :Nf, 0], \
        #                                        vels_t[:, :Nf, 1], \
        #                                        vels_t[:, :Nf, 2]
        #
        #     fluid_v_t = np.sqrt(fluid_vx_t**2 + fluid_vy_t**2 + fluid_vz_t**2) / A_per_molecule
        #
        #     # Velocity in the fluid
        #     v_t_ch = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
        #
        # #---------------------------------------------------------
        # # Fluid Virial Pressure ----------------------------------
        # #---------------------------------------------------------
        # if 'no-virial' not in sys.argv:
        #     voronoi_vol = np.array(voronoi_vol_data[start:end]).astype(np.float32)
        #     virial = np.array(virial_data[start:end]).astype(np.float32)
        #
        #     Vi = voronoi_vol[:, :Nf] * bulk_region
        #     totVi = np.sum(Vi, axis=1)
        #
        #     vir_ch = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
        #
        #
        # #---------------------------------------------------------
        # # Wall Stresses -------------------------------------------------
        # #---------------------------------------------------------
        # forcesU_data = np.array(forcesU[start:end]).astype(np.float32)
        # forcesL_data = np.array(forcesL[start:end]).astype(np.float32)
        #
        # surfU_fx,surfU_fy,surfU_fz = forcesU_data[:, solid_start:, 0][:,surfU_indices], \
        #                              forcesU_data[:, solid_start:, 1][:,surfU_indices], \
        #                              forcesU_data[:, solid_start:, 2][:,surfU_indices]
        #
        # surfL_fx,surfL_fy,surfL_fz = forcesL_data[:, solid_start:, 0][:,surfL_indices], \
        #                              forcesL_data[:, solid_start:, 1][:,surfL_indices], \
        #                              forcesL_data[:, solid_start:, 2][:,surfL_indices]
        #
        # surfU_fx_ch = np.zeros([chunksize, Nx], dtype=np.float32)
        # surfU_fy_ch = np.zeros_like(surfU_fx_ch)
        # surfU_fz_ch = np.zeros_like(surfU_fx_ch)
        # surfL_fx_ch = np.zeros_like(surfU_fx_ch)
        # surfL_fy_ch = np.zeros_like(surfU_fx_ch)
        # surfL_fz_ch = np.zeros_like(surfU_fx_ch)
        #
        # # # Correction for the cell volume ---------------------------------------
        # # avg_gap_height_cell = np.zeros([Nx, Nz], dtype=np.float32)
        # # avg_surfU_begin_cell = np.zeros_like(avg_gap_height_cell)
        # # avg_surfL_end_cell = np.zeros_like(avg_gap_height_cell)
        # # avgmax_fluidZ_cell = np.zeros_like(avg_gap_height_cell)
        # # avgmin_fluidZ_cell = np.zeros_like(avg_gap_height_cell)
        # # zlo = np.zeros_like(avg_gap_height_cell)
        # # zhi = np.zeros_like(avg_gap_height_cell)
        # #
        # #
        # # surfU_zcoords_ch = np.zeros([chunksize, N_surfU], dtype=np.float32)
        # # surfL_zcoords_ch = np.zeros([chunksize, N_surfL], dtype=np.float32)
        # #
        # # # create a mask to filter only particles in the grid cell
        # # # Solid -----------------------------------------
        # #
        # # cellzU, N_surfU = utils.cell1B(Nx, surfU_zcoords, surfU_xcoords,
        # #                            xx_surfU, chunksize)
        # #
        # # cellzL, N_surfL = utils.cell1B(Nx, surfL_zcoords, surfL_xcoords,
        # #                            xx_surfL, chunksize)
        # #
        # # # Fluid -----------------------------------------
        # # cell_fluid, N_fluid = utils.cell2B(Nx, Nz, fluid_xcoords, fluid_zcoords,
        # #                            xx_fluid, zz_fluid, chunksize)
        # #
        # # print(surfU_zcoords_ch.shape)
        # # print(cellzU.shape)
        # # print(surfU_zcoords.shape)
        # #
        # # for i in range(Nx):
        # #     surfU_zcoords_ch[:,i] = cellzU * surfU_zcoords
        # #
        # # avg_surfU_begin_cell[:,:] = utils.cnonzero_min(cellzU)[2]
        # # avg_surfL_end_cell[:,:] = utils.cmax(cellzL)[2]
        # # avgmin_fluidZ_cell[:,:] = utils.cnonzero_min(cell_fluid)[2]
        # # avgmax_fluidZ_cell[:,:] = utils.cmax(cell_fluid)[2]
        # #
        # # print(avg_surfU_begin_cell)
        # #
        # # zlo[:,:] = (avgmin_fluidZ_cell[:,:] + avg_surfL_end_cell[:,:]) / 2.
        # # zhi[:,:] = (avgmax_fluidZ_cell[:,:] + avg_surfU_begin_cell[:,:]) / 2.
        # #
        # # avg_gap_height_cell[:,:] = zhi[:,:] - zlo[:,:]
        # #
        # # print(avg_gap_height_cell)
        # #
        # # # Update the z-bounds to get the right chunk volume
        # # if Nx > 1:
        # #     # # TODO: For now we take the last chunk and repeat it twice to
        # #     # get the dimensionality right but shouldn't use this later
        # #     zlo = np.append(zlo, zlo[-1])
        # #     zhi = np.append(zhi, zhi[-1])
        # #     zz = np.array([zlo, zhi])               # Array of min and max z in the region
        # #     zz = np.expand_dims(zz, axis=2)
        # #     zz = np.transpose(zz)                   # Region min and max are in one line
        # #     zz = np.concatenate((zz,zz), axis=0)    # Shape it like in the mesh
        # #     zz = np.transpose(zz, (1, 0, 2))
        # #     # Update the z-increment and the volume
        # #     dx = xx_fluid[1:, 1:, 1:] - xx_fluid[:-1, :-1, :-1]
        # #     dy = yy_fluid[1:, 1:, 1:] - yy_fluid[:-1, :-1, :-1]
        # #     dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]
        # #     vol_fluid = dx * dy * dz
        # #
        # # print(zz)
        #
        #
        # # Spatial bins in the regions
        # # mask: for every chunk array with dimensions: 0- time, 1- coords
        # # N: for every chunk array with dimensions:  0- time
        #
        # N_fluid_mask = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
        # N_stable_mask = np.zeros_like(N_fluid_mask)
        # N_bulk_mask = np.zeros_like(N_fluid_mask)
        # den_ch = np.zeros_like(N_fluid_mask)
        # jx_ch = np.zeros_like(N_fluid_mask)
        #
        # N_U_mask = np.zeros([chunksize, Nx], dtype=np.float32)
        # N_L_mask = np.zeros_like(N_U_mask)
        # den_ch_bulk = np.zeros_like(N_U_mask)
        #
        #
        # for i in range(Nx):
        #     for k in range(Nz):
        #
        # # -------------------------------------------------------
        # # Cell Definition ---------------------------------------
        # # -------------------------------------------------------
        #
        # # Fluid -----------------------------------------
        #         maskx_fluid = utils.region(fluid_xcoords, fluid_xcoords,
        #                                 xx_fluid[i, 0, k], xx_fluid[i+1, 0, k])[1]
        #         maskz_fluid = utils.region(fluid_zcoords, fluid_zcoords,
        #                                 zz_fluid[i, 0, k], zz_fluid[i, 0, k+1])[1]
        #         mask_fluid = np.logical_and(maskx_fluid, maskz_fluid)
        #
        #         # Count particles in the fluid cell
        #         N_fluid_mask[:, i, k] = np.sum(mask_fluid, axis=1)
        #         # Avoid having zero particles in the cell
        #         Nzero_fluid = np.less(N_fluid_mask[:, i, k], 1)
        #         N_fluid_mask[Nzero_fluid, i, k] = 1
        #
        # # Stable --------------------------------------
        #         maskx_stable = utils.region(fluid_xcoords, fluid_xcoords,
        #                             xx_stable[i, 0, k], xx_stable[i+1, 0, k])[1]
        #         maskz_stable = utils.region(fluid_zcoords, fluid_zcoords,
        #                             zz_stable[i, 0, k], zz_stable[i, 0, k+1])[1]
        #         mask_stable = np.logical_and(maskx_stable, maskz_stable)
        #
        #         # Count particles in the stable cell
        #         N_stable_mask[:, i, k] = np.sum(mask_stable, axis=1)
        #         # Avoid having zero particles in the cell
        #         Nzero_stable = np.less(N_stable_mask[:, i, k], 1)
        #         N_stable_mask[Nzero_stable, i, k] = 1
        #
        # # Bulk -----------------------------------
        #         maskx_bulk = utils.region(fluid_xcoords, fluid_xcoords,
        #                                 xx_bulk[i, 0, k], xx_bulk[i+1, 0, k])[1]
        #         maskz_bulk = utils.region(fluid_zcoords, fluid_zcoords,
        #                                 zz_bulk[i, 0, k], zz_bulk[i, 0, k+1])[1]
        #         mask_bulk = np.logical_and(maskx_bulk, maskz_bulk)
        #
        #         # Count particles in the bulk cell
        #         N_bulk_mask[:, i, k] = np.sum(mask_bulk, axis=1)
        #         # Avoid having zero particles in the cell
        #         Nzero_bulk = np.less(N_bulk_mask[:, i, k], 1)
        #         N_bulk_mask[Nzero_bulk, i, k] = 1
        #
        # # SurfU -----------------------------------
        #         maskxU = utils.region(surfU_xcoords, surfU_xcoords,
        #                                 xx_surfU[i, 0, 0], xx_surfU[i+1, 0, 0])[1]
        #         N_U_mask = np.sum(maskxU[:, i])
        #
        # # SurfL -----------------------------------
        #         maskxL = utils.region(surfL_xcoords, surfL_xcoords,
        #                                 xx_surfL[i, 0, 0], xx_surfL[i+1, 0, 0])[1]
        #         N_L_mask = np.sum(maskxL[:, i])
        #
        # # -----------------------------------------------------
        # # Cell Averages ---------------------------------------
        # # -----------------------------------------------------
        #
        #         # Velocities in the stable region
        #         vx_ch[:, i, k] = np.sum(fluid_vx * mask_stable, axis=1) / N_stable_mask[:, i, k]
        #
        #         # Unavgd velocities--------------------------------------
        #         if 'no-temp' not in sys.argv:
        #             v_t_ch[:, i, k] = np.sum((fluid_v_t**2) * mask_fluid, axis=1) / \
        #                                            (3 * N_fluid_mask[:, i, k]/A_per_molecule)
        #
        #         # Virial pressure--------------------------------------
        #         if 'no-virial' not in sys.argv:
        #             W1_ch = np.sum(virial[:, :Nf, 0] * mask_bulk, axis=1)
        #             W2_ch = np.sum(virial[:, :Nf, 1] * mask_bulk, axis=1)
        #             W3_ch = np.sum(virial[:, :Nf, 2] * mask_bulk, axis=1)
        #             vir_ch[:, i, k] = -(W1_ch + W2_ch + W3_ch) #/ vol[i,0,k]
        #
        #         # Wall stresses--------------------------------------
        #         surfU_fx_ch[:, i] = np.sum(surfU_fx * maskxU, axis=1)
        #         surfU_fy_ch[:, i] = np.sum(surfU_fy * maskxU, axis=1)
        #         surfU_fz_ch[:, i] = np.sum(surfU_fz * maskxU, axis=1)
        #         surfL_fx_ch[:, i] = np.sum(surfL_fx * maskxL, axis=1)
        #         surfL_fy_ch[:, i] = np.sum(surfL_fy * maskxL, axis=1)
        #         surfL_fz_ch[:, i] = np.sum(surfL_fz * maskxL, axis=1)
        #
        #         # Mass flux in the pump region ---------------------
        #         vels_pump = (fluid_vx/A_per_molecule) * pump_region
        #         mflux_pump = (np.sum(vels_pump, axis=1) * (ang_to_m/fs_to_ns) * \
        #                     (mf/sci.N_A)) / (pump_vol * (ang_to_m)**3)      # g/m2.ns
        #
        #         # Mass flux in the stable region -------------------
        #         vels_stable = (fluid_vx/A_per_molecule) * stable_region
        #         mflux_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * \
        #                     (mf/sci.N_A))/ (stable_vol * (ang_to_m)**3)      # g/m2.ns
        #         mflowrate_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * \
        #                     (mf/sci.N_A)) / (stable_length * ang_to_m)
        #
        #         # Density (bulk) -----------------------------------
        #         den_ch_bulk[:, i] = (N_bulk_mask[:, i, 0] / A_per_molecule) / vol_bulk[i, 0, 0]
        #
        #         # Density (whole fluid) ----------------------------
        #         den_ch[:, i, k] = (N_fluid_mask[:, i, k] / A_per_molecule) / vol_fluid[i, 0, k]
        #
        #         # Mass flux (whole fluid)
        #         jx_ch[:, i, k] = vx_ch[:, i, k] * den_ch[:, i, k]
        #
        #
        # if rank == 0:
        #     t1 = timer.time()
        #     print(' ======> Slice: {}, Time Elapsed: {} <======='.format(slice+1, t1-t0))

        # slice1 = slice2
        # slice2 += slice_size


# if __name__ == '__main__':
#     proc(sys.argv[1],np.int(sys.argv[2]),np.int(sys.argv[3]),np.int(sys.argv[4]))
