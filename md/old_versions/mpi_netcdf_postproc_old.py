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

t0 = timer.time()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.set_printoptions(threshold=sys.maxsize)

Kb = 1.38064852e-23 # m2 kg s-2 K-1
ang_to_cm = 1e-8
ang_to_m = 1e-10
fs_to_ns = 1e-6
A_per_fs_to_m_per_s = 1e5
kcalpermolA_to_N = 6.947694845598684e-11
atm_to_mpa = 0.101325
g_to_kg = 1e-3

def proc(infile, Nx, Nz, Ny=1):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    infile = comm.bcast(infile, root=0)
    Nx = comm.bcast(Nx, root=0)
    Ny = comm.bcast(Ny, root=0)
    Nz = comm.bcast(Nz, root=0)

    data = netCDF4.Dataset(infile)

    steps = data.variables["time"].shape[0]

    # Chunk the data: each processor has a time chunk of the data
    nlocal = steps // size          # no. of tsteps each proc should handle
    start = rank * nlocal
    end = (rank + 1) * nlocal

    chunksize = end - start

    if rank == size - 1:
        nlocal += steps % size
        start = steps - nlocal
        end = steps

    # Ignore the first snapshot of the trajectory
    if start == 0:
        start += 1

    chunksize = end - start
    # print(start,end)

    # Variables
    # for varobj in data.variables.keys():
    #     print(varobj)

    if 'lj' in sys.argv:
        mf, A_per_molecule = 39.948, 1
    elif 'propane' in sys.argv:
        mf, A_per_molecule = 44.09, 3
    elif 'pentane' in sys.argv:
        mf, A_per_molecule = 72.15, 5
    elif 'heptane' in sys.argv:
        mf, A_per_molecule = 100.21, 7

    # Time (timesteps*timestep_size)
    t = data.variables["time"]
    sim_time = np.array(t[start:end]).astype(np.float32)
    time = comm.gather(sim_time, root=0)

    if rank == 0:
        print('Total simualtion time: {} {}'.format(int(time[-1][-1]),t.units))

    thermo_freq = 1000
    # tSteps = int((sim_time[-1]-sim_time[0])/(scale*thermo_freq))
    # tSteps = int((sim_time[-1]-sim_time[0])/(t.scale_factor*thermo_freq))
    # tSample = tSteps+1
    # tSteps_array = [i for i in range (tSteps+1)]

    # EXTRACT DATA  ------------------------------------------
    #---------------------------------------------------------

    #cell_lengths = np.array(cell_lengths[start:end]).astype(np.float32)
    cell_lengths = np.array(data.variables["cell_lengths"])[0, :]

    # Particle Types ------------------------------
    type = data.variables["type"]
    type = np.array(type[0, :]).astype(np.float32)

    fluid_index,solid_index=[],[]

    for idx,val in enumerate(type):
        # Lennard-Jones
        if np.max(type)==2:
            if val == 1:    #Fluid
                fluid_index.append(idx)
            elif val == 2:  #Solid
                solid_index.append(idx)

        # Hydrocarbons
        elif np.max(type)==3:
            if val == 1 or val == 2:    #Fluid
                fluid_index.append(idx)
            elif val == 3:  #Solid
                solid_index.append(idx)

    Nf,Nm,Ns=len(fluid_index),np.int(len(fluid_index)/A_per_molecule),len(solid_index)

    if rank == 0:
        print('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,Nm,Ns))

    # Coordinates ------------------------------

    coords = data.variables["coordinates"]
    # coords = data.variables["f_position"]
    coords = np.array(coords[start:end]).astype(np.float32)

    # Array dimensions:
        # dim 0: time       dim 1: index           dim 2: dimesnion
    fluid_coords=coords[:, :np.max(fluid_index)+1, :]

    fluid_xcoords,fluid_ycoords,fluid_zcoords=coords[:, :np.max(fluid_index)+1, 0], \
                                              coords[:, :np.max(fluid_index)+1, 1], \
                                              coords[:, :np.max(fluid_index)+1, 2]

    # Nevery = 10                  # Time average every 10 tsteps (as was used in LAMMPS)
    # a = sq.block_ND_arr(tSample,fluid_xcoords,Nf,Nevery)
    # b = np.mean(a,axis=1)

    # Extrema of fluidZ coord in each timestep / dim: (tSteps)
    max_fluidX, min_fluidX = np.asarray(np.max(fluid_xcoords,axis=1)), np.asarray(np.min(fluid_xcoords,axis=1))
    max_fluidY, min_fluidY = np.asarray(np.max(fluid_ycoords,axis=1)), np.asarray(np.min(fluid_ycoords,axis=1))
    max_fluidZ , min_fluidZ = np.max(fluid_zcoords,axis=1), np.min(fluid_zcoords,axis=1)

    # Time Average of extrema fluidZ coord / dim: () (Avg. in each tchunk or each processor)
    avgmax_fluid, avgmin_fluid = np.mean(max_fluidZ), np.mean(min_fluidZ)

    solid_xcoords = coords[:, np.min(solid_index):, 0]
    solid_zcoords = coords[:, np.min(solid_index):, 2]

    # PBC correction
        # Z-axis is non-periodic
    nonperiodic=(2,)
    pdim = [n for n in range(3) if not(n in nonperiodic)]
    # print(pdim)
    hi = np.greater(fluid_coords[:, :, pdim], cell_lengths[pdim]).astype(int)
    lo = np.less(fluid_coords[:, :, pdim], 0.).astype(int)
    # print(lo.shape,hi.shape,fluid_coords[:, :, pdim].shape)
    fluid_coords[:, :, pdim] += (lo - hi) * cell_lengths[pdim]

    min_fluidX = comm.allreduce(np.min(min_fluidX), op=MPI.MIN)
    max_fluidX = comm.allreduce(np.max(max_fluidX), op=MPI.MAX)
    min_fluidY = comm.allreduce(np.min(min_fluidY), op=MPI.MIN)
    max_fluidY = comm.allreduce(np.max(max_fluidY), op=MPI.MAX)

    fluid_posmin = np.array([min_fluidX, min_fluidY,(avgmin_fluid)])
    fluid_posmax = np.array([max_fluidX, max_fluidY,(avgmax_fluid)])
    lengths = fluid_posmax - fluid_posmin

    xlength,ylength = lengths[0],lengths[1]

    # Velocities ------------------------------

    vels = data.variables["velocities"]
    # vels = data.variables["f_velocity"]
    vels = np.array(vels[start:end]).astype(np.float32)

    fluid_vx,fluid_vy,fluid_vz = vels[:, :np.max(fluid_index)+1, 0], \
                                 vels[:, :np.max(fluid_index)+1, 1], \
                                 vels[:, :np.max(fluid_index)+1, 2]
    fluid_v = np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)

    # Forces ------------------------------

    # forces = data.variables["forces"]
    # forces = data.variables["f_force"]
    # forces = np.array(forces[start:end]).astype(np.float32)
    #
    # fluid_fx,fluid_fy,fluid_fz = forces[:, :np.max(fluid_index)+1, 0], \
    #                              forces[:, :np.max(fluid_index)+1, 1], \
    #                              forces[:, :np.max(fluid_index)+1, 2]
    # solid_fx,solid_fy,solid_fz = forces[:, np.min(solid_index):, 0], \
    #                              forces[:, np.min(solid_index):, 1], \
    #                              forces[:, np.min(solid_index):, 2]

    # Temperature --------------------------------------
    vels_t = data.variables["velocities"]
    vels_t = np.array(vels_t[start:end]).astype(np.float32)

    fluid_vx_t,fluid_vy_t,fluid_vz_t = vels_t[:, :np.max(fluid_index)+1, 0], \
                                       vels_t[:, :np.max(fluid_index)+1, 1], \
                                       vels_t[:, :np.max(fluid_index)+1, 2]

    fluid_v_t = np.sqrt(fluid_vx_t**2+fluid_vy_t**2+fluid_vz_t**2)/A_per_molecule

    # Stresses in the wall ------------------------------
    forcesU = data.variables["f_fAtomU_avg"]
    forcesU = np.array(forcesU)[start:end].astype(np.float32)
    forcesL = data.variables["f_fAtomL_avg"]
    forcesL = np.array(forcesL)[start:end].astype(np.float32)

    surfU_fx,surfU_fy,surfU_fz = forcesU[:, np.min(solid_index):, 0][:,:N_surfU], \
                                 forcesU[:, np.min(solid_index):, 1][:,:N_surfU], \
                                 forcesU[:, np.min(solid_index):, 2][:,:N_surfU]

    surfL_fx,surfL_fy,surfL_fz = forcesL[:, np.min(solid_index):, 0][:,N_surfU:], \
                                 forcesL[:, np.min(solid_index):, 1][:,N_surfU:], \
                                 forcesL[:, np.min(solid_index):, 2][:,N_surfU:]

    # Virial Pressure in the fluid -------------------------
    # voronoi_vol = data.variables["f_Vi_avg"]
    # voronoi_vol = np.array(voronoi_vol[start:end]).astype(np.float32)
    #
    # Vi = voronoi_vol[: , :np.max(fluid_index)+1] * bulk_region
    # totVi = np.sum(Vi,axis=1)
    #
    # virial = data.variables["f_Wi_avg"]
    # virial = np.array(virial[start:end]).astype(np.float32)

    # REGIONS ------------------------------------------------
    #---------------------------------------------------------

    # SurfU and SurfL Regions
        # SurfU and SurfL groups definitions (lower than half the COM of the fluid in
        # the first tstep is assigned to surfL otherwise it is a surfU atom)

    ## TODO: REPLACE THIS LATER

    # comm.Reduce(np.array(data.variables["f_position"])[0],fluid_COM_global, MPI.MAX, root=0)
    # surfL=np.less_equal(solid_zcoords,np.max(fluid_zcoords[0]/2.))
    # surfU=np.greater_equal(solid_zcoords,np.max(fluid_zcoords[0]/2.))

    surfL=np.less_equal(solid_zcoords,27.539106)
    surfU=np.greater_equal(solid_zcoords,27.539106)

    N_surfL,N_surfU = np.sum(surfL[0]),np.sum(surfU[0])

    if N_surfU != N_surfL:
        logger.warning("No. of surfU atoms != No. of surfL atoms")

    surfL_xcoords,surfU_xcoords = solid_xcoords*surfL , solid_xcoords*surfU
    surfL_zcoords,surfU_zcoords = solid_zcoords*surfL , solid_zcoords*surfU

    surfU_begin,surfU_end = np.min(surfU_zcoords[:,:N_surfU],axis=1), \
                            np.max(surfU_zcoords[:,:N_surfU],axis=1)

    surfL_begin,surfL_end = np.min(surfL_zcoords[:,N_surfU:],axis=1), \
                            np.max(surfL_zcoords[:,N_surfU:],axis=1)

    avgsurfU_begin,avgsurfL_end = np.mean(surfU_begin), np.mean(surfL_end)

    gap_height =(max_fluidZ-min_fluidZ+surfU_begin-surfL_end)/2.

    comZ = (surfU_end-surfL_begin)/2.

    # Update the lengths z-component
    fluid_posmin[2],fluid_posmax[2] = (avgmin_fluid+avgsurfL_end)/2., (avgmax_fluid+avgsurfU_begin)/2.
    lengths[2] = fluid_posmax[2] - fluid_posmin[2]

    avg_gap_height = (-avgmin_fluid-avgsurfL_end+avgmax_fluid+avgsurfU_begin)/2.# lengths[2]

    chunked_heights = np.array(comm.gather(avg_gap_height, root=0))

    if rank == 0:
        avg_gap_height = np.mean(chunked_heights)
        print('Average gap Height in the sampling time is {0:.3f}'.format(avg_gap_height))

    # Bulk Region ----------------------------------
    total_box_Height=np.mean(surfU_end)-np.mean(surfL_begin)
    bulkStartZ=0.25*total_box_Height
    bulkEndZ=0.75*total_box_Height
    bulkHeight=bulkEndZ-bulkStartZ
    bulk_hi = np.less_equal(fluid_zcoords, bulkEndZ)
    bulk_lo = np.greater_equal(fluid_zcoords, bulkStartZ)
    bulk_region = np.logical_and(bulk_lo, bulk_hi)
    bulk_vol=bulkHeight*xlength*ylength
    # print(bulk_vol)
    bulk_N = np.sum(bulk_region, axis=1)

    #Stable Region ---------------------------------
    stableStartX=0.4*xlength
    stableEndX=0.8*xlength
    stable_length=stableEndX-stableStartX
    stable_hi = np.less_equal(fluid_xcoords, stableEndX)
    stable_lo = np.greater_equal(fluid_xcoords, stableStartX)
    stable_region = np.logical_and(stable_lo, stable_hi)
    stable_vol=gap_height*stable_length*ylength
    stable_N = np.sum(stable_region, axis=1)

    #Pump Region --------------------------------------
    pumpStartX=0.0*xlength
    pumpEndX=0.2*xlength
    pump_length=pumpEndX-pumpStartX
    pump_hi = np.less_equal(fluid_xcoords, pumpEndX)
    pump_lo = np.greater_equal(fluid_xcoords, pumpStartX)
    pump_region = np.logical_and(pump_lo, pump_hi)
    pump_vol=gap_height*pump_length*ylength
    pump_N = np.sum(pump_region, axis=1)

    # The Grid
    dim = np.array([Nx, Ny, Nz])

    # Bounds Fluid
    bounds = [np.arange(dim[i] + 1) / dim[i] * lengths[i] + fluid_posmin[i]
                    for i in range(3)]

    # print(bounds[0])

    xx, yy, zz = np.meshgrid(bounds[0], bounds[1], bounds[2])

    xx = np.transpose(xx, (1, 0, 2))
    yy = np.transpose(yy, (1, 0, 2))
    zz = np.transpose(zz, (1, 0, 2))

    dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
    dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
    dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]

    vol = dx * dy * dz

    # Bounds Stable
    stableRange = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX

    bounds_stable = [stableRange, np.array([bounds[1]]), np.array([bounds[2]])]
    xx_stable, yy_stable, zz_stable = np.meshgrid(bounds_stable[0], bounds_stable[1], bounds_stable[2])
    xx_stable = np.transpose(xx_stable, (1, 0, 2))
    yy_stable = np.transpose(yy_stable, (1, 0, 2))
    zz_stable = np.transpose(zz_stable, (1, 0, 2))
    dx_stable = xx_stable[1:, 1:, 1:] - xx_stable[:-1, :-1, :-1]
    dy_stable = yy_stable[1:, 1:, 1:] - yy_stable[:-1, :-1, :-1]
    dz_stable = zz_stable[1:, 1:, 1:] - zz_stable[:-1, :-1, :-1]
    vol_stable = dx_stable * dy_stable * dz_stable

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

    # Thermodynamic and Mechanical Quantities ---------------------------------
    # -------------------------------------------------------------------------

    # # local buffers
    N = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
    N_bulk = np.zeros_like(N)
    v_t_ch = np.zeros_like(N)
    den_ch = np.zeros_like(N)
    vx_ch = np.zeros_like(N)
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

            Nzero = np.less(N[:, i, k], 1)
            N[Nzero, i, k] = 1

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

            # Mass flux (whole fluid)
            jx_ch[:, i, k] = vx_ch[:, i, k] * den_ch[:, i, k]

            # Virial Pressure
            # W1_ch = np.sum(virial[:,:np.max(fluid_index)+1, 0] * cell_bulk, axis=1)
            # W2_ch = np.sum(virial[:,:np.max(fluid_index)+1, 1] * cell_bulk, axis=1)
            # W3_ch = np.sum(virial[:,:np.max(fluid_index)+1, 2] * cell_bulk, axis=1)
            # vir_ch[:, i, k] = -(W1_ch + W2_ch + W3_ch)

            # For Solid
            xloS = np.less(surfU_xcoords[:,:N_surfU], xx[i+1, 0, 0])
            xhiS = np.greater_equal(surfU_xcoords[:,:N_surfU], xx[i, 0, 0])
            cellxS = np.logical_and(xloS, xhiS)

            # Wall stresses
            surfU_fx_ch[:, i] = np.sum(surfU_fx * cellxS, axis=1)
            surfU_fy_ch[:, i] = np.sum(surfU_fy * cellxS, axis=1)
            surfU_fz_ch[:, i] = np.sum(surfU_fz * cellxS, axis=1)
            surfL_fx_ch[:, i] = np.sum(surfL_fx * cellxS, axis=1)
            surfL_fy_ch[:, i] = np.sum(surfL_fy * cellxS, axis=1)
            surfL_fz_ch[:, i] = np.sum(surfL_fz * cellxS, axis=1)


    # # Temperature of the fluid region-------------------------------

    vels_stable = (fluid_vx/A_per_molecule) * stable_region
    mflux_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * (mf/sci.N_A))/ (stable_vol * (ang_to_m)**3)      # g/m2.ns
    mflowrate_stable = (np.sum(vels_stable, axis=1) * (ang_to_m/fs_to_ns) * (mf/sci.N_A)) / (stable_length * ang_to_m)

    # Measure mass flux in the pump region
    vels_pump = (fluid_vx/A_per_molecule) * pump_region
    mflux_pump = (np.sum(vels_pump, axis=1) * (ang_to_m/fs_to_ns) * (mf/sci.N_A)) / (pump_vol * (ang_to_m)**3)      # g/m2.ns

    if rank==0:

        Lx, Ly, h = xlength, ylength, avg_gap_height
        # print(N.shape)
        time = np.array(data.variables["time"])[1:]
        # Center of Mass in Z
        comZ_global = np.zeros((steps-1), dtype=np.float32)
        # Gap Height
        gap_height_global = np.zeros_like(comZ_global)
        # Flow Rate and Mass flux (In time)
        mflowrate_stable_global = np.zeros_like(comZ_global)
        mflux_stable_global = np.zeros_like(comZ_global)
        # Voronoi vol
        # vor_vol_global = np.zeros_like(comZ_global)

        # Surface Forces
        surfU_fx_ch_global = np.zeros([steps-1, Nx], dtype=np.float32)
        surfU_fy_ch_global = np.zeros_like(surfU_fx_ch_global)
        surfU_fz_ch_global = np.zeros_like(surfU_fx_ch_global)
        surfL_fx_ch_global = np.zeros_like(surfU_fx_ch_global)
        surfL_fy_ch_global = np.zeros_like(surfU_fx_ch_global)
        surfL_fz_ch_global = np.zeros_like(surfU_fx_ch_global)
        den_ch_bulk_global = np.zeros_like(surfU_fx_ch_global)

        # Temperature
        v_t_global = np.zeros([steps-1, Nx, Nz], dtype=np.float32)
        # Velocity (chunked in stable region)
        vx_ch_global = np.zeros_like(v_t_global)
        # Density
        den_ch_global = np.zeros_like(v_t_global)
        # Mass flux
        jx_ch_global = np.zeros_like(v_t_global)
        # Virial
        # vir_ch_global = np.zeros_like(v_t_global)

    else:
        comZ_global = None
        gap_height_global = None
        mflowrate_stable_global = None
        mflux_stable_global = None
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
        # vir_ch_global = None
        vor_vol_global = None

    sendcounts_time = np.array(comm.gather(comZ.size, root=0))
    sendcounts_chunk = np.array(comm.gather(v_t_ch.size, root=0))
    sendcounts_chunkS = np.array(comm.gather(surfU_fx_ch.size, root=0))
    sendcounts_chunk_bulk = np.array(comm.gather(den_ch_bulk.size, root=0))

    comm.Gatherv(sendbuf=comZ, recvbuf=(comZ_global, sendcounts_time), root=0)
    comm.Gatherv(sendbuf=gap_height, recvbuf=(gap_height_global, sendcounts_time), root=0)
    comm.Gatherv(sendbuf=mflowrate_stable, recvbuf=(mflowrate_stable_global, sendcounts_time), root=0)
    comm.Gatherv(sendbuf=mflux_stable, recvbuf=(mflux_stable_global, sendcounts_time), root=0)
    # comm.Gatherv(sendbuf=totVi, recvbuf=(vor_vol_global, sendcounts_time), root=0)

    comm.Gatherv(sendbuf=v_t_ch, recvbuf=(v_t_global, sendcounts_chunk), root=0)
    comm.Gatherv(sendbuf=vx_ch, recvbuf=(vx_ch_global, sendcounts_chunk), root=0)
    comm.Gatherv(sendbuf=den_ch, recvbuf=(den_ch_global, sendcounts_chunk), root=0)
    comm.Gatherv(sendbuf=jx_ch, recvbuf=(jx_ch_global, sendcounts_chunk), root=0)
    # comm.Gatherv(sendbuf=vir_ch, recvbuf=(vir_ch_global, sendcounts_chunk), root=0)

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
        vx_ch_global = vx_ch_global * A_per_fs_to_m_per_s       # m/s

        # Density
        den_ch_global = den_ch_global * (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
        den_ch_bulk_global = den_ch_bulk_global * (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3

        # Mass flux
        jx_ch_global = jx_ch_global * (ang_to_m/fs_to_ns) * (mf/sci.N_A) / (ang_to_m**3)

        # Virial Pressure
        # vir_ch_global = vir_ch_global * atm_to_mpa

        # Forces in the wall
        fx_upper = surfU_fx_ch_global*kcalpermolA_to_N       # N
        fy_upper = surfU_fy_ch_global*kcalpermolA_to_N
        fz_upper = surfU_fz_ch_global*kcalpermolA_to_N

        fx_lower = surfL_fx_ch_global*kcalpermolA_to_N
        fy_lower = surfL_fy_ch_global*kcalpermolA_to_N
        fz_lower = surfL_fz_ch_global*kcalpermolA_to_N

        fx_wall_avg = 0.5 * (fx_upper-fx_lower)
        fy_wall_avg = 0.5 * (fy_upper-fy_lower)
        fz_wall_avg = 0.5 * (fz_upper-fz_lower)

        # avg_mflux_stable = np.mean(sq.block_1D_arr(mflux_stable,100))
        avg_mflowrate = np.mean(mflowrate_stable_global)
        avg_mflux_stable = np.mean(mflux_stable_global)

        print('Average mass flux in the stable region is {} g/m2.ns \nAverage mass flow rate in the stable region is {} g/ns' \
               .format(avg_mflux_stable,avg_mflowrate))

        outfile = f"{infile.split('.')[0]}_{Nx}x{Nz}_parallel.nc"
        out = netCDF4.Dataset(outfile, 'w', format='NETCDF3_64BIT_OFFSET')
        out.createDimension('x', Nx)
        out.createDimension('z', Nz)
        out.createDimension('step', steps-1)

        out.setncattr("Lx", Lx)
        out.setncattr("Ly", Ly)
        out.setncattr("h", h)

        time_var = out.createVariable('Time', 'f4', ('step'))
        com_var =  out.createVariable('COM', 'f4', ('step'))
        gap_height_var =  out.createVariable('Height', 'f4', ('step'))
        # voronoi_volumes = out.createVariable('Voronoi_volumes', 'f4', ('step'))

        temp_var =  out.createVariable('Temperature', 'f4', ('step', 'x', 'z'))
        vx_var =  out.createVariable('Vx', 'f4', ('step', 'x', 'z'))
        jx_var =  out.createVariable('Jx', 'f4', ('step', 'x', 'z'))
        # vir_var = out.createVariable('Virial', 'f4', ('step', 'x', 'z'))
        den_var = out.createVariable('Density', 'f4',  ('step', 'x', 'z'))

        fx_wall_var =  out.createVariable('Fx_wall', 'f4',  ('step', 'x'))
        fy_wall_var =  out.createVariable('Fy_wall', 'f4',  ('step', 'x'))
        fz_wall_var =  out.createVariable('Fz_wall', 'f4',  ('step', 'x'))
        density_bulk_var =  out.createVariable('Density_Bulk', 'f4',  ('step', 'x'))

        time_var[:] = time
        com_var[:] = comZ_global
        gap_height_var[:] = gap_height_global
        # voronoi_volumes[:] = vor_vol_global

        temp_var[:] = temp
        vx_var[:] = vx_ch_global
        jx_var[:] = jx_ch_global
        # vir_var[:] = vir_ch_global
        den_var[:] = den_ch_global

        fx_wall_var[:] = fx_wall_avg
        fy_wall_var[:] = fy_wall_avg
        fz_wall_var[:] = fz_wall_avg
        density_bulk_var[:] = den_ch_bulk_global


        t1 = timer.time()

        print(t1-t0)

if __name__ == "__main__":
   proc(sys.argv[1],np.int(sys.argv[2]),np.int(sys.argv[3]))
