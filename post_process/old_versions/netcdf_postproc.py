#!/usr/bin/env python

import netCDF4
import sys
import numpy as np
import scipy.constants as sci
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import sample_quality as sq
import tesellation as tes
import warnings
import logging
import spatial_bin as sb
import os
# import create_chunks
# import matscipy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.set_printoptions(threshold=sys.maxsize)

mf = 39.948      # Molar mass of fluid in gram/mol
Kb = 1.38064852e-23 # m2 kg s-2 K-1
ang_to_cm= 1e-8
ang_to_m=1e-10
fs_to_ns=1e-6
A_per_fs_to_m_per_s = 1e5
kcalpermolA_to_N = 6.947694845598684e-11
atm_to_mpa = 0.101325

def proc(infile,tSkip,Nx,Nz):

    data = netCDF4.Dataset(infile)
    thermo_freq = 1000

    # #Variables
    # for varobj in data.variables.keys():
    #     print(varobj)

    # #Name the output files according to the input
    # base=os.path.basename(infile)
    # filename=os.path.splitext(base)[0]

    # Attributes are lost with nco split
    # for name in data.variables["time"].ncattrs():
    #     print("Global attr {} = {}".format(name, getattr(data.variables["time"], name)))

    if 'lj' in sys.argv:
        mf, A_per_molecule,scale_factor = 39.948, 1, 3
    elif 'propane' in sys.argv:
        mf, A_per_molecule,scale_factor = 44.09, 3, 1
    elif 'pentane' in sys.argv:
        mf, A_per_molecule,scale_factor = 72.15, 5, 1
    elif 'heptane' in sys.argv:
        mf, A_per_molecule,scale_factor = 100.21, 7, 1

    # Time (timesteps*timestep_size)

    sim_time = np.array(data.variables["time"])

    print('Total simualtion time: {} {}'.format(int(sim_time[-1]),data.variables["time"].units))

    tSteps = int((sim_time[-1]-sim_time[0])/(data.variables["time"].scale_factor*thermo_freq))
    tSample = tSteps-tSkip+1
    tSteps_array, tSample_array = [i for i in range (tSteps+1)], \
                                            [i for i in range (tSkip,tSteps+1)]

    # EXTRACT DATA  ------------------------------------------
    #---------------------------------------------------------

    cell_lengths = np.array(data.variables["cell_lengths"])

    # Particle Types ------------------------------
    type=np.array(data.variables["type"])
    fluid_index,solid_index=[],[]

    for idx,val in enumerate(type[0]):
        # Lennard-Jones
        if np.max(type[0])==2:
            if val == 1:    #Fluid
                fluid_index.append(idx)
            elif val == 2:  #Solid
                solid_index.append(idx)

        # Hydrocarbons
        elif np.max(type[0])==3:
            if val == 1 or val == 2:    #Fluid
                fluid_index.append(idx)
            elif val == 3:  #Solid
                solid_index.append(idx)

    Nf,Nm,Ns=len(fluid_index),np.int(len(fluid_index)/A_per_molecule),len(solid_index)
    print('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,Nm,Ns))

    # Coordinates ------------------------------

    # coords = np.array(data.variables["f_position"])
    coords = np.array(data.variables["coordinates"])
    # Array dimensions:
        # dim 0: time       dim 1: index           dim 2: dimesnion
    fluid_coords=coords[tSkip:, :np.max(fluid_index)+1, :]

    fluid_xcoords,fluid_ycoords,fluid_zcoords=coords[tSkip:, :np.max(fluid_index)+1, 0], \
                                              coords[tSkip:, :np.max(fluid_index)+1, 1], \
                                              coords[tSkip:, :np.max(fluid_index)+1, 2]

    # Nevery = 10                  # Time average every 10 tsteps (as was used in LAMMPS)
    # a = sq.block_ND_arr(tSample,fluid_xcoords,Nf,Nevery)
    # b = np.mean(a,axis=1)

    # Extrema of fluidZ coord in each timestep / dim: (tSteps)
    max_fluidZ, min_fluidZ = np.amax(fluid_zcoords,axis=1),np.amin(fluid_zcoords,axis=1)

    # Time Average of extrema fluidZ coord / dim: ()
    avgmax_fluid, avgmin_fluid = np.mean(max_fluidZ), np.mean(min_fluidZ)

    solid_xcoords = coords[tSkip:, np.min(solid_index):, 0]
    solid_zcoords = coords[tSkip:, np.min(solid_index):, 2]

    # PBC correction
        # Z-axis is non-periodic
    nonperiodic=(2,)
    pdim = [n for n in range(3) if not(n in nonperiodic)]

    hi = np.greater(fluid_coords[:, :, pdim], cell_lengths[0][pdim]).astype(int)
    lo = np.less(fluid_coords[:, :, pdim], 0.).astype(int)
    fluid_coords[:, :, pdim] += (lo - hi) * cell_lengths[0][pdim]

    fluid_posmin = np.array([np.min(fluid_xcoords),np.min(fluid_ycoords), \
                                (avgmin_fluid)])
    fluid_posmax = np.array([np.max(fluid_xcoords),np.max(fluid_ycoords), \
                              (avgmax_fluid)])
    lengths = fluid_posmax - fluid_posmin

    xlength,ylength = lengths[0],lengths[1]

    # print(avgmin_fluid,avgmax_fluid)

    # Velocities ------------------------------

    # vels = np.array(data.variables["f_velocity"])       # average of a few tsteps see Nevery, Nrep, Nfreq in LAMMPS doc
    vels = np.array(data.variables["velocities"])     # snapshot (every 1000 tSteps)

    fluid_vx,fluid_vy,fluid_vz = vels[tSkip:, :np.max(fluid_index)+1, 0], \
                                 vels[tSkip:, :np.max(fluid_index)+1, 1], \
                                 vels[tSkip:, :np.max(fluid_index)+1, 2]
    fluid_v = np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)/A_per_molecule

    # Forces ------------------------------


    # forces = np.array(data.variables["f_force"])
    forces = np.array(data.variables["forces"])

    fluid_fx,fluid_fy,fluid_fz = forces[tSkip:, :np.max(fluid_index)+1, 0], \
                                 forces[tSkip:, :np.max(fluid_index)+1, 1], \
                                 forces[tSkip:, :np.max(fluid_index)+1, 2]
    solid_fx,solid_fy,solid_fz = forces[tSkip:, np.min(solid_index):, 0], \
                                 forces[tSkip:, np.min(solid_index):, 1], \
                                 forces[tSkip:, np.min(solid_index):, 2]

    # REGIONS ------------------------------------------------
    #---------------------------------------------------------

    # SurfU and SurfL Regions
        # SurfU and SurfL groups definitions (lower than half the COM of the fluid in
        # the first tstep is assigned to surfL otherwise it is a surfU atom)
    surfL=np.less_equal(solid_zcoords,np.max(fluid_zcoords[0]/2.))

    # print(np.max(fluid_zcoords[0]/2.))

    surfU=np.greater_equal(solid_zcoords,np.max(fluid_zcoords[0]/2.))

    N_surfL,N_surfU = np.sum(surfL[0]),np.sum(surfU[0])
    if N_surfU != N_surfL:
        logger.warning("No. of surfU atoms != No. of surfL atoms")

    surfL_xcoords,surfU_xcoords = solid_xcoords*surfL , solid_xcoords*surfU
    # print(surfU_xcoords.shape)
    surfL_zcoords,surfU_zcoords = solid_zcoords*surfL , solid_zcoords*surfU

    surfU_begin,surfU_end = np.min(surfU_zcoords[:,:N_surfU],axis=1), \
                            np.max(surfU_zcoords[:,:N_surfU],axis=1)

    surfL_begin,surfL_end = np.min(surfL_zcoords[:,N_surfU:],axis=1), \
                            np.max(surfL_zcoords[:,N_surfU:],axis=1)

    avgsurfU_begin,avgsurfL_end = np.mean(surfU_begin), \
                                  np.mean(surfL_end)

    gap_height =(max_fluidZ-min_fluidZ+surfU_begin-surfL_end)/2.

    comZ = (surfU_end-surfL_begin)/2.

    # Update the lengths z-component
    fluid_posmin[2],fluid_posmax[2] = (avgmin_fluid+avgsurfL_end)/2., (avgmax_fluid+avgsurfU_begin)/2.
    lengths[2] = fluid_posmax[2] - fluid_posmin[2]

    avg_gap_height = (-avgmin_fluid-avgsurfL_end+avgmax_fluid+avgsurfU_begin)/2.# lengths[2]

    print('Average gap Height in the sampling time is {0:.3f}'.format(avg_gap_height))

    # Bulk Region ----------------------------------
    total_box_Height=np.mean(surfU_end)-np.mean(surfL_begin)
    bulkStartZ=0.25*total_box_Height
    bulkEndZ=0.75*total_box_Height
    print(bulkStartZ, bulkEndZ)
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
    # print(stable_N[0])

    #Pump Region --------------------------------------
    pumpStartX=0.0*xlength
    pumpEndX=0.2*xlength
    pump_length=pumpEndX-pumpStartX
    pump_hi = np.less_equal(fluid_xcoords, pumpEndX)
    pump_lo = np.greater_equal(fluid_xcoords, pumpStartX)
    pump_region = np.logical_and(pump_lo, pump_hi)
    pump_vol=gap_height*pump_length*ylength
    pump_N = np.sum(pump_region, axis=1)

    # Thermodynamic and Mechanical Quantities ---------------------------------
    # -------------------------------------------------------------------------

    # Gap Height over time
    if "comZ" in sys.argv:
        np.savetxt('comZ.txt', np.c_[tSample_array,comZ],
                delimiter="  ",header="time           comZ")

    # Gap Height over time
    if "gapH" in sys.argv:
        np.savetxt('h.txt', np.c_[tSample_array,gap_height],
                delimiter="  ",header="time           h")

    # Check atoms velocity distribution

    if 'mflux' in sys.argv and Nx==1 and Nz==1:
        # Measure mass flux in the stable region
        vels_stable = fluid_vx*stable_region/A_per_molecule
        mflux_stable = (np.sum(vels_stable, axis=1)*ang_to_m*mf)/(sci.N_A*fs_to_ns*stable_vol*(ang_to_m)**3)      # g/m2.ns
        mflowrate_stable = (np.sum(vels_stable, axis=1)*ang_to_m*mf)/(sci.N_A*fs_to_ns*stable_length*ang_to_m)          # g/ns

        np.savetxt('flux-stable-time.txt', np.c_[tSample_array,mflux_stable],
                delimiter="  ",header="time         mflux")

        # np.savetxt('mflowrate-stable-time.txt', np.c_[tSample_array,mflowrate_stable],          # g/ns
        #         delimiter="  ",header="time         mflux")

        # Measure mass flux in the pump region
        vels_pump = fluid_vx*pump_region/A_per_molecule
        mflux_pump = (np.sum(vels_pump, axis=1)*ang_to_m*mf)/(sci.N_A*fs_to_ns*pump_vol*(ang_to_m)**3)      # g/m2.ns

        # np.savetxt('flux-pump-time.txt', np.c_[tSample_array,mflux_pump],
        #         delimiter="  ",header="time         mflux")


        avg_mflux_stable = np.mean(mflux_stable)
        avg_mflowrate = np.mean(mflowrate_stable)
        avg_mflux_pump = np.mean(mflux_pump)

        # sq.get_bse(mflux_stable)
        # np.savetxt('bse.txt', np.c_[sq.get_bse(mflux_stable)[0],sq.get_bse(mflux_stable)[1]],
        #         delimiter="  ",header="n            bse")

        print('Average mass flux in the stable region is {} g/m2.ns \nAverage mass flow rate in the stable region is {} g/ns \
            \nAverage mass flux in the pump region is {} g/m2.ns' \
               .format(avg_mflux_stable, avg_mflowrate, avg_mflux_pump))


        # jacf = sq.acf(mflux_stable)

        # # Fourier transform of the ACF > Spectrum density
        # j_tq = np.fft.fft(jacf)

        # np.savetxt('flux-acf.txt', np.c_[tSample_array,jacf],
        #         delimiter="  ",header="time         acf")

        # np.savetxt('flux-spectra.txt', np.c_[tSample_array,j_tq.real],
        #         delimiter="  ",header="time         sprctra")

        # # jacf_avg = np.mean(jacf,axis=1)
        # # j_avg = np.mean(rho_tq,axis=1)

        # Inverse DFT
        # rho_itq = np.fft.ifftn(rho_tq,axes=(0,1))
        # rho_itq_avg = np.mean(rho_itq,axis=1)

        # dacf=acf(density_tc[:t_cut,:,0,0])
        # dacf_avg = np.mean(dacf,axis=1)
        #
        # # Fourier transform of the ACF > Spectrum density
        # rho_tq = np.fft.fftn(dacf,axes=(0,1))
        # rho_tq_avg = np.mean(rho_tq,axis=1)

    elif 'temp' in sys.argv:

        vels = np.array(data.variables["velocities"])

        fluid_vx,fluid_vy,fluid_vz = vels[tSkip:, :np.max(fluid_index)+1, 0], \
                                     vels[tSkip:, :np.max(fluid_index)+1, 1], \
                                     vels[tSkip:, :np.max(fluid_index)+1, 2]
        fluid_v = np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)/A_per_molecule

        # Measure Temperature of the fluid region
        temp = np.sum(mf*1e-3*(fluid_v)**2,axis=1)*A_per_fs_to_m_per_s**2 \
                                                /(sci.N_A*3*Nm*Kb)                      # Kelvin

        np.savetxt('temp.txt', np.c_[tSample_array,temp],
                delimiter="  ",header="time         temp")

    elif 'fx' in sys.argv:
        fx = np.sum(fluid_fx * kcalpermolA_to_N, axis=1)

        np.savetxt('fx.txt', np.c_[tSample_array,fx],
                delimiter="  ",header="time         fx")


    # Stresses in the walltSteps+1
    elif 'sigwall' in sys.argv:
        forcesU = np.array(data.variables["f_fAtomU_avg"])
        forcesL = np.array(data.variables["f_fAtomL_avg"])

        surfU_fx,surfU_fy,surfU_fz = forcesU[tSkip:, np.min(solid_index):, 0][:,:N_surfU], \
                                     forcesU[tSkip:, np.min(solid_index):, 1][:,:N_surfU], \
                                     forcesU[tSkip:, np.min(solid_index):, 2][:,:N_surfU]

        surfL_fx,surfL_fy,surfL_fz = forcesL[tSkip:, np.min(solid_index):, 0][:,N_surfU:], \
                                     forcesL[tSkip:, np.min(solid_index):, 1][:,N_surfU:], \
                                     forcesL[tSkip:, np.min(solid_index):, 2][:,N_surfU:]

        # # Measure sigmazz in the upper wall
        sigmaZZ_upper = np.sum(surfU_fz,axis=1)*kcalpermolA_to_N*1e-6/(xlength*ylength*ang_to_m**2)       # MPa
        sigmaZZ_lower = np.sum(surfL_fz,axis=1)*kcalpermolA_to_N*1e-6/(xlength*ylength*ang_to_m**2)       # MPa
        sigmaXZ_upper = np.sum(surfU_fx,axis=1)*kcalpermolA_to_N*1e-6/(xlength*ylength*ang_to_m**2)       # MPa
        sigmaXZ_lower = np.sum(surfL_fx,axis=1)*kcalpermolA_to_N*1e-6/(xlength*ylength*ang_to_m**2)       # MPa
        sigmaYZ_upper = np.sum(surfU_fy,axis=1)*kcalpermolA_to_N*1e-6/(xlength*ylength*ang_to_m**2)       # MPa
        sigmaYZ_lower = np.sum(surfL_fy,axis=1)*kcalpermolA_to_N*1e-6/(xlength*ylength*ang_to_m**2)       # MPa

        sigmaXZ = 0.5 * (sigmaXZ_upper-sigmaXZ_lower)
        sigmaYZ = 0.5 * (sigmaYZ_upper-sigmaYZ_lower)
        sigmaZZ = 0.5 * (sigmaZZ_upper-sigmaZZ_lower)

        np.savetxt('sig-wall-both.txt', np.c_[tSample_array, sigmaXZ_upper, sigmaXZ_lower,
                                                       sigmaYZ_upper, sigmaYZ_lower,
                                                       sigmaZZ_upper, sigmaZZ_lower],
                   delimiter="  ",header="time         sigxzU        sigxzL    sigyzU \
                                                       sigyzL        sigzzU    sigzzL")

        np.savetxt('sig-wall-avg.txt', np.c_[tSample_array, sigmaXZ, sigmaYZ, sigmaZZ],
                  delimiter="  ",header="time         sigxz        sigxz    sigzz" )


    elif 'virial' in sys.argv:

        vornoi_vol = np.array(data.variables["f_Vi_avg"])
        Vi = vornoi_vol[tSkip:,:np.max(fluid_index)+1] * bulk_region
        totVi = np.sum(Vi,axis=1)

        virial = np.array(data.variables["f_Wi_avg"])
        W1 = np.sum(virial[tSkip:,:np.max(fluid_index)+1,0],axis=1)
        W2 = np.sum(virial[tSkip:,:np.max(fluid_index)+1,1],axis=1)
        W3 = np.sum(virial[tSkip:,:np.max(fluid_index)+1,2],axis=1)
        totWi = W1+W2+W3

        virial_press = -totWi*atm_to_mpa/(3*totVi)

        np.savetxt('virial-time.txt', np.c_[tSample_array,virial_press],
                delimiter="  ",header="time         virial")


    vels_t = np.array(data.variables["velocities"])
    vx_t = vels_t[tSkip:,:np.max(fluid_index)+1,0]
    vy_t = vels_t[tSkip:,:np.max(fluid_index)+1,1]
    vz_t = vels_t[tSkip:,:np.max(fluid_index)+1,2]
    v_t = np.sqrt(vx_t**2+vy_t**2+vz_t**2)/A_per_molecule


    # Spatial Binning ---------------------------------------------------------
    # -------------------------------------------------------------------------



    # # TODO:  TEMP IN CHUNKS

    Ny = 1
    size = np.array([Nx, Ny, Nz])

    # Chunk in flow direction
    if Nx>1:
        # Bounds
        bounds = [np.arange(size[i] + 1) / size[i] * lengths[i] + fluid_posmin[i]
                        for i in range(3)]

        # Bounds Stable
        stableRange = np.arange(size[0] + 1) / size[0] * stable_length + stableStartX

        bounds_stable = [stableRange, np.array([bounds[1]]), np.array([bounds[2]])]
        xx_stable, yy_stable, zz_stable = np.meshgrid(bounds_stable[0], bounds_stable[1], bounds_stable[2])
        xx_stable = np.transpose(xx_stable, (1, 0, 2))
        yy_stable = np.transpose(yy_stable, (1, 0, 2))
        zz_stable = np.transpose(zz_stable, (1, 0, 2))
        dx_stable = xx_stable[1:, 1:, 1:] - xx_stable[:-1, :-1, :-1]
        dy_stable = yy_stable[1:, 1:, 1:] - yy_stable[:-1, :-1, :-1]
        dz_stable = zz_stable[1:, 1:, 1:] - zz_stable[:-1, :-1, :-1]
        vol_stable = dx_stable * dy_stable * dz_stable

        if 'density' in sys.argv:
            bounds[2] = [bulkStartZ, bulkEndZ]

        # print(bounds)

        # Stable region bounds
        # bounds_fluid[0]= np.asarray([stableStartX,stableEndX])

        xx, yy, zz = np.meshgrid(bounds[0], bounds[1], bounds[2])

        xx = np.transpose(xx, (1, 0, 2))
        yy = np.transpose(yy, (1, 0, 2))
        zz = np.transpose(zz, (1, 0, 2))

        dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
        dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
        dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]

        vol = dx * dy * dz

        density = np.empty([ tSample, Nx, Ny, Nz])
        flux = np.empty([ tSample, Nx, Ny, Nz])
        vx = np.empty([ tSample, Nx, Ny, Nz])
        fx = np.empty([ tSample, Nx, Ny, Nz])
        sigxzU = np.empty([ tSample, Nx, Ny, Nz])
        sigyzU = np.empty([ tSample, Nx, Ny, Nz])
        sigzzU = np.empty([ tSample, Nx, Ny, Nz])
        sigxzL = np.empty([ tSample, Nx, Ny, Nz])
        sigyzL = np.empty([ tSample, Nx, Ny, Nz])
        sigzzL = np.empty([ tSample, Nx, Ny, Nz])
        virial_press = np.empty([ tSample, Nx, Ny, Nz])

        Ntotal = 0
        chunk_den = np.zeros([tSample, Nx])
        chunk_flux = np.zeros([tSample, Nx])
        chunk_vx = np.zeros([tSample, Nx])
        chunk_fx = np.zeros([tSample, Nx])
        chunk_sigxzU = np.zeros([tSample, Nx])
        chunk_sigyzU = np.zeros([tSample, Nx])
        chunk_sigzzU = np.zeros([tSample, Nx])
        chunk_sigxzL = np.zeros([tSample, Nx])
        chunk_sigyzL = np.zeros([tSample, Nx])
        chunk_sigzzL = np.zeros([tSample, Nx])
        chunk_virialP = np.zeros([tSample, Nx])
        chunk_v_t  = np.zeros([tSample, Nx])


        length=np.arange(dx[0]/2.0,xlength,dx[0])
        length/=10 # nm


        for i in range(Nx):
            # create a mask to filter only particles in grid cell
            xlo = np.less(fluid_xcoords, xx[i+1, 0, 0])
            xhi = np.greater_equal(fluid_xcoords, xx[i, 0, 0])
            cellx = np.logical_and(xlo, xhi)

            # Measure the bulk density
            if 'density' in sys.argv:
                zlo = np.less(fluid_zcoords, zz[0, 0, 1])
                zhi = np.greater_equal(fluid_zcoords, zz[0, 0, 0])
                cellz = np.logical_and(zlo, zhi)
                cellx = np.logical_and(cellx, cellz)

            # count particles in cell
            N = np.sum(cellx, axis=1)
            Ntotal += N

            N_stable = np.sum(cellx, axis=1)

            ## TODO: Measure the force before the second integration (f(t+delta t)) NOT REALLY!!
            if 'fx' in sys.argv:
                fx[:, i, 0, 0] = np.sum(fluid_fx * cellx, axis=1) / N
                chunk_fx[:,i]=fx[:,i,0,0]

            elif 'temp' in sys.argv:
                chunk_v_t[:,i]= np.sum(v_t * cellx, axis=1) / N

            elif 'vx' in sys.argv:
                xlo_stable = np.less(fluid_xcoords, xx_stable[i+1, 0, 0])
                xhi_stable = np.greater_equal(fluid_xcoords, xx_stable[i, 0, 0])
                cell_stablex = np.logical_and(xlo_stable, xhi_stable)

                vx[:, i, 0, 0] = np.sum(fluid_vx * cell_stablex, axis=1) / N_stable
                chunk_vx[:,i]=vx[:, i, 0, 0]/A_per_molecule

            elif 'density' in sys.argv:
                density[:, i, 0, 0] = N / vol[i, 0, 0]
                chunk_den[:,i]=density[:, i, 0, 0]/A_per_molecule

            elif 'mflux' in sys.argv:
                density[:, i, 0, 0] = N / vol[i, 0, 0]
                vcmx = np.sum(fluid_vx * cellx, axis=1) / N
                flux[:, i, 0, 0] = vcmx * density[:, i, 0, 0]
                chunk_flux[:,i]=flux[:, i, 0, 0]

            elif 'sigwall' in sys.argv:
                xloU = np.less(surfU_xcoords[:,:N_surfU], xx[i+1, 0, 0])
                xhiU = np.greater_equal(surfU_xcoords[:,:N_surfU], xx[i, 0, 0])
                cellxU = np.logical_and(xloU, xhiU)
                xloL = np.less(surfL_xcoords[:,N_surfU:], xx[i+1, 0, 0])
                xhiL = np.greater_equal(surfL_xcoords[:,N_surfU:], xx[i, 0, 0])
                cellxL = np.logical_and(xloL, xhiL)

                sigxzU[:, i, 0, 0] = np.sum(surfU_fx * cellxU, axis=1)
                sigyzU[:, i, 0, 0] = np.sum(surfU_fy * cellxU, axis=1)
                sigzzU[:, i, 0, 0] = np.sum(surfU_fz * cellxU, axis=1)
                sigxzL[:, i, 0, 0] = np.sum(surfL_fx * cellxL, axis=1)
                sigyzL[:, i, 0, 0] = np.sum(surfL_fy * cellxL, axis=1)
                sigzzL[:, i, 0, 0] = np.sum(surfL_fz * cellxL, axis=1)

                chunk_sigxzU[:,i] = sigxzU[:, i, 0, 0]
                chunk_sigyzU[:,i] = sigyzU[:, i, 0, 0]
                chunk_sigzzU[:,i] = sigzzU[:, i, 0, 0]
                chunk_sigxzL[:,i] = sigxzL[:, i, 0, 0]
                chunk_sigyzL[:,i] = sigyzL[:, i, 0, 0]
                chunk_sigzzL[:,i] = sigzzL[:, i, 0, 0]

            elif 'virial' in sys.argv:

                #totVi_chunk = np.sum(vornoi_vol[tSkip:,:np.max(fluid_index)+1] * cellx, axis=1) # This will
                # result in virial pressure varying with the chunk size

                W1_chunk = np.sum(virial[tSkip:,:np.max(fluid_index)+1,0] * cellx, axis=1)
                W2_chunk = np.sum(virial[tSkip:,:np.max(fluid_index)+1,1] * cellx, axis=1)
                W3_chunk = np.sum(virial[tSkip:,:np.max(fluid_index)+1,2] * cellx, axis=1)
                totWi_chunk = W1_chunk+W2_chunk+W3_chunk

                virial_press[:, i, 0, 0] = -totWi_chunk
                chunk_virialP[:,i] = virial_press[:, i, 0, 0]

        if 'temp' in sys.argv:
            # chunk_v_t = np.mean(chunk_v_t,axis=0)
            # print(mf)
            # temp = ((mf*1e-3)/sci.N_A) * chunk_v_t * A_per_fs_to_m_per_s**2 / Kb

            temp = (mf*1e-3/sci.N_A) * (np.mean(chunk_v_t,axis=0)*A_per_fs_to_m_per_s)**2 \
                                        /(3 * (N/A_per_molecule) * Kb)          # Kelvin

            np.savetxt('tempX.txt', np.c_[length[1:-1],temp[1:-1]],
                    delimiter="  ",header="length           temp")

        elif 'virial' in sys.argv:
            totVi_chunk = bulk_vol/Nx
            virialP=np.mean(chunk_virialP,axis=0)*atm_to_mpa/(3*totVi_chunk)

            np.savetxt('virialX.txt', np.c_[length[1:-1],virialP[1:-1]],
                    delimiter="  ",header="length           virial press.")

        elif 'sigwall' in sys.argv:

            countU = np.count_nonzero(cellxU[0])      # Should be the same in each chunk
            countU2 = np.count_nonzero(cellxU[-1])
            countL = np.count_nonzero(cellxL[0])

            if countU != countL:
                logger.warning("No. of surfU atoms in chunk 1 != No. of surfL atoms in chunk 1")

            if countU != countU2:
                logger.warning("No. of surfU atoms in chunk 1 != No. of surfU atoms in the last chunk")

            sigxzU= np.mean(chunk_sigxzU,axis=0)*kcalpermolA_to_N*1e-6/(dx[0][0]*ylength*ang_to_m**2)
            sigyzU= np.mean(chunk_sigyzU,axis=0)*kcalpermolA_to_N*1e-6/(dx[0][0]*ylength*ang_to_m**2)
            sigzzU= np.mean(chunk_sigzzU,axis=0)*kcalpermolA_to_N*1e-6/(dx[0][0]*ylength*ang_to_m**2)

            sigxzL= np.mean(chunk_sigxzL,axis=0)*kcalpermolA_to_N*1e-6/(dx[0][0]*ylength*ang_to_m**2)
            sigyzL= np.mean(chunk_sigyzL,axis=0)*kcalpermolA_to_N*1e-6/(dx[0][0]*ylength*ang_to_m**2)
            sigzzL= np.mean(chunk_sigzzL,axis=0)*kcalpermolA_to_N*1e-6/(dx[0][0]*ylength*ang_to_m**2)

            sigxz = 0.5 * (sigxzU-sigxzL)
            sigyz = 0.5 * (sigyzU-sigyzL)
            sigzz = 0.5 * (sigzzU-sigzzL)

            np.savetxt('sig-wall-bothX.txt', np.c_[length, sigxzU, sigyzU, sigzzU,
                                                           sigxzL, sigyzL, sigzzL],
                    delimiter="  ",header="length           sigxzU        sigyzU    sigzzU \
                                                            sigyzL        sigzzU    sigzzL")

            np.savetxt('sig-wall-avgX.txt', np.c_[length, sigxz, sigyz, sigzz],
                    delimiter="  ",header="length         sigxz        sigxz    sigzz" )

        elif 'fx' in sys.argv:
            fx=np.mean(chunk_fx,axis=0)*kcalpermolA_to_N

            np.savetxt('fxX.txt', np.c_[length,fx],
                    delimiter="  ",header="length           fx")

        elif 'vx' in sys.argv:
            vx=np.mean(chunk_vx,axis=0)*A_per_fs_to_m_per_s

            np.savetxt('vxX.txt', np.c_[length,vx],
                    delimiter="  ",header="length           vx")

        elif 'density' in sys.argv:
            density=np.mean(chunk_den,axis=0)*mf/(sci.N_A*(ang_to_cm**3))

            np.savetxt('densityX.txt', np.c_[length[1:-1],density[1:-1]],
                    delimiter="  ",header="length           rho")

        elif 'mflux' in sys.argv:
            flux=np.mean(chunk_flux,axis=0)*(ang_to_m/fs_to_ns)*mf/(sci.N_A*(ang_to_m**3))

            np.savetxt('fluxX.txt', np.c_[length,flux],
                    delimiter="  ",header="length           jx")



    if Nz>1:

        bounds = [np.arange(size[i] + 1) / size[i] * lengths[i] + fluid_posmin[i]
                    for i in range(3)]

        if 'vx' in sys.argv:
        #     bounds[0] = [stableStartX, stableEndX]
            print(bounds)

        # bounds[0]= np.asarray([stableStartX,stableEndX])

        xx, yy, zz = np.meshgrid(bounds[0], bounds[1], bounds[2])

        xx = np.transpose(xx, (1, 0, 2))
        yy = np.transpose(yy, (1, 0, 2))
        zz = np.transpose(zz, (1, 0, 2))

        dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
        dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
        dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]

        vol = dx * dy * dz

        density = np.empty([tSample, Nx, Ny, Nz])
        flux = np.empty([tSample, Nx, Ny, Nz])
        vx = np.empty([tSample, Nx, Ny, Nz])
        fx = np.empty([tSample, Nx, Ny, Nz])

        Ntotal=0
        chunk_den=np.zeros([tSample, Nz])
        chunk_flux=np.zeros([tSample, Nz])
        chunk_vx=np.zeros([tSample, Nz])
        chunk_fx=np.zeros([tSample, Nz])

        height=np.arange(fluid_posmin[2]+(dz[0,0,0]/2),fluid_posmax[2],dz[0,0,0])
        height/=10 # nm

        for i in range(Nz):
            #create a mask to filter only particles in grid cell
            zlo = np.less(fluid_zcoords, zz[0, 0, i+1])
            zhi = np.greater_equal(fluid_zcoords, zz[0, 0, i])
            xlo = np.less(fluid_xcoords, stableEndX)
            xhi = np.greater_equal(fluid_xcoords, stableStartX)

            # print(zz[0, 0, i],zz[0, 0, i+1])
            cellx = np.logical_and(xlo, xhi)
            cellz = np.logical_and(zlo, zhi)
            cell = np.logical_and(cellx,cellz)

            # count particles in cell
            N = np.sum(cell, axis=1)
            Ntotal += N

            # count particles in cell
            # N = np.sum(cellz, axis=1)
            # Ntotal += N
            #
            # # To avoid averaging over zero
            Nzero = np.less(N, 1)
            N[Nzero] = 1

            # if 'vx' in sys.argv:
            #     xlo = np.less(fluid_xcoords, xx[1, 0, 0])
            #     xhi = np.greater_equal(fluid_xcoords, xx[0, 0, 0])
            #     cellx = np.logical_and(xlo, xhi)
            #     cellz = np.logical_and(cellx, cellz)

            if 'fx' in sys.argv:
                fx[tSkip:, 0, 0, i] = np.sum(fluid_fx * cellz, axis=1)/ N
                chunk_fx[:,i]=fx[:,0,0,i]

            if 'density' in sys.argv:
                density[:, 0, 0, i] = N / (A_per_molecule*vol[0, 0, i])
                chunk_den[:,i]=density[:,0,0,i]

            if 'vx' in sys.argv:
                vx[:, 0, 0, i] = np.sum(fluid_vx * cell, axis=1)/ N
                chunk_vx[:,i]=vx[:,0,0,i]/A_per_molecule


        if 'fx' in sys.argv:
            fx=np.mean(chunk_fx,axis=0)*kcalpermolA_to_N
            np.savetxt('fxZ.txt', np.c_[height[1:-1],fx[1:-1]],
                    delimiter="  ",header="height           fx")

        elif 'vx' in sys.argv:
            vx=np.mean(chunk_vx,axis=0)*A_per_fs_to_m_per_s
            np.savetxt('vxZ-stable.txt', np.c_[height[1:-1],vx[1:-1]],
                    delimiter="  ",header="height           vx")

        elif 'density' in sys.argv:
            density=np.mean(chunk_den,axis=0)*mf/(sci.N_A*(ang_to_cm**3))
            np.savetxt('densityZ.txt', np.c_[height,density],
                    delimiter="  ",header="height           rho")



if __name__ == "__main__":
   proc(sys.argv[1],np.int(sys.argv[2]),np.int(sys.argv[3]),np.int(sys.argv[4]))
