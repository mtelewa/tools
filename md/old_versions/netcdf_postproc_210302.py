#!/usr/bin/env python

import netCDF4
import sys
import numpy as np
import scipy.constants as sci
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
from block_data import block_ND_arr as bnd
from block_data import block_1D_arr as b1d
# import ase
# from ase import io
# from ase import neighborlist
import tesellation as tes
import warnings
import logging
# import create_chunks
# import matscipy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
np.set_printoptions(threshold=sys.maxsize)

# zlength = 65    # Angstrom
# Boffset = 1     # Angstrom
# offset = 5      # Angstrom
# h = 40          # Angstrom

r_cutoff = 15 # Angstrom
T = 100 #K

mf = 39.948      # Molar mass of fluid in gram/mol
# mf_gram = 6.6335209e-23     # g
Kb = 1.38064852e-23 # m2 kg s-2 K-1
ang_to_cm= 1e-8
A_per_fs_to_m_per_s = 1e5
kcalpermolA_to_N = 6.947694845598684e-11
ang_to_m=1e-10
fs_to_ns=1e-6


def proc(infile,tSkip,loaded='No'):

    data = netCDF4.Dataset(infile)
    thermo_freq = 1000

    # Variables
    for varobj in data.variables.keys():
        print(varobj)

    # Time (timesteps*timestep_size)
    sim_time = np.array(data.variables["time"])
    print('Total simualtion time: {} {}'.format(int(sim_time[-1]),data.variables["time"].units))
    tSteps = int((sim_time[-1]-sim_time[0])/(data.variables["time"].scale_factor*thermo_freq))
    tSample=tSteps-tSkip+1
    tSteps_array,tSample_array=[i for i in range (tSteps+1)], \
                               [i for i in range (tSkip,tSteps+1)]

    # print(len(tSample_array))
    # print(tSample)

    # EXTRACT DATA  -------------------------------
    #----------------------------------------------

    cell_lengths = np.array(data.variables["cell_lengths"])
    # print(cell_lengths)

    # Particle Types ------------------------------
    type = np.array(data.variables["type"])
    fluid_index=[]
    solid_index=[]
    # if there are walls, zposition is adjusted
    for idx,val in enumerate(type[0]):
        if val == 1:    #Fluid
            fluid_index.append(idx)
        elif val == 2:  #Solid
            solid_index.append(idx)

    Nf,Ns=len(fluid_index),len(solid_index)
    # print(Nf,Ns)

    # Coordinates ------------------------------
    coords = np.array(data.variables["coordinates"])


    # arr dimensions:(time,index,dim[x/y/z])

    def xyz(arr):
        whole_fluid = arr[:, :np.max(fluid_index)+1, :]
        whole_solid = arr[:,  np.min(solid_index):,  :]
        x,y,z = arr[:, :, 0],arr[:, :, 1],arr[:, :, 2]
        x_fluid,y_fluid,z_fluid = x[:, :np.max(fluid_index)+1], \
                                  y[:, :np.max(fluid_index)+1], \
                                  z[:, :np.max(fluid_index)+1]
        x_solid,y_solid,z_solid = x[:, np.min(solid_index):], \
                                  y[:, np.min(solid_index):], \
                                  z[:, np.min(solid_index):]
        x_sample,y_sample,z_sample = x[tSkip:, :], \
                                     y[tSkip:, :], \
                                     z[tSkip:, :]
        x_sample_fluid,y_sample_fluid,z_sample_fluid = x[tSkip:, :np.max(fluid_index)+1], \
                                                       y[tSkip:, :np.max(fluid_index)+1], \
                                                       z[tSkip:, :np.max(fluid_index)+1]
        x_sample_solid,y_sample_solid,z_sample_solid = x[tSkip:, np.min(solid_index):], \
                                                       y[tSkip:, np.min(solid_index):], \
                                                       z[tSkip:, np.min(solid_index):]

                #0         1
        return (whole_fluid,whole_solid, \
                #2        3      4
               x_fluid,y_fluid,z_fluid, \
                #5        6      7
               x_solid,y_solid,z_solid,\
                #8        9      10
               x_sample,y_sample,z_sample, \
                #11              12              13
               x_sample_fluid,y_sample_fluid,z_sample_fluid, \
                #14              15              16
               x_sample_solid,y_sample_solid,z_sample_solid)

    # Array dimensions
    # dim 0: time
    # dim 1: atoms
    # dim 2: dimesnions

    fluid_coords=xyz(coords)[0]
    fluid_xcoords,fluid_ycoords,fluid_zcoords=xyz(coords)[2],xyz(coords)[3],xyz(coords)[4]
    fluid_xcoords_sample,fluid_ycoords_sample,fluid_zcoords_sample=xyz(coords)[11],xyz(coords)[12],xyz(coords)[13]
    solid_zcoords=xyz(coords)[7]

    # Zcoord at timestep 0 for all solid atoms
    # print(solid_zcoords[0,:])

    surfL_idx,surfU_idx=[],[]

    # SurfU and SurfL groups definitions (lower than half the COM of the fluid in
    #       the first tstep is assigned to surfL otherwise it is a surfU atom)
    for idx,value in enumerate(solid_zcoords[0,:]):
        if value < np.max(fluid_zcoords[0])/2.:
            # print(idx)
            surfL_idx.append(idx)
        else:
            surfU_idx.append(idx)

    surfL_zcoords=solid_zcoords[tSkip:,surfL_idx]
    surfU_zcoords=solid_zcoords[tSkip:,surfU_idx]

    surfU_begin,surfU_end=np.min(surfU_zcoords,axis=1),np.max(surfU_zcoords,axis=1)
    surfL_begin,surfL_end=np.min(surfL_zcoords,axis=1),np.max(surfL_zcoords,axis=1)

    # No. of surfU atoms (same for surfL) for the first timestep
    NsurfU = len(surfU_zcoords[0])
    NsurfL = len(surfL_zcoords[0])
    if NsurfU != NsurfL:
        logger.warning("No. of surfU atoms != No. of surfL atoms")

    # Get the maxiumum zcoord in each timestep in the sampling time
    max_fluid,min_fluid=np.amax(fluid_zcoords_sample,axis=1),\
                        np.amin(fluid_zcoords_sample,axis=1)

    gap_height =(max_fluid-min_fluid+surfU_begin-surfL_end)/2.

    # print(len(gap_height),len(tSample))

    np.savetxt('h-nc.txt', np.c_[tSample_array,gap_height],
            delimiter="  ",header="time           h")


    avgmax_fluid,avgmin_fluid=np.mean(max_fluid),np.mean(min_fluid)
    avgsurfU_begin,avgsurfL_end=np.mean(surfU_begin),np.mean(surfL_end)

    avg_gap_height=(avgmax_fluid-avgmin_fluid+avgsurfU_begin-avgsurfL_end)/2.
    print('Average gap Height is {0:.3f}'.format(avg_gap_height))

    exit()


    # PBC correction:
    # Z-axis is non-periodic
    nonperiodic=(2,)
    pdim = [n for n in range(3) if not(n in nonperiodic)]

    hi = np.greater(fluid_coords[:, :, pdim], cell_lengths[0][pdim]).astype(int)
    lo = np.less(fluid_coords[:, :, pdim], 0.).astype(int)
    fluid_coords[:, :, pdim] += (lo - hi) * cell_lengths[0][pdim]

    fluid_posmin = np.array([np.min(fluid_xcoords),np.min(fluid_ycoords),(avgsurfL_end+avgmin_fluid)/2.])
    fluid_posmax = np.array([np.max(fluid_xcoords),np.max(fluid_ycoords),(avgsurfU_begin+avgmax_fluid)/2.])
    lengths = fluid_posmax - fluid_posmin

    xlength,ylength=lengths[0],lengths[1]

    # Velocities ------------------------------
    vels = np.array(data.variables["velocities"])

    fluid_vx,fluid_vy,fluid_vz=xyz(vels)[2],xyz(vels)[3],xyz(vels)[4]
    fluid_vx_sample,fluid_vy_sample,fluid_vz_sample=xyz(vels)[11],xyz(vels)[12],xyz(vels)[13]
    # print(fluid_vx_sample.shape)

    # Forces ------------------------------
    forces = np.array(data.variables["forces"])

    # dimension(time,atoms)
    fluid_fx,fluid_fy,fluid_fz=xyz(forces)[2],xyz(forces)[3],xyz(forces)[4]
    fluid_fx_sample,fluid_fy_sample,fluid_fz_sample=xyz(forces)[11],xyz(forces)[12],xyz(forces)[13]
    solid_fx_sample,solid_fy_sample,solid_fz_sample=xyz(forces)[14],xyz(forces)[15],xyz(forces)[16]

    # REGIONS -------------------------------

    # # Bulk Region
    # total_box_Height=np.mean(surfU_end)-np.mean(surfL_begin)
    # bulkStartZ=0.25*total_box_Height
    # bulkEndZ=0.75*total_box_Height
    # # print(bulkEndZ)
    # bulkHeight=bulkEndZ-bulkStartZ
    # # print(bulkHeight)
    # bulk_lo = np.less_equal(fluid_zcoords_sample, bulkEndZ)
    # bulk_hi = np.greater_equal(fluid_zcoords_sample, bulkStartZ)
    # bulk_region = np.logical_and(bulk_lo, bulk_hi)
    # # print(bulk_region.shape)
    # bulk_vol=bulkHeight*xlength*ylength
    # # print(bulk_vol)
    # # Number of bulk atoms in each timestep
    # bulk_N = np.sum(bulk_region, axis=1)
    # # print(bulk_N)
    #
    # bulk_atoms_idx_all=[]
    # bulk_atoms_coords_all=[]
    # for i in range(tSample):
    #     bulk_atoms_idx=[]
    #     for idx,value in enumerate(bulk_region[i]):
    #         if value == 1:
    #             bulk_atoms_idx.append(idx)
    #     # Bulk atoms indeces at all timesteps (a list of lists)
    #     bulk_atoms_idx_all.append(bulk_atoms_idx)
    #
    # # bulk_atoms_idx=np.asarray(bulk_atoms_idx)
    # if len(bulk_atoms_idx_all[0])!=bulk_N[0]:
    #     logger.warning("The no. of bulk atoms in the first timestep is not the same \
    #                     as the sum of atomic indeces of atoms in the bulk region")


    #Stable Region
    stableStartX=0.4*xlength
    stableEndX=0.8*xlength
    stable_length=stableEndX-stableStartX

    stable_lo = np.less_equal(fluid_xcoords_sample, stableEndX)
    stable_hi = np.greater_equal(fluid_xcoords_sample, stableStartX)
    stable_region = np.logical_and(stable_lo, stable_hi)

    stable_vol=gap_height*stable_length*ylength

    # Number of stable atoms in each timestep
    stable_N = np.sum(stable_region, axis=1)

    stable_atoms_idx_all=[]
    for i in range(tSample):
        stable_atoms_idx=[]
        for idx,value in enumerate(stable_region[i]):
            if value == 1:
                stable_atoms_idx.append(idx)
        # Bulk atoms indeces at all timesteps (a list of lists)
        stable_atoms_idx_all.append(stable_atoms_idx)

    if len(stable_atoms_idx_all[0])!=stable_N[0]:
        logger.warning("The no. of Stable atoms in the first timestep is not the same \
                        as the sum of atomic indeces of atoms in the stable region")


    exit()


    vx_stable=[]
    vels_stable=[]
    vels_stable_avg=[]
    for i in range(tSample):
        vels_stable.append(vels[i, stable_atoms_idx_all[i],:])
        vx_stable.append(vels_stable[i][:,0])
        vels_stable_avg.append(np.mean(vx_stable[i])*stable_N[i])
        # print('ok')

    vels_stable_avg=np.asarray(vels_stable_avg)

    mflux=(vels_stable_avg*ang_to_m*mf)/(sci.N_A*fs_to_ns*stable_vol*(ang_to_m)**3)
    mflux=np.asarray(mflux)

    np.savetxt('flux-time-nc.txt', np.c_[tSample_array,mflux],
            delimiter="  ",header="time         mflux")

    # Test block size
    n=np.asarray([50,75,100,150,200,250,300,400,500,600])
    n_chosen=300

    M=tSample/n
    M_array=np.arange(1,(tSample/n_chosen)+1,1)

    blocks_flux=b1d(mflux,n_chosen)

    # blocks_flux,std_dev=[],[]
    # for i in n:
    #     blocks_flux.append(b1d(mflux,i))
    #     std_dev.append(np.std(blocks_flux[-1], ddof=1))
    #
    # std_dev=np.asarray(std_dev)
    # # Block standard error
    # bse=std_dev/np.sqrt(M)

    np.savetxt('flux-blocks-nc.txt', np.c_[M_array,blocks_flux],
            delimiter="  ",header="time         mflux")

    # np.savetxt('bse-blocks-nc.txt', np.c_[n,bse],
    #         delimiter="  ",header="n            bse")


    # avg_d=np.mean(blocks_flux)
    # print(avg_d)


    exit()




    # # AUTOCORRELATION Functions ----------------------------------
    #
    # def acf(f_tq):
    #     n = f_tq.shape[0]
    #     var = np.var(f_tq, axis=0)
    #     f_tq -= np.mean(f_tq, axis=0)
    #
    #     C = np.array([np.sum(f_tq * f_tq, axis=0) if i == 0 else np.sum(f_tq[i:] * f_tq[:-i], axis=0)
    #                   for i in range(n)]) / (n*var)
    #
    #     return C
    #
    # # dimensions of density_tc
    # # dim 0: time
    # # dim 1: chunks in x
    # # dim 2: chunks in y
    # # dim 3: chunks in z
    #
    # # Cutoff for the VACF (for figures)
    # # t_cut=200
    # jacf=acf(mflux)
    # # jacf_avg = np.mean(jacf,axis=1)
    #
    # # Fourier transform of the ACF > Spectrum density
    # j_tq = np.fft.fft(jacf)
    # # j_avg = np.mean(rho_tq,axis=1)
    #
    # np.savetxt('flux-acf-nc.txt', np.c_[tSample_array,jacf],
    #         delimiter="  ",header="time         acf")
    #
    # np.savetxt('flux-spectra-nc.txt', np.c_[tSample_array,j_tq.real],
    #         delimiter="  ",header="time         sprctra")
    #
    # exit()

    # create grid

    Nx = 1
    Ny = 1
    Nz = 48

    size = np.array([Nx, Ny, Nz])

    bounds = [np.arange(size[i] + 1) / size[i] * lengths[i] + fluid_posmin[i]
                for i in range(3)]

    bounds[0]= np.asarray([stableStartX,stableEndX])

    xx, yy, zz = np.meshgrid(bounds[0], bounds[1], bounds[2])

    xx = np.transpose(xx, (1, 0, 2))
    yy = np.transpose(yy, (1, 0, 2))
    zz = np.transpose(zz, (1, 0, 2))

    dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
    dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
    dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]
    # print(dx[0],dy[0],dz[0])
    vol = dx * dy * dz

    # CHUNKING DATA ----------------------------------------

    def chunkX():

        density = np.empty([tSteps+1, Nx, Ny, Nz])
        flux = np.empty([tSteps+1, Nx, Ny, Nz])
        vx = np.empty([tSteps+1, Nx, Ny, Nz])
        fx = np.empty([tSteps+1, Nx, Ny, Nz])

        Ntotal = 0
        chunk_den=np.zeros([tSample, Nx])
        chunk_flux=np.zeros([tSample, Nx])
        chunk_vx=np.zeros([tSample, Nx])
        chunk_fx=np.zeros([tSample, Nx])

        for i in range(Nx):
            # create a mask to filter only particles in grid cell
            xlo = np.less(fluid_xcoords_sample, xx[i+1, 0, 0])
            xhi = np.greater_equal(fluid_xcoords_sample, xx[i, 0, 0])
            cellx = np.logical_and(xlo, xhi)

            # count particles in cell
            N = np.sum(cellx, axis=1)
            Ntotal += N

            # compute fx
            fx[tSkip:, i, 0, 0] = np.sum(fluid_fx_sample * cellx, axis=1) / N
            # compute vx
            vx[tSkip:, i, 0, 0] = np.sum(fluid_vx_sample * cellx, axis=1) / N
            # compute mass density
            density[tSkip:, i, 0, 0] = N / vol[i, 0, 0]
            # sum  over the vels
            vcmx = np.sum(fluid_vx_sample * cellx, axis=1) / N
            # compute mass flux
            flux[tSkip:, i, 0, 0] = vcmx * density[tSkip:, i, 0, 0]

            chunk_den[:,i]=density[tSkip:,i,0,0]
            chunk_flux[:,i]=flux[tSkip:,i,0,0]
            chunk_vx[:,i]=vx[tSkip:,i,0,0]
            chunk_fx[:,i]=fx[tSkip:,i,0,0]

        density=np.mean(chunk_den,axis=0)*mf/(sci.N_A*(ang_to_cm**3))
        flux=np.mean(chunk_flux,axis=0)*(ang_to_m/fs_to_ns)*mf/(sci.N_A*(ang_to_m**3))
        vx=np.mean(chunk_vx,axis=0)*A_per_fs_to_m_per_s
        fx=np.mean(chunk_fx,axis=0)*kcalpermolA_to_N

        length=np.arange(dx[0]/2.0,xlength,dx[0])
        length/=10 # nm

        np.savetxt('densityX-nc.txt', np.c_[length,density],
                delimiter="  ",header="length           rho")

        np.savetxt('fluxX-nc.txt', np.c_[length,flux],
                delimiter="  ",header="length           jx")

        np.savetxt('vxX-nc.txt', np.c_[length,vx],
                delimiter="  ",header="length           vx")

        np.savetxt('fxX-nc.txt', np.c_[length,fx],
                delimiter="  ",header="length           fx")

    def chunkZ():

        density = np.empty([tSteps+1, Nx, Ny, Nz])
        flux = np.empty([tSteps+1, Nx, Ny, Nz])
        vx = np.empty([tSteps+1, Nx, Ny, Nz])
        fx = np.empty([tSteps+1, Nx, Ny, Nz])

        Ntotal = 0
        chunk_den=np.zeros([tSample, Nz])
        chunk_flux=np.zeros([tSample, Nz])
        chunk_vx=np.zeros([tSample, Nz])
        chunk_fx=np.zeros([tSample, Nz])

        for i in range(Nz):
            #create a mask to filter only particles in grid cell
            zlo = np.less_equal(fluid_zcoords_sample, zz[0, 0, i+1])
            zhi = np.greater_equal(fluid_zcoords_sample, zz[0, 0, i])
            xlo = np.less_equal(fluid_xcoords_sample, xx[1, 0, 0])
            xhi = np.greater_equal(fluid_xcoords_sample, xx[0, 0, 0])

            cellz1 = np.logical_and(zlo, zhi)
            cellz2 = np.logical_and(xlo, xhi)
            cellz = np.logical_and(cellz1,cellz2)

            # count particles in cell
            N = np.sum(cellz, axis=1)
            Ntotal += N

            # To avoid averaging over zero
            Nzero = np.less(N, 1)
            N[Nzero] = 1

            # compute fx
            fx[tSkip:, 0, 0, i] = np.sum(fluid_fx_sample * cellz, axis=1)/ N
            # # compute vx
            vx[tSkip:, 0, 0, i] = np.sum(fluid_vx_sample * cellz, axis=1)/ N
            # compute mass density
            density[tSkip:, 0, 0, i] = N / vol[0, 0, i]

            # sum  over the vels
            vcmx = np.sum(fluid_vx_sample * cellz, axis=1) / N

            # compute mass flux
            flux[tSkip:, 0, 0, i] = vcmx * density[tSkip:, 0, 0, i]

            chunk_den[:,i]=density[tSkip:,0,0,i]
            chunk_flux[:,i]=flux[tSkip:,0,0,i]
            chunk_vx[:,i]=vx[tSkip:,0,0,i]
            chunk_fx[:,i]=fx[tSkip:,0,0,i]

        density=np.mean(chunk_den,axis=0)*mf/(sci.N_A*(ang_to_cm**3))
        flux=np.mean(chunk_flux,axis=0)*mf/(sci.N_A*(ang_to_cm**3))
        vx=np.mean(chunk_vx,axis=0)*A_per_fs_to_m_per_s
        fx=np.mean(chunk_fx,axis=0)*kcalpermolA_to_N

        height=np.arange(fluid_posmin[2]+(dz[0,0,0]/2),fluid_posmax[2],dz[0,0,0])
        height/=10 # nm

        np.savetxt('densityZ-nc.txt', np.c_[height,density],
                delimiter="  ",header="height           rho")

        np.savetxt('fluxZ-nc.txt', np.c_[height[1:-1],flux[1:-1]],
                delimiter="  ",header="height           jx")

        np.savetxt('vxZ-nc.txt', np.c_[height[1:-1],vx[1:-1]],
                delimiter="  ",header="height           vx")

        np.savetxt('fxZ-nc.txt', np.c_[height[1:-1],fx[1:-1]],
                delimiter="  ",header="height           fx")



    # chunkX()
    chunkZ()

    exit()


    # Check atoms velocity distribution ---------------------------

    v=np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)
    # # Evaluate blocks
    # np.transpose(v)
    # n=100   # blocks
    # blocks=bnd(tSample,v,Nf,n)
    # # For first time block
    # avg_block=np.mean(blocks[0])



    def get_stats(arr):
        avg_over_time=np.mean(arr,axis=0)   # Average along time for each atom

        avg_arr=np.mean(avg_over_time)     # Mean of all velocities
        std_dev=np.std(avg_over_time, ddof=1)
        sigma_lower,sigma_upper=avg_arr-(3*std_dev),avg_arr+(3*std_dev)
        dist=norm(avg_arr,std_dev)
        values = [value for value in np.linspace(sigma_lower, sigma_upper, num=50, endpoint=True)]
        probabilities = [dist.pdf(value) for value in values]

        return avg_over_time,values,probabilities

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False)
    #plt.hist(avg_over_time, bins=30,normed=True)
    plt.plot(get_stats(fluid_vx)[1], get_stats(fluid_vx)[2], '-o', label='vx')
    # plt.plot(get_stats(fluid_vy)[1], get_stats(fluid_vy)[2], '-o', label='vy')
    # plt.plot(get_stats(fluid_vz)[1], get_stats(fluid_vz)[2], '-o', label='vz')
    # plt.plot(get_stats(v)[1], get_stats(v)[2], '-o', label='v')
    # plt.axvline(x=avg_a, color='k', linestyle='dashed', label='$\mu$')
    # plt.axvline(x=confidence_intervalL, color='r', linestyle='dashed', label='$2.5^{th}$ percentile')
    # plt.axvline(x=confidence_intervalU, color='r', linestyle='dashed', label='$97.5^{th}$ percentile')
    plt.legend()
    plt.savefig('vel_dist.png', dpi=100)


    exit()


    # AUTOCORRELATION Functions ----------------------------------

    def acf(f_tq):
        n = f_tq.shape[0]
        var = np.var(f_tq, axis=0)
        f_tq -= np.mean(f_tq, axis=0)

        C = np.array([np.sum(f_tq * f_tq, axis=0) if i == 0 else np.sum(f_tq[i:] * f_tq[:-i], axis=0)
                      for i in range(n)]) / (n*var)

        return C

    # dimensions of density_tc
    # dim 0: time
    # dim 1: chunks in x
    # dim 2: chunks in y
    # dim 3: chunks in z

    # Cutoff for the VACF (for figures)
    t_cut=200
    dacf=acf(density_tc[:t_cut,:,0,0])
    dacf_avg = np.mean(dacf,axis=1)

    # Fourier transform of the ACF > Spectrum density
    rho_tq = np.fft.fftn(dacf,axes=(0,1))
    rho_tq_avg = np.mean(rho_tq,axis=1)

    # Inverse DFT
    # rho_itq = np.fft.ifftn(rho_tq,axes=(0,1))
    # rho_itq_avg = np.mean(rho_itq,axis=1)

    # dacf_avg=np.mean(dacf,axis=1)
    # print(dacf_avg)

    # rho_acf=acf(rho_tq)
    # print(rho_acf)

    # faults = np.not_equal(Ntotal, Natoms)
    # count_fault = np.sum(faults)
    # if count_fault > 0:
    #     warnings.warn(f"Wrong number of atoms in {count_fault} time steps")


if __name__ == "__main__":
   proc(sys.argv[1],np.int(sys.argv[2]))
