#!/usr/bin/env python

import netCDF4
import sys
import numpy as np
import scipy.constants as sci
import time
from block_data import block_data_profile as bd

zlength = 65    # Angstrom
Boffset = 1     # Angstrom
offset = 5      # Angstrom
h = 40          # Angstrom

def proc(infile,tSkip):

    data = netCDF4.Dataset(infile)

    # # Print global attributes
    # for name in data.ncattrs():
    #     print("Global attr {} = {}".format(name, getattr(data, name)))

    # print("---- Dimensions ------------------")
    # for dimobj in data.dimensions.keys():
    #     print(dimobj)

    # # Print the whole dict
    # for varobj in data.variables.items():
    #     print(varobj)
    #
    # # Print they values
    # for varobj in data.variables.values():
    #     print(varobj)

    # print("----- Variables ------------------")

    # Print the keys
    # for varobj in data.variables.keys():
    #      print(varobj)
    # print(data.variables)

    # # Spatial: the three spatial dimensions (X,Y,Z)
    # spatial_data = np.array(data.variables["spatial"])
    # print(spatial_data)
    #
    # # Cell spatial: the three lengths (a,b,c) that define the size of the unit cell
    # cell_spatial_data = np.array(data.variables["cell_spatial"])
    # print(cell_spatial_data)
    #
    # # Cell angular: the three angles (alpha,beta,gamma) that define the shape of the unit cell.
    # cell_angular = np.array(data.variables["cell_angular"])
    # print(cell_angular)
    #
    # Time (timesteps*timestep_size)
    sim_time = np.array(data.variables["time"])
    print('Total simualtion time: {} {}'.format(int(sim_time[-1]),data.variables["time"].units))
    tSteps = int(sim_time[-1]/3000)
    tSample=tSteps-tSkip
    tSteps_array,tSample_array=[i for i in range (tSteps+1)], \
                               [i for i in range (tSkip,tSample+1)]

    # cell_origin = np.array(data.variables["cell_origin"])
    # print(cell_origin)

    cell_lengths = np.array(data.variables["cell_lengths"])
    # print(cell_lengths.shape)
    print('Block dimensions: {} {} {} {}'.format(cell_lengths[0,0],cell_lengths[0,1],
                                             cell_lengths[0,2], "Angstrom"))
    #
    # cell_angles = np.array(data.variables["cell_angles"])
    # print(cell_angles[0])
    #
    # id = np.array(data.variables["id"])
    # print(id)
    #
    type = np.array(data.variables["type"])

    coords = np.array(data.variables["coordinates"])
    # print(coords)

    xcoords,ycoords,zcoords = coords[:,:,0],coords[:,:,1],coords[:,:,2]

    # if there are walls, zposition is adjusted
    solid_atoms_index=[]
    surfL_atoms_index=[]
    surfU_atoms_index=[]

    for idx,val in enumerate(type[0]):
        if val == 2:
            solid_atoms_index.append(idx)
            if zcoords[0][idx] < np.max(zcoords)/2.:
                surfL_atoms_index.append(idx)
            else:
                surfU_atoms_index.append(idx)

    solid_atoms_index=np.asarray(solid_atoms_index)
    surfL_atoms_index=np.asarray(surfL_atoms_index)
    surfU_atoms_index=np.asarray(surfU_atoms_index)

    #Get the maximum z-coordinate in the lower wall and the min z-coordinate in the Upper
    surfL_pos=[]
    surfU_pos=[]

    for i,j in zip(surfL_atoms_index,surfU_atoms_index):
        surfL_pos.append(zcoords[0][i])
        surfU_pos.append(zcoords[0][j])

    surfL_end=np.max(surfL_pos)
    surfU_begin=np.min(surfU_pos)
    gap_height=surfU_begin-surfL_end

    print(gap_height)


    # PBC correction:
    # Z-axis is non-periodic
    nonperiodic=(2,)
    pdim = [n for n in range(3) if not(n in nonperiodic)]
    hi = np.greater(coords[:, :, pdim], cell_lengths[0][pdim]).astype(int)
    lo = np.less(coords[:, :, pdim], 0.).astype(int)
    coords[:, :, pdim] += (lo - hi) * cell_lengths[0][pdim]

    posmin = np.array([np.min(xcoords),np.min(ycoords),surfL_end])
    posmax = np.array([np.max(xcoords),np.max(ycoords),surfU_begin])
    lengths = posmax - posmin

    print("Min fluid atomic coordinates {}".format(posmin))
    print("Max fluid atomic coordinates {}".format(posmax))

    # start = time.time()
    # posmin = [np.amin(coords, axis=(0, 1))]
    # end = time.time()
    # tot_time = end-start
    # print(tot_time)


    # create grid

    Nx = 48
    Ny = 1
    Nz = 1

    size = np.array([Nx, Ny, Nz])

    bounds = [np.arange(size[i] + 1) / size[i] * lengths[i] + posmin[i] for i in range(3)]
    xx, yy, zz = np.meshgrid(bounds[0], bounds[1], bounds[2])

    xx = np.transpose(xx, (1, 0, 2))
    yy = np.transpose(yy, (1, 0, 2))
    zz = np.transpose(zz, (1, 0, 2))

    dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
    dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
    dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]

    # print(dx[0],dy[0],dz[0])
    vol = dx * dy * dz

    density_tc = np.empty([len(sim_time), Nx, Ny, Nz])
    fluxX_tc = np.empty([len(sim_time), Nx, Ny, Nz])
    fluxY_tc = np.empty([len(sim_time), Nx, Ny, Nz])
    fluxZ_tc = np.empty([len(sim_time), Nx, Ny, Nz])

    Ntotal = 0

    vels = np.array(data.variables["velocities"])
    print(vels.shape)

    forces = np.array(data.variables["forces"])
    # print(forces)

    for i in range(Nx):
    # for j in range(Ny):
    #     for k in range(Nz):

        # create a mask to filter only particles in grid cell
        xlo = np.less(coords[:, :, 0], xx[i + 1, 0, 0])
        xhi = np.greater_equal(coords[:, :, 0], xx[i, 0, 0])
        # ylo = np.less(coords[:, :, 1], yy[i, j + 1, k])
        # yhi = np.greater_equal(coords[:, :, 1], yy[i, j, k])
        # zlo = np.less_equal(coords[:, :, 2], zz[i, j, k + 1])
        # zhi = np.greater_equal(coords[:, :, 2], zz[i, j, k])

        cellx = np.logical_and(xlo, xhi)
        # celly = np.logical_and(ylo, yhi)
        # cellz = np.logical_and(zlo, zhi)
        # cell = np.logical_and(cellx, celly)
        # cell = np.logical_and(cell, cellz)

        # count particles in cell
        N = np.sum(cellx, axis=1)
        Ntotal += N
        # compute mass density
        density_tc[:, i, 0, 0] = N / vol[i, 0, 0]

        # compute COM velocity
        Nzero = np.less(N, 1)
        N[Nzero] = 1


        vcmx_t = np.sum(vels[:, :, 0] * cellx, axis=1) / N
        # vcmy_t = np.sum(vels[:, :, 1] * cell, axis=1) / N
        # vcmz_t = np.sum(vels[:, :, 2] * cell, axis=1) / N

        # compute mass flux
        fluxX_tc[:, i, 0, 0] = vcmx_t * density_tc[:, i, 0, 0]

        # fluxY_tc[:, i, j, k] = vcmy_t * density_tc[:, i, j, k]
        # fluxZ_tc[:, i, j, k] = vcmz_t * density_tc[:, i, j, k]

        #print(density_tc.shape)
        #density=np.transpose(density_tc,(1,0,2))

    print('denshape = {}'.format(density_tc[11:,:,0,0].shape))


    chunk_den_data=np.zeros([tSample, Nx])
    chunk_flux_data=np.zeros([tSample, Nx])
    for i in range (Nx):
        chunk_den_data[:,i]=density_tc[11:,i,0,0]
        chunk_flux_data[:,i]=fluxX_tc[11:,i,0,0]

    # Molar mass of fluid
    mf = 39.948      # in gram/mol

    averaged_den_data=np.mean(bd(tSample,chunk_den_data,Nx),axis=0)
    averaged_flux_data=np.mean(bd(tSample,chunk_flux_data,Nx),axis=0)

    density=averaged_den_data*mf
    fluxX=averaged_flux_data*mf

    length=np.arange(dx[0]/2.0,cell_lengths[0,0],dx[0])
    length/=10
    # print(length)
    # print(len(density))

    np.savetxt('density.txt', np.c_[length,density],
            delimiter="  ",header="length           rho")

    np.savetxt('fluxX.txt', np.c_[length,fluxX],
            delimiter="  ",header="length           jx")

    # Fourier transform
    rho_tq = np.fft.fftn(density_tc[:,:,0,0])
    fluxX_tq = np.fft.fftn(fluxX_tc[:,:,0,0])

    # print(rho_tq)

    # def acf(f_tq):
    #     n = f_tq.shape[0]
    #     var = np.var(f_tq, axis=0)
    #     f_tq -= np.mean(f_tq, axis=0)
    #
    #     C = np.array([np.sum(f_tq * f_tq, axis=0) if i == 0 else np.sum(f_tq[i:] * f_tq[:-i], axis=0)
    #                   for i in range(n)]) / n
    #
    #     return C
    #
    #
    # den_acf=acf(density_tc[:,:,0,0])
    # den_acf_avg=np.mean(den_acf,axis=1)
    # np.savetxt('dacf.txt', np.c_[tSteps_array,den_acf_avg],
    #             delimiter="  ",header="time           DACF")
    #
    # flux_acf=acf(fluxX_tc[:,:,0,0])
    # flux_acf_avg=np.mean(flux_acf,axis=1)
    # np.savetxt('jacf.txt', np.c_[tSteps_array,flux_acf_avg],
    #             delimiter="  ",header="time           JACF")
    #
    #
    # flux_spectrum=acf(fluxX_tq)
    # flux_spectrum_avg=np.mean(flux_spectrum,axis=1)
    # np.savetxt('j_spectrum.txt', np.c_[tSteps_array,flux_spectrum_avg.real],
    #             delimiter="  ",header="time           j_spectrum")

    # faults = np.not_equal(Ntotal, Natoms)
    # count_fault = np.sum(faults)
    # if count_fault > 0:
    #     warnings.warn(f"Wrong number of atoms in {count_fault} time steps")


if __name__ == "__main__":
    proc(sys.argv[1],sys.argv[2])
