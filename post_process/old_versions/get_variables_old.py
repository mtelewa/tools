#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4
import sys
import os
import re
import sample_quality as sq

import scipy.integrate
# import scipy.special as special

# np.set_printoptions(threshold=sys.maxsize)

ang_to_m = 1e-10
pa_to_Mpa = 1e-6
Kb = 1.38064852e-23 # m2 kg s-2 K-1

def get_data(infile, skip):
    """
    Return plot of time autocorrelation function, for density, longitudinal and transverse current.
    Parameters
    -------
    choice : str
        Choice for autocorrelation function, default="all"
    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object
    ax : matplotlib.axes._suplots.AxesSubplot or array of those
        axes containing the data to plot
    """

    data = netCDF4.Dataset(infile)

    #Variables
    # for varobj in data.variables.keys():
    #     print(varobj)
    # #Global Attributes
    # for name in data.ncattrs():
    #     print("Global attr {} = {}".format(name, getattr(data, name)))

    dim = data.__dict__

    Lx = dim["Lx"]      # A
    Ly = dim["Ly"]      # A
    h = dim["h"]        # A

    matches_before = re.findall("[-+]?\d+", infile.split('_')[-1])
    Nx = np.int(matches_before[0])

    matches_after = re.findall("[-+]?\d+", infile.split('x')[1])
    Nz = np.int(matches_after[0])

    # step in each dimension
    dx = Lx/Nx
    length_array = np.arange(dx/2.0, Lx, dx)
    length_array /= 10           #nm

    pd_length = 0.8 * Lx/10.       # nm

    # Dimensions: axis 0 : time ------------------------------------------------

    time =  np.array(data.variables["Time"])
    time_whole = np.sort(time.flatten())
    if time_whole[-1]> 3e6:
        time =  np.sort(time.flatten())[skip:]

    steps = np.int(time[-1]/1000)

    h = np.array(data.variables["Height"])
    h = h.flatten()[skip:]
    # np.savetxt("h.txt", np.c_[time, h],delimiter="  ",header="time       h")

    avg_gap_height = np.mean(h)
    dz = avg_gap_height / Nz
    height_array = np.arange(dz/2.0, avg_gap_height, dz)
    height_array /= 10          #nm
    # print(len(height_array))

    com = np.array(data.variables["COM"])
    com = com.flatten()[skip:]

    #np.savetxt("com.txt", np.c_[time, com],delimiter="  ",header="time     h")

    totVi = np.array(data.variables["Voronoi_volumes"])
    totVi = totVi.flatten()[skip:]

    mflux_stable = np.array(data.variables["mflux_stable"])
    mflux_stable = mflux_stable.flatten()[skip:]

    mflux_pump = np.array(data.variables["mflux_pump"])
    mflux_pump = mflux_pump.flatten()[skip:]

    mflowrate_stable = np.array(data.variables["mflow_rate_stable"])
    mflowrate_stable = mflowrate_stable.flatten()[skip:]

    # Mass flow rate and flux in the stable region
    avg_mflowrate = np.mean(mflowrate_stable)
    avg_mflux_stable = np.mean(mflux_stable)
    avg_mflux_pump = np.mean(mflux_pump)

    # print('Average mass flux in the stable region is {} g/m2.ns \
    #         \nAverage mass flow rate in the stable region is {} g/ns \
    #         \nAverage mass flux in the pump region is {} g/m2.ns' \
    #        .format(avg_mflux_stable, avg_mflowrate, avg_mflux_pump))

    # Get the bulk height
    surfL_begin = np.array(data.variables["SurfL_begin"])
    surfL_begin = surfL_begin.flatten()[skip:]
    surfU_end = np.array(data.variables["SurfU_end"])
    surfU_end = surfU_end.flatten()[skip:]

    total_box_Height = np.mean(surfU_end) - np.mean(surfL_begin)
    # bulkStartZ, bulkEndZ = 0.4 * total_box_Height, 0.6 * total_box_Height     # 0.4 and 0.6 LJ
    bulkStartZ, bulkEndZ = (0.4 * avg_gap_height) + 4.711, (0.6 * avg_gap_height) + 4.711
    # print(bulkStartZ, bulkEndZ)
    bulkHeight = bulkEndZ - bulkStartZ

    # Dimensions: axis 0 : time , axis 1 : Nx ----------------------------------

    # Bulk Density ---------------------
    density_Bulk = np.array(data.variables["Density_Bulk"])
    density_Bulk = np.reshape(density_Bulk, (len(time_whole),Nx))
    density_Bulk = density_Bulk[skip:]
    bulk_density_avg = np.mean(density_Bulk, axis=(0,1))
    den_chunkX = np.mean(density_Bulk, axis=0)

    # Wall Stresses ------------------
    fx_Upper = np.array(data.variables["Fx_Upper"])
    fx_Upper = np.reshape(fx_Upper, (len(time_whole),Nx))
    fx_Upper = fx_Upper[skip:]
    fy_Upper = np.array(data.variables["Fy_Upper"])
    fy_Upper = np.reshape(fy_Upper, (len(time_whole),Nx))
    fy_Upper = fy_Upper[skip:]
    fz_Upper = np.array(data.variables["Fz_Upper"])
    fz_Upper = np.reshape(fz_Upper, (len(time_whole),Nx))
    fz_Upper = fz_Upper[skip:]
    fx_Lower = np.array(data.variables["Fx_Lower"])
    fx_Lower = np.reshape(fx_Lower, (len(time_whole),Nx))
    fx_Lower = fx_Lower[skip:]
    fy_Lower = np.array(data.variables["Fy_Lower"])
    fy_Lower = np.reshape(fy_Lower, (len(time_whole),Nx))
    fy_Lower = fy_Lower[skip:]
    fz_Lower = np.array(data.variables["Fz_Lower"])
    fz_Lower = np.reshape(fz_Lower, (len(time_whole),Nx))
    fz_Lower = fz_Lower[skip:]

    fx_wall = 0.5 * (fx_Upper - fx_Lower)
    fx_wall_avg = np.mean(fx_wall, axis=(0,1))

    fy_wall = 0.5 * (fy_Upper - fy_Lower)
    fz_wall = 0.5 * (fz_Upper - fz_Lower)

    sigxz_t = np.sum(fx_Upper,axis=1) * pa_to_Mpa / (Lx * Ly * 1e-20)
    avg_sigxz_t = np.mean(sigxz_t)

    sigxz_chunkX = np.mean(fx_Upper,axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)


    # Get the viscosity from Green-Kubo
    # blocks_tau_xz = sq.block_1D_arr(sigxz_t,10)
    # n_array = np.arange(1, len(blocks_tau_xz)+1, 1)

    # sigxz_t_pa = np.sum(fx_Upper,axis=1) / (Lx * Ly * 1e-20)
    # vol = Lx*Ly*avg_gap_height*1e-30
    # T = 300
    # viscosity = (vol/(Kb*T)) * np.trapz(sq.acf(sigxz_t_pa[:10]), time[:10])
    # np.savetxt("tau_acf.txt", np.c_[time[:10], sq.acf(sigxz_t_pa[:10])],delimiter="  ",header="time       var")


    sigzz_t = np.sum(fz_wall,axis=1) * pa_to_Mpa / (Lx * Ly * 1e-20)
    sigzz_chunkX = np.mean(fz_Upper,axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)

    # np.savetxt("sigzz.txt", np.c_[length_array[1:-1], sigzz_chunkX[1:-1]],delimiter="  ",header="Length(nm)       var")

    nChunks = 4
    sigzz_chunkXi = np.zeros([nChunks,len(sigzz_chunkX)])

    # x = 0
    # iter = 10
    # for i in range(nChunks):
    #     sigzz_chunkXi[i,:] = np.mean(fz_wall[x:(i+1) * iter], axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)
    #     x += iter
    # sigzz_chunkXi[0,:] = np.mean(fz_wall[0:100], axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)
    # sigzz_chunkXi[1,:] = np.mean(fz_wall[1000:1100], axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)
    # sigzz_chunkXi[2,:] = np.mean(fz_wall[2000:2100], axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)
    # sigzz_chunkXi[3,:] = np.mean(fz_wall[3000:3100], axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)
    # sigzz_chunkXi[4,:] = np.mean(fz_wall, axis=0) * pa_to_Mpa / (dx * Ly * 1e-20)

    # Dimensions: axis 0 : time , axis 1 : Nx , axis 2 : Nz --------------------

    # Density ---------------------
    density = np.array(data.variables["Density"])[skip:]

    den_t = np.mean(density,axis=(1,2))
    den_chunkZ = np.mean(density,axis=(0,1))


    # Temperature ------------------
    if 'no-temp' not in sys.argv:
        temp = np.array(data.variables["Temperature"])[skip:]
        temp_t = np.sum(temp,axis=(1,2))
        # print(temp_t)
        tempX = np.mean(temp,axis=(0,2))
        # print(tempX)
        # # TODO:  FIX tempZ
        tempZ = np.mean(temp,axis=(0,1))
        # print(tempZ)

    # Velocity -------------------------------
    vx = np.array(data.variables["Vx"])[skip:]

    vx_t = np.sum(vx, axis=(1,2))
    vx_chunkZ = np.mean(vx, axis=(0,1))
    remove_chunks = np.where(vx_chunkZ == 0)[0]

    height_array_mod = np.delete(height_array, remove_chunks)
    vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]

    # np.savetxt("vx.txt", np.c_[height_array_mod[1:-1], vx_chunkZ_mod[1:-1]],delimiter="  ",header="height(nm)       var")

    # Velocity distribution
    # fluid_vx = np.array(data.variables["Fluid_Vx"])
    # fluid_vx = np.reshape(fluid_vx,(steps, 14700))[skip:] / (5 * 35000)
    # values, probabilities = sq.get_err(fluid_vx)


    #vx_t = np.sum(fluid_vx, axis=(1,2))
    # exit()
    # vx_chunkZ = np.mean(vx, axis=(0,1))
    # remove_chunks = np.where(vx_chunkZ == 0)[0]
    #
    # height_array_mod = np.delete(height_array, remove_chunks)
    # vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]

    # Mass flux (whole simulation domain)
    jx = np.array(data.variables["Jx"])[skip:]

    jx_t = np.sum(jx, axis=(1,2))
    jx_chunkZ = np.mean(jx, axis=(0,1))
    # remove_chunks_j = np.where(jx_chunkZ == 0)[0]
    # height_array_mod = np.delete(height_array, remove_chunks_j)
    jx_chunkZ_mod = jx_chunkZ[jx_chunkZ !=0]

    # Virial Pressure ---------------------
    if 'no-virial' not in sys.argv:
        vir = np.array(data.variables["Virial"])[skip:]

        chunk_vol = 3 * dx * Ly * bulkHeight
        bulk_voro_vol = 3 * totVi

        # print(bulkHeight)

        vir_t = np.sum(vir, axis=(1,2)) / bulk_voro_vol
        vir_chunkX = np.mean(vir, axis=(0,2)) / chunk_vol

        vir_chunkXi = np.zeros([nChunks,len(vir_chunkX)])

        if 'pgrad' in sys.argv:
            vir_out = np.array(data.variables["Virial"])[skip:,29] / chunk_vol        # 29
            blocks_out = sq.block_1D_arr(vir_out, 100)
            vir_out_err = sq.get_err(blocks_out)[2]
            vir_out = np.mean(vir_out)

            vir_in = np.array(data.variables["Virial"])[skip:,-2] / chunk_vol
            blocks_in = sq.block_1D_arr(vir_in, 100)
            vir_in_err = sq.get_err(blocks_in)[2]
            vir_in = np.mean(vir_in)

            pDiff = vir_out - vir_in
            # print(pDiff)
            pDiff_err = np.sqrt(vir_in_err**2+vir_out_err**2)

            print('The measured pressure difference is %g MPa with an error of %g MPa' %(pDiff,pDiff_err))

            pGrad = pDiff / pd_length       # MPa/nm
            # print(pGrad)

        else:
            pGrad = 1

    else:
        vir_chunkX, vir_t = [], []
        pGrad = []

        # np.savetxt("vir.txt", np.c_[length_array, vir_chunkX],delimiter="  ",header="Length(nm)       var")

        # vir_chunkXi[0,:] = np.mean(vir[0:100], axis=(0,2)) / (3 * dx * Ly * bulkHeight)
        # vir_chunkXi[1,:] = np.mean(vir[1000:1100], axis=(0,2)) / (3 * dx * Ly * bulkHeight)
        # vir_chunkXi[2,:] = np.mean(vir[2000:2100], axis=(0,2)) / (3 * dx * Ly * bulkHeight)
        # vir_chunkXi[3,:] = np.mean(vir[3000:3100], axis=(0,2)) / (3 * dx * Ly * bulkHeight)

    # vir_chunkX = 0
    # vir_chunkXi = 0

        if len(height_array_mod)>136:
            height_array_mod = height_array_mod[:-1]
            vx_chunkZ_mod = vx_chunkZ_mod[:-1]


#               0           1           2               3               4               5
    return length_array, vir_chunkX, sigzz_chunkX, sigzz_chunkXi, height_array_mod, vx_chunkZ_mod, \
            den_chunkZ, den_chunkX, vir_chunkXi, height_array, avg_gap_height, avg_mflux_stable, \
            avg_sigxz_t, fx_wall_avg, pGrad, time, h, mflux_stable, vir_t, jx_chunkZ_mod, jx_chunkZ_mod, jx_chunkZ_mod, \
            den_t, vir_t, sigxz_t, jx_t, avg_mflowrate, Ly, bulk_density_avg


if __name__ == "__main__":
    get_data(sys.argv[1], np.int(sys.argv[2]))




#a = np.count_nonzero(surfU_xcoords[0])
# b = np.where(totVi == 0)[0]
# print(b)

# Block standard Error
# sq.get_bse(mflux_stable)
# np.savetxt('bse.txt', np.c_[sq.get_bse(mflux_stable)[0],sq.get_bse(mflux_stable)[1]],
#         delimiter="  ",header="n            bse")

# Auto-correlation function
# jacf = sq.acf(mflux_stable)
# jacf_avg = np.mean(jacf,axis=1)
# # Fourier transform of the ACF > Spectrum density
# j_tq = np.fft.fft(jacf)
# j_tq_avg = np.mean(rho_tq,axis=1)

# Inverse DFT
# rho_itq = np.fft.ifftn(rho_tq,axes=(0,1))
# rho_itq_avg = np.mean(rho_itq,axis=1)
