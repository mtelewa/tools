#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4
import sys
import os
import re
import sample_quality as sq
import funcs
import scipy.constants as sci

import scipy.integrate
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
# import scipy.special as special

# Converions
ang_to_cm = 1e-8
A_per_fs_to_m_per_s = 1e5
g_to_kg = 1e-3
pa_to_Mpa = 1e-6
fs_to_ns = 1e-6
kcalpermolA_to_N = 6.947694845598684e-11

class derive_data:

    def __init__(self, infile, skip):

        self.infile = infile
        self.data = netCDF4.Dataset(self.infile)
        self.skip = skip

        # # Variables
        # for varobj in self.data.variables.keys():
        #     print(varobj)
        # # Global Attributes
        # for name in self.data.ncattrs():
        #     print("Global attr {} = {}".format(name, getattr(self.data, name)))

        dim = self.data.__dict__

        # print(dim)
        self.Lx = dim["Lx"]      # A
        self.Ly = dim["Ly"]      # A
        # self.Lz = dim["Lz"]      # A

        Nx = self.data.dimensions['x'].size
        Nz = self.data.dimensions['z'].size
        # h = dim["h"]        # A

        # matches_before = re.findall("[-+]?\d+", self.infile.split('_')[-1])
        # Nx = np.int(matches_before[0])
        #
        # matches_after = re.findall("[-+]?\d+", self.infile.split('x')[1])
        # Nz = np.int(matches_after[0])

        # Time
        self.time =  np.array(self.data.variables["Time"])
        self.time_whole = np.sort(self.time.flatten())
        if self.time_whole[-1]> 1e6:
            self.time =  np.sort(self.time.flatten())[self.skip:]

        steps = np.int(self.time[-1]/1000)

        # Spatial dimensions
        self.dx = self.Lx/ Nx
        length_array = np.arange(self.dx/2.0, self.Lx, self.dx)
        self.length_array = length_array / 10           #nm
        # press-driven region length
        self.pd_length = 0.8 * self.Lx / 10.       # nm

        h = np.array(self.data.variables["Height"])
        self.h = h.flatten()[self.skip:]

        avg_gap_height = np.mean(self.h)
        dz = avg_gap_height / Nz

        height_array = np.arange(dz/2.0, avg_gap_height, dz)
        self.height_array = height_array/ 10          #nm

        # Get the bulk height
        # surfL_begin = np.array(self.data.variables["SurfL_begin"])
        # surfL_begin = surfL_begin.flatten()[self.skip:]
        # surfU_end = np.array(self.data.variables["SurfU_end"])
        # surfU_end = surfU_end.flatten()[self.skip:]
        # surfL_end = np.array(data.variables["SurfL_end"])
        # surfL_end = surfL_end.flatten()[skip:]
        # total_box_Height = np.mean(surfU_end) - np.mean(surfL_begin)
        # bulkStartZ, bulkEndZ = 0.4 * total_box_Height, 0.6 * total_box_Height     # 0.4 and 0.6 LJ
        bulkStartZ, bulkEndZ = (0.4 * avg_gap_height) + 4.711, (0.6 * avg_gap_height) + 4.711
        #bulkStartZ, bulkEndZ = (0.4 * avg_gap_height) + surfL_end, (0.6 * avg_gap_height) + surfL_end
        self.bulkHeight = bulkEndZ - bulkStartZ

        com = np.array(self.data.variables["COM"])
        com = com.flatten()[self.skip:]

    # Dimensions: axis 0 : time , axis 1 : Nx , axis 2 : Nz --------------------

    def velocity(self):
        """
        Returns
        -------
        height_array_mod : height array with the zero-mean chunks removed.
        vx_chunkZ_mod : velocity along the gap height with the zero-mean chunks
                        removed.
        """

        vx = np.array(self.data.variables["Vx"])[self.skip:] * A_per_fs_to_m_per_s  # m/s

        vx_t = np.sum(vx, axis=(1,2))
        vx_chunkZ = np.mean(vx, axis=(0,1))

        remove_chunks = np.where(vx_chunkZ == 0)[0]

        height_array_mod = np.delete(self.height_array, remove_chunks)
        height_array_mod = height_array_mod[1:-1]

        vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]
        vx_chunkZ_mod = vx_chunkZ_mod[1:-1]

        # Fitting to a parabola
        coeffs_fit = np.polyfit(height_array_mod, vx_chunkZ_mod, 2)     #returns the polynomial coefficients

        npoints = len(height_array_mod)
        # construct the polynomial
        f = np.poly1d(coeffs_fit)
        # x-data to fit the parabola
        x1 = np.linspace(height_array_mod[0], height_array_mod[-1], npoints)

        return height_array_mod, vx_chunkZ_mod, x1, f(x1)

    def vx_distrib(self):
        #Velocity distribution
        fluid_vx = np.array(self.data.variables["Fluid_Vx"]) *  A_per_fs_to_m_per_s       # m/s
        values, probabilities = sq.get_err(fluid_vx)[0], sq.get_err(fluid_vx)[1]

        return values, probabilities

    def slip_length(self):

        # Positions ro inter/extrapolate
        x2 = np.linspace(-22, height_array_mod[1], npoints)
        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
        # do inter/extrapolation
        extrapolate = InterpolatedUnivariateSpline(height_array_mod, dd.veloctity()[3], k=order)
        coeffs_extrapolate = np.polyfit(x2, extrapolate(x2), 1)
        # Slip lengths (extrapolated)
        roots = np.roots(coeffs_extrapolate)
        Ls = np.abs(roots[-1])*1e-9      #  m
        # Slip velocities according to Navier boundary
        Vs = Ls * shear_rate                                # m/s
        # print('Slip velocity: Navier boundary (m/s) -----')
        # print(Vs)           # m/s
        Ls *= 1e9                                           # nm
        # print('Slip Length (nm) -----')
        # print(Ls)       # nm

        return Ls, Vs, x2, extrapolate(x2)

    def mflux(self):
        """
        Returns
        -------
        height_array_mod : height array with the zero-mean chunks removed.
        vx_chunkZ_mod : velocity along the gap height with the zero-mean chunks
                        removed.
        """

        jx_stable = np.array(self.data.variables["mflux_stable"])
        jx_stable = jx_stable.flatten()[self.skip:]
        avg_jx_stable = np.mean(jx_stable)

        jx_pump = np.array(self.data.variables["mflux_pump"])
        jx_pump =jx_pump.flatten()[self.skip:]
        avg_jx_pump = np.mean(jx_pump)

        mflowrate_stable = np.array(self.data.variables["mflow_rate_stable"])
        mflowrate_stable = mflowrate_stable.flatten()[self.skip:]
        avg_mflowrate = np.mean(mflowrate_stable)

        print('Average mass flux in the stable region is {} g/m2.ns \
             \nAverage mass flow rate in the stable region is {} g/ns \
             \nAverage mass flux in the pump region is {} g/m2.ns' \
             .format(avg_jx_stable, avg_mflowrate, avg_jx_pump))

        # Mass flux (whole simulation domain)
        jx = np.array(self.data.variables["Jx"])[self.skip:] * (sci.angstrom/fs_to_ns) * (mf/sci.N_A) / (sci.angstrom**3)

        jx_t = np.sum(jx, axis=(1,2))
        jx_chunkX = np.mean(jx, axis=(0,2))
        jx_chunkX_mod = jx_chunkX[jx_chunkX !=0]

        return jx_chunkX_mod, jx_t, jx_stable, mflowrate_stable


    def density(self):
        """
        Returns
        -------
        den_chunkX : Avg. bulk density along the length
        den_chunkZ : Avg. fluid density along the height
        den_t : Avg. fluid density with time.
        """

        # Bulk Density ---------------------
        density_Bulk = np.array(self.data.variables["Density_Bulk"]) * (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
        density_Bulk = np.reshape(density_Bulk, (len(self.time_whole),len(self.length_array)))[self.skip:]

        # bulk_density_avg = np.mean(density_Bulk, axis=(0,1))
        den_chunkX = np.mean(density_Bulk, axis=0)

        # Fluid Density ---------------------
        density = np.array(self.data.variables["Density"])[self.skip:] * (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3

        den_t = np.mean(density,axis=(1,2))
        den_chunkZ = np.mean(density,axis=(0,1))

        return den_chunkX, den_chunkZ, den_t


    def sigwall(self):
        """
        Returns
        -------
        sigxz_chunkX : Avg. Shear stress in the walls along the length
        sigzz_chunkX : Avg. Normal stress in the walls along the length
        sigxz_t : Avg. shear stress in the walls with time.
        sigzz_t : Avg. normal stress in the walls with time.
        """

        # in Newtons
        fx_Upper = np.array(self.data.variables["Fx_Upper"])
        fx_Upper = np.reshape(fx_Upper, (len(self.time_whole),len(self.length_array)))[self.skip:]
        fy_Upper = np.array(self.data.variables["Fy_Upper"])
        fy_Upper = np.reshape(fy_Upper, (len(self.time_whole),len(self.length_array)))[self.skip:]
        fz_Upper = np.array(self.data.variables["Fz_Upper"])
        fz_Upper = np.reshape(fz_Upper, (len(self.time_whole),len(self.length_array)))[self.skip:]
        fx_Lower = np.array(self.data.variables["Fx_Lower"])
        fx_Lower = np.reshape(fx_Lower, (len(self.time_whole),len(self.length_array)))[self.skip:]
        fy_Lower = np.array(self.data.variables["Fy_Lower"])
        fy_Lower = np.reshape(fy_Lower, (len(self.time_whole),len(self.length_array)))[self.skip:]
        fz_Lower = np.array(self.data.variables["Fz_Lower"])
        fz_Lower = np.reshape(fz_Lower, (len(self.time_whole),len(self.length_array)))[self.skip:]

        fx_wall = 0.5 * (fx_Upper - fx_Lower)
        fx_wall_avg = np.mean(fx_wall, axis=(0,1))
        fy_wall = 0.5 * (fy_Upper - fy_Lower)
        fy_wall_avg = np.mean(fy_wall, axis=(0,1))
        fz_wall = 0.5 * (fz_Upper - fz_Lower)
        fz_wall_avg = np.mean(fz_wall, axis=(0,1))

        wall_A = self.Lx * self.Ly * 1e-20
        chunk_A =  self.dx * self.Ly * 1e-20

        sigxz_t = np.sum(fx_Upper,axis=1) * pa_to_Mpa / wall_A
        sigxz_chunkX = np.mean(fx_wall,axis=0) * pa_to_Mpa / chunk_A
        # avg_sigxz_t = np.mean(sigxz_t)

        sigzz_t = np.sum(fz_wall,axis=1) * pa_to_Mpa / wall_A
        sigzz_chunkX = np.mean(fz_wall,axis=0) * pa_to_Mpa / chunk_A

        # print(np.mean(fz_wall[:,40])* pa_to_Mpa / chunk_A)
        # print(np.mean(fz_Upper[200:,27]))
        # print(np.mean(fz_wall[200:,27]))
        # print(sigzz_chunkX[27])

        return sigxz_chunkX, sigzz_chunkX, sigxz_t, sigzz_t


    def virial(self):
        """
        Returns
        -------
        tempX : Avg. Shear stress in the walls along the length
        tempZ : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        totVi = np.array(self.data.variables["Voronoi_volumes"])
        totVi = totVi.flatten()[self.skip:]
        bulk_voro_vol = 3 * totVi

        chunk_vol = 3 * self.dx * self.Ly * self.bulkHeight

        vir = np.array(self.data.variables["Virial"])[self.skip:]

        vir_t = np.sum(vir, axis=(1,2)) / bulk_voro_vol
        vir_chunkX = np.mean(vir, axis=(0,2)) / chunk_vol

        # # TODO: The first and last chunk shall not be input
        vir_out = np.array(self.data.variables["Virial"])[self.skip:,2] / chunk_vol
        vir_in = np.array(self.data.variables["Virial"])[self.skip:,-2] / chunk_vol
        pDiff = np.mean(vir_out) - np.mean(vir_in)
        pGrad = pDiff / self.pd_length       # MPa/nm

        blocks_out, blocks_in = sq.block_1D_arr(vir_out, 100), sq.block_1D_arr(vir_in, 100)
        vir_out_err, vir_in_err = sq.get_err(blocks_out)[2], sq.get_err(blocks_in)[2]
        pDiff_err = np.sqrt(vir_in_err**2 + vir_out_err**2)

        print('The measured pressure difference is %g MPa with an error of %g MPa.\
        \n The pressure gradient is %g' %(pDiff, pDiff_err, pGrad))

        return vir_chunkX, vir_t, pGrad


    def temp(self):
        """
        Returns
        -------
        tempX : Avg. Shear stress in the walls along the length
        tempZ : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        temp = np.array(self.data.variables["Temperature"])[self.skip:]

        temp_t = np.sum(temp,axis=(1,2))
        tempX = np.mean(temp,axis=(0,2))
        # # TODO:  FIX tempZ
        tempZ = np.mean(temp,axis=(0,1))

        return tempX, tempZ, temp_t

    def viscosity(self):

        dd = derive_data(self.infile,self.skip)
        sigxz_avg = np.mean(dd.sigwall()[2])
        shear_rate = dd.velocity()[2]
        mu = sigxz_avg * 1e9 / shear_rate            # mPa.S
        # print('Viscosity (mPa.s) -----')
        # print(mu)       # mPa.s
        return mu

    def shear_rate(self):

        dd = derive_data(self.infile,self.skip)

        height_array_mod = dd.velocity()[0]
        vx_chunkZ_mod = dd.velocity()[1]
        coeffs_fit = np.polyfit(height_array_mod, vx_chunkZ_mod, 2)

        # Nearest point to the wall (to evaluate the shear rate at (For Posieuille flow))
        z_wall = height_array_mod[1]
        # In the bulk to measure viscosity
        z_bulk = height_array_mod[20]
        # Centerline velocity (Poiseuille flow)
        Uc = np.max(vx_chunkZ_mod)
        # velocity at the wall for evaluating slip
        vx_wall = vx_chunkZ_mod[1]
        # Shear Rates
        shear_rate = funcs.quad_slope(z_wall,coeffs_fit[0],coeffs_fit[1]) * 1e9      # S-1
        shear_rate_bulk = funcs.quad_slope(z_bulk,coeffs_fit[0],coeffs_fit[1]) * 1e9      # S-1
        # print('Shear rate (s^-1) -----')
        # print(shear_rate)  # s^-1
        return shear_rate, shear_rate_bulk

        # if len(height_array_mod) > 136:
        #     height_array_mod = height_array_mod[:-1]
        #     vx_chunkZ_mod = vx_chunkZ_mod[:-1]


if __name__ == "__main__":

    if 'viscosity' in sys.argv:
        derive_data(sys.argv[-1], np.int(sys.argv[1])).viscosity()

    if 'mflux' in sys.argv:
        derive_data(sys.argv[-1], np.int(sys.argv[1])).mflux()

    if 'sigzz' in sys.argv:
        derive_data(sys.argv[-1], np.int(sys.argv[1])).sigwall()

    # print(len(derive_data(sys.argv[1],5000).length_array))
    # get_data(sys.argv[1], np.int(sys.argv[2]))




# Get the viscosity from Green-Kubo
# blocks_tau_xz = sq.block_1D_arr(sigxz_t,10)
# n_array = np.arange(1, len(blocks_tau_xz)+1, 1)

# sigxz_t_pa = np.sum(fx_Upper,axis=1) / (self.Lx * self.Ly * 1e-20)
# vol = self.Lx*self.Ly*avg_gap_height*1e-30
# T = 300
# viscosity = (vol/(Kb*T)) * np.trapz(sq.acf(sigxz_t_pa[:10]), time[:10])
# np.savetxt("tau_acf.txt", np.c_[time[:10], sq.acf(sigxz_t_pa[:10])],delimiter="  ",header="time       var")






#vx_t = np.sum(fluid_vx, axis=(1,2))
# vx_chunkZ = np.mean(vx, axis=(0,1))
# remove_chunks = np.where(vx_chunkZ == 0)[0]
#
# height_array_mod = np.delete(height_array, remove_chunks)
# vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]


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
