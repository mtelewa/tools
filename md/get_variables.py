#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input:
------
NetCDF file with AMBER convention (after `cdo mergetime` or `cdo merge` on the time slices).
Output:
-------
NetCDF file(s) with AMBER convention. Each file represents a timeslice
Time slices are then merged with `cdo mergetime` or `cdo merge` commands.
"""

import numpy as np
import netCDF4
import sys, os, re
import sample_quality as sq
import funcs
import scipy.constants as sci
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline


if 'lj' in sys.argv:
    mf, A_per_molecule = 39.948, 1
elif 'propane' in sys.argv:
    mf, A_per_molecule = 44.09, 3
elif 'pentane' in sys.argv:
    mf, A_per_molecule = 72.15, 5
elif 'heptane' in sys.argv:
    mf, A_per_molecule = 100.21, 7
else:
    raise NameError('The fluid is not defined!')

# print(mf)

# Converions
ang_to_cm = 1e-8
A_per_fs_to_m_per_s = 1e5
g_to_kg = 1e-3
pa_to_Mpa = 1e-6
kcalpermolA_to_N = 6.947694845598684e-11
fs_to_ns = 1e-6


class derive_data:
    """
    Compute the radial distribution function (RDF) of a MD trajectory.
    Take the average of RDFs of <Nevery>th time frame.
    Parameters
    ----------
    traj : str
        Filename of the netCDF4 trajectory (AMBER format)
    Nevery : int
        Frequency of time frames to compute the RDF (the default is 50).
    Nbins : int
        Number of bins in radial direction (the default is 100).
    maxDist : float
        Maximum distance (the default is 5.).
    Returns
    -------
    numpy.ndarray
        Array containing distance (r) and RDF values (g(r))
    """

    def __init__(self, infile_x, infile_z, skip):

        self.infile_x = infile_x
        self.infile_z = infile_z
        self.data_x = netCDF4.Dataset(self.infile_x)
        self.data_z = netCDF4.Dataset(self.infile_z)
        self.skip = skip

        # # Variables
        # for varobj in self.data_x.variables.keys():
        #     print(varobj)
        # # Global Attributes
        # for name in self.data_x.ncattrs():
        #     print("Global attr {} = {}".format(name, getattr(self.data_x, name)))

        # Time
        self.time =  np.array(self.data_x.variables["Time"])[self.skip:]
        if not self.time.size:
            raise ValueError('The array is empty! Reduce the skipped timesteps.')


        # Box spatial dimensions ------------------------------------------
        dim = self.data_x.__dict__
        self.Lx = dim["Lx"] / 10      # nm
        self.Ly = dim["Ly"] / 10      # nm
        # Gap height (in each timestep)
        self.h = np.array(self.data_x.variables["Height"])[self.skip:] / 10      # nm
        self.avg_gap_height = np.mean(self.h)
        # COM (in each timestep)
        com = np.array(self.data_x.variables["COM"])[self.skip:] / 10      # nm

        # Number of chunks
        Nx = self.data_x.dimensions['x'].size
        Nz = self.data_z.dimensions['z'].size

        # steps
        self.dx = self.Lx/ Nx
        dz = self.avg_gap_height / Nz

        # The length and height arrays for plotting
        self.length_array = np.arange(self.dx/2.0, self.Lx, self.dx)   #nm
        self.height_array = np.arange(dz/2.0, self.avg_gap_height, dz)     #nm

        # If the bulk height is given
        try:
            self.bulk_height = np.array(self.data_x.variables["Bulk_Height"])[self.skip:] / 10
            self.avg_bulk_height = np.mean(self.bulk_height)
            dz_bulk = self.avg_bulk_height / Nz

            bulkStart = self.bulk_height[0] + (dz_bulk/2.0)
            self.bulk_height_array = np.arange(bulkStart, self.bulk_height[0]+self.avg_bulk_height , dz_bulk)     #nm
        except KeyError:
            pass


    def velocity(self):
        """
        Returns
        -------
        height_array_mod : arr
            height with the zero-mean chunks removed
        vx_chunkZ_mod : arr
            velocity along the gap height with the zero-mean chunks removed
        xdata : arr
            x-axis range for the fitting parabola
        polynom(xdata): floats
            Fitting parameters for the velocity profile
        """
        vx = np.array(self.data_z.variables["Vx"])[self.skip:] * A_per_fs_to_m_per_s  # m/s

        vx_t = np.sum(vx, axis=(1,2))
        vx_chunkZ = np.mean(vx, axis=(0,1))

        # Remove chunks with no atoms
        height_array_mod = self.height_array[vx_chunkZ !=0]
        vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]

        # Fitting to a parabola
        coeffs_fit = np.polyfit(height_array_mod, vx_chunkZ_mod, 2)     #returns the polynomial coefficients
        # construct the polynomial
        polynom = np.poly1d(coeffs_fit)
        xdata = np.linspace(height_array_mod[0], height_array_mod[-1], len(height_array_mod))

        return {'height_array_mod': height_array_mod, 'vx_data': vx_chunkZ,
                'xdata': xdata, 'fit_data': polynom(xdata)}

    def hydrodynamic(self):

        dd = derive_data(self.infile_x, self.infile_z, self.skip)
        z = dd.velocity()['height_array_mod']
        h = dd.avg_gap_height
        mu = dd.viscosity()['mu']
        vs = dd.velocity()['vx_data'][0]
        pgrad = dd.virial()['pGrad']

        v_hydrodynamic = ( (1 / (2 * mu)) * -pgrad * (z**2 - h*z) ) + vs

        return {'v_hydro': v_hydrodynamic}

    def slip_length(self):
        """
        Returns
        -------
        Ls: int
            slip length in nm (obtained from extrapolation of the height at velocty of zero)
        Vs : int
            slip velocity in m/s (Slip velocty * Shear rate)
        xdata: arr
            Positions to inter/extrapolate
        extrapolate: arr
            Extrapolated data
        """
        dd = derive_data(self.infile_x, self.infile_z,self.skip)
        vels = dd.velocity()

        npoints = len(vels['height_array_mod'])
        # Positions to inter/extrapolate<
        xdata_left = np.linspace(-100, vels['height_array_mod'][0], npoints)
        xdata_right = np.linspace(vels['height_array_mod'][-1], 100 , npoints)

        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
        # do inter/extrapolation
        extrapolate = InterpolatedUnivariateSpline(vels['height_array_mod'], vels['fit_data'], k=order)
        coeffs_extrapolate_left = np.polyfit(xdata_left, extrapolate(xdata_left), 1)
        coeffs_extrapolate_right = np.polyfit(xdata_right, extrapolate(xdata_right), 1)

        # Slip lengths (extrapolated)
        root_left = np.roots(coeffs_extrapolate_left)
        root_right = np.roots(coeffs_extrapolate_right)

        # Slip length is the extrapolated length in addition to the small
        # length where there are no atoms
        Ls_left = np.abs(root_left) + vels['height_array_mod'][0]
        Ls_right = np.abs(root_right - self.avg_gap_height) + vels['height_array_mod'][0]

        # Check that slip lengths are the same at both walls
        np.testing.assert_allclose(Ls_left, Ls_right, atol=0.1)

        print('Slip Length {} (nm) -----' .format(Ls_left))
        # Slip velocity according to Navier boundary
        Vs = Ls_left * sci.nano * dd.shear_rate()['shear_rate']                               # m/s
        print('Slip velocity: Navier boundary {} (m/s) -----'.format(Vs))

        return {'root_left':root_left, 'root_right':root_right, 'Vs':Vs,
                'xdata_left':xdata_left, 'xdata_right':xdata_right,
                'extrapolate_left':extrapolate(xdata_left), 'extrapolate_right':extrapolate(xdata_right)}

    def viscosity(self):
        """
        Returns
        -------
        mu: float
            Dynamic viscosity
        """
        dd = derive_data(self.infile_x, self.infile_z, self.skip)
        sigxz_avg = np.mean(dd.sigwall()['sigxz_t'])
        mu = sigxz_avg * 1e9 / dd.shear_rate()['shear_rate']            # mPa.S
        mu_bulk = sigxz_avg * 1e9 / dd.shear_rate()['shear_rate_bulk']            # mPa.S

        print('Viscosity is {} (mPa.s) -----'.format(mu))

        return {'mu':mu, 'mu_bulk':mu_bulk}

    def vel_distrib(self):
        """
        Returns
        -------
        values: arr
            fluid velocities averaged over time for each atom/molecule
        probabilities : arr
            probabilities of these velocities
        """
        fluid_vx = np.array(self.data_x.variables["Fluid_Vx"]) *  A_per_fs_to_m_per_s       # m/s
        fluid_vy = np.array(self.data_x.variables["Fluid_Vy"]) *  A_per_fs_to_m_per_s       # m/s

        values_x, probabilities_x = sq.get_err(fluid_vx)[0], sq.get_err(fluid_vx)[1]
        values_y, probabilities_y = sq.get_err(fluid_vy)[0], sq.get_err(fluid_vy)[1]

        return {'vx_values': values_x, 'vx_prob': probabilities_x,
                'vy_values': values_y, 'vy_prob': probabilities_y}


    def mflux(self):
        """
        Returns
        -------
        jx_chunkX : height array with the zero-mean chunks removed.
        jx_t : velocity along the gap height with the zero-mean chunks
                        removed.
        """

        jx_stable = np.array(self.data_x.variables["mflux_stable"])[self.skip:]
        avg_jx_stable = np.mean(jx_stable)

        mflowrate_stable = np.array(self.data_x.variables["mflow_rate_stable"])[self.skip:]
        avg_mflowrate_stable = np.mean(mflowrate_stable)

        jx_pump = np.array(self.data_x.variables["mflux_pump"])[self.skip:]
        avg_jx_pump = np.mean(jx_pump)

        mflowrate_pump = np.array(self.data_x.variables["mflow_rate_pump"])[self.skip:]
        avg_mflowrate_pump = np.mean(mflowrate_pump)

        print(f'Average mass flux in the stable region is {avg_jx_stable} g/m2.ns \
              \nAverage mass flow rate in the stable region is {avg_mflowrate_stable} g/ns \
              \nAverage mass flux in the pump region is {avg_jx_pump} g/m2.ns \
              \nAverage mass flow rate in the pump region is {avg_mflowrate_pump} g/m2.ns')

        # Mass flux (whole simulation domain)
        jx = np.array(self.data_x.variables["Jx"])[self.skip:]

        jx_t = np.sum(jx, axis=(1,2)) * (sci.angstrom/fs_to_ns) * (mf/sci.N_A) / (sci.angstrom**3)
        jx_chunkX = np.mean(jx, axis=(0,2)) * (sci.angstrom/fs_to_ns) * (mf/sci.N_A) / (sci.angstrom**3)
        jx_chunkX_mod = jx_chunkX[jx_chunkX !=0]

        return {'jx_chunkX': jx_chunkX_mod, 'jx_t': jx_t, 'jx_stable': jx_stable,
                'mflowrate_stable':mflowrate_stable}


    def density(self):
        """
        Returns
        -------
        den_chunkX : Avg. bulk density along the length
        den_chunkZ : Avg. fluid density along the height
        den_t : Avg. fluid density with time.
        """

        # Bulk Density ---------------------
        density_Bulk = np.array(self.data_x.variables["Density_Bulk"])[self.skip:] * (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
        den_chunkX = np.mean(density_Bulk, axis=0)

        # Fluid Density ---------------------
        density = np.array(self.data_z.variables["Density"])[self.skip:] * (mf/sci.N_A) / (ang_to_cm**3)
        den_chunkZ = np.mean(density,axis=(0,1))     # g/cm^3

        den_t = np.mean(density,axis=(1,2))    # g/cm^3

        return {'den_chunkX': den_chunkX, 'den_chunkZ': den_chunkZ, 'den_t': den_t}

    def virial(self, pump_size):
        """
        Parameters
        ----------
        parameters
        ----------
        pump_size: float
            multiple of the box length that represents the size of the pump

        Returns
        -------
        tempX : Avg. Shear stress in the walls along the length
        tempZ : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        totVi = np.array(self.data_x.variables["Voronoi_volumes"])[self.skip:]
        chunk_vol = self.dx * 10 * self.Ly * 10 * np.mean(self.avg_bulk_height) * 10     # A^3

        vir_x = np.array(self.data_x.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa / (3 * chunk_vol)
        vir_z = np.array(self.data_z.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa / (3 * chunk_vol)

        vir_t = np.sum(vir_x, axis=(1,2)) / (3 * totVi)
        vir_chunkX = np.mean(vir_x, axis=(0,2))
        vir_chunkZ = np.mean(vir_z, axis=(0,1))

        # press-driven region length
        pd_length = self.Lx - pump_size*self.Lx      # nm

        out_chunk = np.argmax(vir_chunkX)
        in_chunk = np.argmin(vir_chunkX)

        # Chunk 29 is at the outlet of the pump
        vir_out = np.array(self.data_x.variables["Virial"])[self.skip:,out_chunk] * sci.atm * pa_to_Mpa / (3 * chunk_vol)
        vir_in = np.array(self.data_x.variables["Virial"])[self.skip:,in_chunk] * sci.atm * pa_to_Mpa / (3 * chunk_vol)

        pDiff = np.mean(vir_out) - np.mean(vir_in)
        pGrad = pDiff / pd_length       # MPa/nm

        blocks_out, blocks_in = sq.block_1D_arr(vir_out, 100), sq.block_1D_arr(vir_in, 100)
        vir_out_err, vir_in_err = sq.get_err(blocks_out)[2], sq.get_err(blocks_in)[2]

        pDiff_err = np.sqrt(vir_in_err**2 + vir_out_err**2)

        print(f"The measured pressure difference is {pDiff:.2f} MPa with an error of {pDiff_err:.2f} MPa.\
        \nThe pressure gradient is {pGrad:.2f} MPa/nm")

        return {'vir_chunkX': vir_chunkX , 'vir_chunkZ': vir_chunkZ,
                'vir_t': vir_t, 'pGrad': pGrad}


    def temp(self):
        """
        Returns
        -------
        tempX : Avg. Shear stress in the walls along the length
        tempZ : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        temp = np.array(self.data_x.variables["Temperature"])[self.skip:]

        temp_t = np.mean(temp,axis=(1,2))
        tempX = np.mean(temp,axis=(0,2))
        tempZ = np.mean(temp,axis=(0,1))

        return tempX, tempZ, temp_t


    def sigwall(self):
        """
        Parameters
        ----------
        F<x|y|z>_<Upper|Lower> : The force
        Returns
        -------
        sigxz_chunkX : Avg. Shear stress in the walls along the length
        sigzz_chunkX : Avg. Normal stress in the walls along the length
        sigxz_t : Avg. shear stress in the walls with time.
        sigzz_t : Avg. normal stress in the walls with time.
        """

        # in Newtons
        fx_Upper = np.array(self.data_x.variables["Fx_Upper"])[self.skip:] * kcalpermolA_to_N
        fy_Upper = np.array(self.data_x.variables["Fy_Upper"])[self.skip:] * kcalpermolA_to_N
        fz_Upper = np.array(self.data_x.variables["Fz_Upper"])[self.skip:] * kcalpermolA_to_N
        fx_Lower = np.array(self.data_x.variables["Fx_Lower"])[self.skip:] * kcalpermolA_to_N
        fy_Lower = np.array(self.data_x.variables["Fy_Lower"])[self.skip:] * kcalpermolA_to_N
        fz_Lower = np.array(self.data_x.variables["Fz_Lower"])[self.skip:] * kcalpermolA_to_N

        fx_wall = 0.5 * (fx_Upper + fx_Lower)
        fy_wall = 0.5 * (fy_Upper - fy_Lower)
        fz_wall = 0.5 * (fz_Upper - fz_Lower)

        wall_A = self.Lx * self.Ly * 1e-18
        chunk_A =  self.dx * self.Ly * 1e-18

        sigxz_t = np.sum(fx_wall,axis=1) * pa_to_Mpa / wall_A
        sigxz_chunkX = np.mean(fx_wall,axis=0) * pa_to_Mpa / chunk_A

        sigzz_t = np.sum(fz_wall,axis=1) * pa_to_Mpa / wall_A
        sigzz_chunkX = np.mean(fz_wall,axis=0) * pa_to_Mpa / chunk_A

        return {'sigxz_chunkX':sigxz_chunkX, 'sigzz_chunkX':sigzz_chunkX,
                'sigxz_t':sigxz_t, 'sigzz_t':sigzz_t}


    def shear_rate(self):

        dd = derive_data(self.infile_x, self.infile_z, self.skip)
        vels = dd.velocity()

        coeffs_fit = np.polyfit(vels['height_array_mod'], vels['vx_data'], 2)

        # Nearest point to the wall (to evaluate the shear rate at (For Posieuille flow))
        z_wall = vels['height_array_mod'][0]
        # In the bulk to measure viscosity
        z_bulk = vels['height_array_mod'][20]
        # Centerline velocity (Poiseuille flow)
        Uc = np.max(vels['vx_data'])
        # velocity at the wall for evaluating slip
        vx_wall = vels['vx_data'][1]
        # Shear Rate is the derivative of the parabolic fit at the wall
        shear_rate = funcs.quad_slope(z_wall, coeffs_fit[0], coeffs_fit[1]) * 1e9      # S-1
        # and at a height in the bulk
        shear_rate_bulk = funcs.quad_slope(z_bulk, coeffs_fit[0], coeffs_fit[1]) * 1e9      # S-1

        print(f"Shear rate at the walls is {shear_rate:e} (s^-1) -----  \
              \nShear rate in the bulk is {shear_rate_bulk:e} (s^-1) -----")

        return {'shear_rate':shear_rate, 'shear_rate_bulk':shear_rate_bulk}

        # if len(height_array_mod) > 136:
        #     height_array_mod = height_array_mod[:-1]
        #     vx_chunkZ_mod = vx_chunkZ_mod[:-1]


if __name__ == "__main__":

    dd = derive_data(sys.argv[-2], sys.argv[-1], np.int(sys.argv[1]))

    if 'viscosity' in sys.argv:
        dd.viscosity()

    if 'mflux' in sys.argv:
        dd.mflux()

    if 'sigzz' in sys.argv:
        dd.sigwall()

    if 'hydro' in sys.argv:
        dd.hydrodynamic()

    if 'rdf' in sys.argv:
        dd.rdf('nvt.nc')

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
