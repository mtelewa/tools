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
from scipy.stats import pearsonr


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

    def __init__(self, skip, infile_x, infile_z):

        self.skip = skip
        self.infile_x = infile_x
        self.infile_z = infile_z
        self.data_x = netCDF4.Dataset(self.infile_x)
        self.data_z = netCDF4.Dataset(self.infile_z)

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
        # Bulk simulations have Lz dimension
        try:
            self.Lz = dim["Lz"] / 10      # nm
        except KeyError:
            pass

        # Gap height (in each timestep)
        self.h = np.array(self.data_x.variables["Height"])[self.skip:] / 10      # nm
        self.avg_gap_height = np.mean(self.h)
        # COM (in each timestep)
        com = np.array(self.data_x.variables["COM"])[self.skip:] / 10      # nm

        # Number of chunks
        self.Nx = self.data_x.dimensions['x'].size
        Nz = self.data_z.dimensions['z'].size

        # steps
        self.dx = self.Lx/ self.Nx
        # The length and height arrays for plotting
        self.length_array = np.arange(self.dx/2.0, self.Lx, self.dx)   #nm

        if self.avg_gap_height != 0:
            dz = self.avg_gap_height / Nz
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

        else:
            dz = self.Lz / Nz
            self.height_array = np.arange(dz/2.0, self.Lz, dz)     #nm



    def velocity(self):
        """
        Returns
        -------
        height_array_mod : arr
            height with the zero-mean chunks removed
        vx_chunkZ_mod : arr
            velocity along the gap height with the zero-mean chunks removed
        """
        vx = np.array(self.data_z.variables["Vx"])[self.skip:] * A_per_fs_to_m_per_s  # m/s

        vx_t = np.sum(vx, axis=(1,2))
        vx_chunkZ = np.mean(vx, axis=(0,1))

        # Remove chunks with no atoms
        height_array_mod = self.height_array[vx_chunkZ !=0]
        vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]

        return {'height_array_mod': height_array_mod, 'vx_data': vx_chunkZ,
                'vx_chunkZ_mod':vx_chunkZ_mod}

    def fit(self,x,y):
        """
        Returns
        -------
        xdata : arr
            x-axis range for the fitting parabola
        polynom(xdata): floats
            Fitting parameters for the velocity profile
        """
        # Fitting to a parabola
        coeffs_fit = np.polyfit(x, y, 2)     #returns the polynomial coefficients
        # construct the polynomial
        xdata = np.linspace(x[0], x[-1], len(x))
        polynom = np.poly1d(coeffs_fit)

        return {'xdata': xdata, 'fit_data': polynom(xdata)}

    def hydrodynamic(self):

        dd = derive_data(self.skip, self.infile_x, self.infile_z)
        z = dd.height_array
        h = dd.avg_gap_height
        # viscosity near the walls
        mu = dd.viscosity()['mu_eff']
        # slip velocity
        vs = dd.velocity()['vx_chunkZ_mod'][0]
        pgrad = dd.virial()['pGrad']

        v_hydrodynamic = ( (1 / (2 * mu)) * pgrad * (z**2 - h*z) ) + (vs)

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
        dd = derive_data(self.skip, self.infile_x, self.infile_z)
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

        # print('Slip Length {} (nm) -----' .format(Ls_left))
        # Slip velocity according to Navier boundary
        Vs = Ls_left * sci.nano * dd.shear_rate()['shear_rate_walls']                               # m/s
        # print('Slip velocity: Navier boundary {} (m/s) -----'.format(Vs))

        return {'root_left':root_left, 'root_right':root_right, 'Vs':Vs,
                'xdata_left':xdata_left, 'xdata_right':xdata_right,
                'extrapolate_left':extrapolate(xdata_left), 'extrapolate_right':extrapolate(xdata_right)}


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

        values_x, probabilities_x = sq.get_err(fluid_vx)['values'], sq.get_err(fluid_vx)['probs']
        values_y, probabilities_y = sq.get_err(fluid_vy)['values'], sq.get_err(fluid_vy)['probs']

        return {'vx_values': values_x, 'vx_prob': probabilities_x,
                'vy_values': values_y, 'vy_prob': probabilities_y}


    def mflux(self, mf):
        """
        Returns
        -------
        jx_chunkX : mass flux along the box length
        jx_t : mass flux time-series
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
        jx = np.array(self.data_x.variables["Jx"])[self.skip:,1:-1]

        jx_t = np.sum(jx, axis=(1,2)) * (sci.angstrom/fs_to_ns) * (mf/sci.N_A) / (sci.angstrom**3)
        jx_chunkX = np.mean(jx, axis=(0,2)) * (sci.angstrom/fs_to_ns) * (mf/sci.N_A) / (sci.angstrom**3)
        # jx_chunkX_mod = jx_chunkX[jx_chunkX !=0]

        return {'jx_chunkX': jx_chunkX, 'jx_t': jx_t, 'jx_stable': jx_stable,
                'mflowrate_stable':mflowrate_stable}


    def density(self, mf):
        """
        Returns
        -------
        den_chunkX : Avg. bulk density along the length
        den_chunkZ : Avg. fluid density along the height
        den_t : Avg. fluid density with time.
        """

        # Bulk Density ---------------------
        density_Bulk = np.array(self.data_x.variables["Density_Bulk"])[self.skip:,1:-1] * (mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
        den_chunkX = np.mean(density_Bulk, axis=0)

        # Fluid Density ---------------------
        density = np.array(self.data_z.variables["Density"])[self.skip:] * (mf/sci.N_A) / (ang_to_cm**3)
        den_chunkZ = np.mean(density,axis=(0,1))     # g/cm^3

        den_t = np.mean(density,axis=(1,2))    # g/cm^3

        return {'den_chunkX': den_chunkX, 'den_chunkZ': den_chunkZ, 'den_t': den_t}

    def virial(self, pump_size=0.2):
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
        # if self.avg_gap_height != 0:
        #     chunk_vol = self.dx * 10 * self.Ly * 10 * np.mean(self.avg_bulk_height) * 10     # A^3
        # else:
        #     chunk_vol = self.dx * 10 * self.Ly * 10 * self.Lz * 10     # A^3

        # Remove first and last chunks in the x-direction
        vir = np.array(self.data_x.variables["Virial"])[self.skip:,] * \
                    sci.atm * pa_to_Mpa
        vir_x = np.array(self.data_x.variables["Virial"])[self.skip:,1:-1] *  \
                    sci.atm * pa_to_Mpa #/ (3 * chunk_vol)
        vir_z = np.array(self.data_z.variables["Virial"])[self.skip:] * \
                    sci.atm * pa_to_Mpa #/ (3 * chunk_vol)

        vir_t = np.sum(vir, axis=(1,2)) / (3 * totVi)
        vir_chunkX = np.mean(vir_x, axis=(0,2))
        vir_chunkZ = np.mean(vir_z, axis=(0,1))

        # pressure gradient ---------------------------------------
        pd_length = self.Lx - pump_size*self.Lx      # nm
        # Virial pressure at the inlet and the outlet of the pump
        out_chunk, in_chunk = np.argmax(vir_chunkX), np.argmin(vir_chunkX)
        # timeseries of the output and input chunks
        vir_out, vir_in = vir_x[:, out_chunk], vir_x[:, in_chunk]
        # Pressure Difference  between inlet and outlet
        pDiff = np.mean(vir_out) - np.mean(vir_in)
        # Pressure gradient in the simulation domain
        pGrad = - pDiff / pd_length       # MPa/nm

        return {'vir_chunkX': vir_chunkX , 'vir_chunkZ': vir_chunkZ,
                'vir_t': vir_t, 'pGrad': pGrad, 'pDiff':pDiff }


    def sigwall(self, pump_size=0.2):
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

        wall_A = self.Lx * self.Ly * 1e-18
        chunk_A =  self.dx * self.Ly * 1e-18
        wall_height = 0.4711    # nm

        # in Newtons (Remove the last chunck)
        fx_Upper = np.array(self.data_x.variables["Fx_Upper"])[self.skip:,1:-1] * kcalpermolA_to_N
        fy_Upper = np.array(self.data_x.variables["Fy_Upper"])[self.skip:,1:-1] * kcalpermolA_to_N
        fz_Upper = np.array(self.data_x.variables["Fz_Upper"])[self.skip:,1:-1] * kcalpermolA_to_N
        fx_Lower = np.array(self.data_x.variables["Fx_Lower"])[self.skip:,1:-1] * kcalpermolA_to_N
        fy_Lower = np.array(self.data_x.variables["Fy_Lower"])[self.skip:,1:-1] * kcalpermolA_to_N
        fz_Lower = np.array(self.data_x.variables["Fz_Lower"])[self.skip:,1:-1] * kcalpermolA_to_N

        fx_wall = 0.5 * (fx_Upper + fx_Lower)
        fy_wall = 0.5 * (fy_Upper - fy_Lower)
        fz_wall = 0.5 * (fz_Upper - fz_Lower)

        sigxz_t = np.sum(fx_wall,axis=1) * pa_to_Mpa / wall_A
        sigyz_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / wall_A
        sigzz_t = np.sum(fz_wall,axis=1) * pa_to_Mpa / wall_A

        sigxy_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / (wall_height * self.Lx * 1e-18)

        sigxz_chunkX = np.mean(fx_wall,axis=0) * pa_to_Mpa / chunk_A
        sigyz_chunkX = np.mean(fy_wall,axis=0) * pa_to_Mpa / chunk_A
        sigzz_chunkX = np.mean(fz_wall,axis=0) * pa_to_Mpa / chunk_A

        # pressure gradient ----------------------------
        pd_length = self.Lx - pump_size*self.Lx      # nm

        out_chunk, in_chunk = np.argmax(sigzz_chunkX), np.argmin(sigzz_chunkX)

        upper_out, lower_out = fz_Upper[:,out_chunk] * pa_to_Mpa / chunk_A ,\
                               fz_Lower[:,out_chunk] * pa_to_Mpa / chunk_A
        upper_in, lower_in = fz_Upper[:,in_chunk] * pa_to_Mpa / chunk_A ,\
                             fz_Lower[:,in_chunk] * pa_to_Mpa / chunk_A

        sigzz_wall_out, sigzz_wall_in = 0.5 * (upper_out - lower_out), \
                                        0.5 * (upper_in - lower_in)

        pDiff = np.mean(sigzz_wall_out) - np.mean(sigzz_wall_in)
        pGrad = - pDiff / pd_length       # MPa/nm

        return {'sigxz_chunkX':sigxz_chunkX, 'sigzz_chunkX':sigzz_chunkX,
                'sigxz_t':sigxz_t, 'sigyz_t':sigyz_t, 'sigzz_t':sigzz_t,
                'sigxy_t':sigxy_t, 'pDiff':pDiff}


    def temp(self):
        """
        Returns
        -------
        tempX : Avg. Shear stress in the walls along the length
        tempZ : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        temp_x = np.array(self.data_x.variables["Temperature"])[self.skip:]
        temp_z = np.array(self.data_z.variables["Temperature"])[self.skip:]

        temp_t = np.mean(temp_x,axis=(1,2))
        tempX = np.mean(temp_x,axis=(0,2))
        tempZ = np.mean(temp_z,axis=(0,1))

        return {'tempX':tempX, 'tempZ':tempZ, 'temp_t':temp_t}

    def shear_rate(self):

        dd = derive_data(self.skip, self.infile_x, self.infile_z)
        vels = dd.velocity()

        coeffs_fit = np.polyfit(vels['height_array_mod'], vels['vx_chunkZ_mod'], 2)

        # Nearest point to the wall (to evaluate the shear rate at (For Posieuille flow))
        z_wall = vels['height_array_mod'][0]
        # In the bulk to measure viscosity
        z_bulk = vels['height_array_mod'][30]
        # Centerline velocity (Poiseuille flow)
        # Uc = np.max(vels['vx_data'])
        # velocity at the wall for evaluating slip
        # vx_wall = vels['vx_data'][1]
        # Shear Rate is the derivative of the parabolic fit at the wall
        shear_rate_walls = funcs.quad_slope(z_wall, coeffs_fit[0], coeffs_fit[1]) * 1e9      # S-1
        # and at a height in the bulk
        shear_rate_bulk = funcs.quad_slope(z_bulk, coeffs_fit[0], coeffs_fit[1]) * 1e9      # S-1

        shear_rate_profile = (funcs.quad_slope(vels['height_array_mod'][:],
                                coeffs_fit[0], coeffs_fit[1]) * 1e9)      # S-1

        print(f"Shear rate at the walls is {shear_rate_walls:e} (s^-1) -----  \
              \nShear rate in the bulk is {shear_rate_bulk:e} (s^-1) -----")

        return {'shear_rate_walls':shear_rate_walls, 'shear_rate_bulk':shear_rate_bulk,
                'shear_rate_profile':shear_rate_profile}


    def viscosity(self):
        """
        Returns
        -------
        mu: float
            Dynamic viscosity
        """

        dd = derive_data(self.skip, self.infile_x, self.infile_z)

        sigxz_avg = np.mean(dd.sigwall()['sigxz_t'])
        mu_walls = sigxz_avg * 1e9 / dd.shear_rate()['shear_rate_walls']            # mPa.S
        mu_bulk = sigxz_avg * 1e9 / dd.shear_rate()['shear_rate_bulk']            # mPa.S
        mu_eff = (mu_walls + mu_bulk) / 2.

        print(f'Viscosity at the walls is {mu_walls} mPa.s ----- \
                \nViscosity in the bulk is {mu_bulk} mPa.s ----- \
                \nEffective viscosity is {mu_eff} mPa.s -----')

        return {'mu_walls':mu_walls, 'mu_bulk':mu_bulk, 'mu_eff':mu_eff}



    def green_kubo(self, cutoff=-1):

        # Get the viscosity from Green-Kubo
        dd = derive_data(self.skip, self.infile_x, self.infile_z)

        sigxy_t = dd.sigwall()['sigxy_t'][:cutoff] * 1e6
        sigxz_t = dd.sigwall()['sigxz_t'][:cutoff] * 1e6  # cut after the correlation time, here i assume 10 timesteps, units in Pa
        sigyz_t = dd.sigwall()['sigyz_t'][:cutoff] * 1e6  # cut after the correlation time, here i assume 10 timesteps, units in Pa

        temp = np.mean(dd.temp()['temp_t'])         # K
        vol = self.Lx * self.Ly * self.avg_gap_height * 1e-27 # m^3

        c_xy = sq.acf(sigxy_t[:cutoff])['non-norm']
        c_xz = sq.acf(sigxz_t[:cutoff])['non-norm']
        c_yz = sq.acf(sigyz_t[:cutoff])['non-norm']

        v_xy = (vol / (sci.k * temp )) * np.sum(c_xy) * 1e-15 * 1e3  # mPa.s
        v_xz = (vol / (sci.k * temp )) * np.sum(c_xz) * 1e-15 * 1e3  # mPa.s
        v_yz = (vol / (sci.k * temp )) * np.sum(c_yz) * 1e-15 * 1e3  # mPa.s

        viscosity = (1./3.)*(v_xz+v_yz+v_xy)

        print(f'viscosity is {viscosity} mPa.s')

        return {'mu':viscosity}


    def uncertainty(self):

        # chunk_vol = self.dx * 10 * self.Ly * 10 * np.mean(self.avg_bulk_height) * 10     # A^3
        vir_x = np.array(self.data_x.variables["Virial"])[self.skip:,1:-1] *  \
                    sci.atm * pa_to_Mpa# / (3 * chunk_vol)

        chunk_A =  self.dx * self.Ly * 1e-18
        fz_Upper = np.array(self.data_x.variables["Fz_Upper"])[self.skip:,1:-1] * kcalpermolA_to_N
        fz_Lower = np.array(self.data_x.variables["Fz_Lower"])[self.skip:,1:-1] * kcalpermolA_to_N

        # Block sampling of the virial and calculating the uncertainty
        vir_blocks = sq.block_ND(len(self.time), vir_x, self.Nx-2, 100)
        vir_err =  sq.get_err(vir_blocks)['uncertainty']
        # Block sampling of the wall stress and calculating the uncertainty
        fz_Upper_blocks = sq.block_ND(len(self.time), fz_Upper, self.Nx-2, 100)
        fz_Lower_blocks = sq.block_ND(len(self.time), fz_Lower, self.Nx-2, 100)
        fz_Upper_err = sq.get_err(fz_Upper_blocks)['uncertainty']
        fz_Lower_err = sq.get_err(fz_Lower_blocks)['uncertainty']

        # The uncertain in each chunk is the Propagation of uncertain of L and U surfaces
        cov_in_chunk = np.zeros(self.Nx-2)
        for i in range(self.Nx-2):
            cov_in_chunk[i] = np.cov(fz_Upper_blocks[:,i], fz_Lower_blocks[:,i])[0,1]

        fz_err = np.sqrt(fz_Upper_err**2 + fz_Lower_err**2 - 2*cov_in_chunk)
        sigzz_err = fz_err * pa_to_Mpa / chunk_A

        return {'vir_length_err':vir_err, 'sigzz_length_err':sigzz_err}


    def uncertainty_pDiff(self):

        # Virial -----------------
        # Uncertainty of the measured pDiff
        blocks_out, blocks_in = sq.block_1D(vir_out, 100), sq.block_1D(vir_in, 100)

        vir_out_err, vir_in_err = sq.get_err(blocks_out)['uncertainty'], \
                                  sq.get_err(blocks_in)['uncertainty']

        # Propagation of uncertainty
        pDiff_err = np.sqrt(vir_in_err**2 + vir_out_err**2 - 2*np.cov(blocks_out, blocks_in)[0,1])

        # Pearson correlation coefficient (Gaussian distributed variables)
        # Check: https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
        corr, _ = pearsonr(blocks_out, blocks_in)


        # Wall stress ----------------
        blocksU_out, blocksU_in = sq.block_1D(upper_out, 100), sq.block_1D(upper_in, 100)
        blocksL_out, blocksL_in = sq.block_1D(lower_out, 100), sq.block_1D(lower_in, 100)

        sigzzU_out_err, sigzzU_in_err = sq.get_err(blocksU_out)['uncertainty'] ,  \
                                        sq.get_err(blocksU_in)['uncertainty']
        sigzzL_out_err, sigzzL_in_err = sq.get_err(blocksL_out)['uncertainty'] ,  \
                                        sq.get_err(blocksL_in)['uncertainty']

        sig_A, sig_B, sig_C, sig_D = sigzzU_out_err, sigzzU_in_err, sigzzL_out_err, sigzzL_in_err
        covAB = np.cov(blocksU_out, blocksU_in)[0,1]
        covAC = np.cov(blocksU_out, blocksL_out)[0,1]
        covAD = np.cov(blocksU_out, blocksL_in)[0,1]
        covBC = np.cov(blocksU_in, blocksL_out)[0,1]
        covBD = np.cov(blocksU_in, blocksL_in)[0,1]
        covCD = np.cov(blocksL_out, blocksL_in)[0,1]

        pDiff_err = np.sqrt(sig_A**2 + sig_B**2 + sig_C**2 + sig_D**2 - 2*covAB -
                            2*covAC - 2*covAD - 2*covBC - 2*covBD - 2*covCD)

        print(f"The measured pressure difference is {pDiff:.2f} MPa \
        \n with an uncerainty of {pDiff_err} MPa \
        \nThe pressure gradient is {pGrad:.2f} MPa/nm")

        return {'pDiff_err':pDiff_err, 'correlation':corr}




#
