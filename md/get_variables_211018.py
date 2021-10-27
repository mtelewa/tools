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
from scipy.optimize import curve_fit
import scipy.constants as sci
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats import pearsonr, spearmanr


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
        self.Nz = self.data_z.dimensions['z'].size

        # steps
        self.dx = self.Lx/ self.Nx
        # The length and height arrays for plotting
        self.length_array = np.arange(self.dx/2.0, self.Lx, self.dx)   #nm

        if self.avg_gap_height != 0:
            dz = self.avg_gap_height / self.Nz
            self.height_array = np.arange(dz/2.0, self.avg_gap_height, dz)     #nm

            # If the bulk height is given
            try:
                self.bulk_height = np.array(self.data_x.variables["Bulk_Height"])[self.skip:] / 10
                self.avg_bulk_height = np.mean(self.bulk_height)
                dz_bulk = self.avg_bulk_height / self.Nz

                bulkStart = self.bulk_height[0] + (dz_bulk/2.0)
                self.bulk_height_array = np.arange(bulkStart, self.bulk_height[0]+self.avg_bulk_height , dz_bulk)     #nm
            except KeyError:
                pass

        else:
            dz = self.Lz / self.Nz
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
                'vx_chunkZ_mod':vx_chunkZ_mod, 'vx_nd':vx}

    def fit(self,x,y,order):
        """
        Returns
        -------
        xdata : arr
            x-axis range for the fitting parabola
        polynom(xdata): floats
            Fitting parameters for the velocity profile
        """
        # Fitting coefficients
        coeffs_fit = np.polyfit(x, y, order)     #returns the polynomial coefficients
        # construct the polynomial
        xdata = np.linspace(x[0], x[-1], len(x))
        polynom = np.poly1d(coeffs_fit)

        return {'xdata': xdata, 'fit_data': polynom(xdata), 'coeffs':coeffs_fit}


    def fit_with_cos(self,x,y):

        dd = derive_data(self.skip, self.infile_x, self.infile_z)
        h = x[-1]

        xdata = np.linspace(x[0], x[-1], len(x))
        coeffs,_ = curve_fit(funcs.quadratic_cosine_series, x, y)
        fitted = quadratic_cosine_series(x,*coeffs)

        return {'xdata': xdata, 'fit_data': fitted, 'coeffs':coeffs}


    def slip_length(self, couette=None, pd=None):
        """
        Returns
        -------
        Ls: int
            slip length in nm (obtained from extrapolation of the height at velocty of zero)
        Vs : int
            slip velocity in m/s (Slip velocty * Shear rate)
        xdata: arr
            Positions to inter/extrapolates= '-', mark
        extrapolate: arr
            Extrapolated data
        """
        dd = derive_data(self.skip, self.infile_x, self.infile_z)
        vels = dd.velocity()

        if couette is not None:
            fit_data = dd.fit(vels['height_array_mod'], vels['vx_chunkZ_mod'], 1)['fit_data']
        if pd is not None:
            fit_data = dd.fit(vels['height_array_mod'], vels['vx_chunkZ_mod'], 2)['fit_data']

        npoints = len(vels['height_array_mod'])
        # Positions to inter/extrapolate
        xdata = np.linspace(-10, vels['height_array_mod'][0], npoints)
        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
        # do inter/extrapolation
        extrapolate = InterpolatedUnivariateSpline(vels['height_array_mod'], fit_data, k=order)
        coeffs_extrapolate = np.polyfit(xdata, extrapolate(xdata), 1)
        # Where the velocity profile vanishes
        root = np.roots(coeffs_extrapolate)
        # If the root is positive, there is much noise in the velocity profile
        if root > 0: root = 0

        # Slip velocity according to Navier boundary
        if couette is not None:
            # Vs = vels['vx_chunkZ_mod'][0]
            # Ls = Vs/dd.transport(couette=1)['shear_rate'] * 1e9         # nm
            Ls = np.abs(root) + vels['height_array_mod'][0]
            Vs = Ls * sci.nano * dd.transport(couette=1)['shear_rate']
            print(f'Slip Length {Ls} (nm) and velocity {Vs} (m/s)')

        if pd is not None:
            # Slip length is the extrapolated length in addition to the depletion region: small
            # length where there are no atoms
            Ls = np.abs(root) + vels['height_array_mod'][0]
            Vs = Ls * sci.nano * dd.transport(pd=1)['shear_rate']        # m/s
            print(f'Slip Length {Ls} (nm) and velocity {Vs} (m/s)')

        return {'root':root, 'Ls':Ls, 'Vs':Vs,
                'xdata':xdata,
                'extrapolate':extrapolate(xdata)}

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
              \nAverage mass flow rate in the pump region is {avg_mflowrate_pump} g/ns')

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
        virX : Avg. Shear stress in the walls along the length
        virZ : Avg. Normal stress in the walls along the length
        vir_t : Avg. shear stress in the walls with time.
        """

        totVi = np.array(self.data_x.variables["Voronoi_volumes"])[self.skip:]

        # Off-Diagonal components
        try:
            Wxy = np.array(self.data_x.variables["Wxy"])[self.skip:,1:-1] * sci.atm * pa_to_Mpa
            Wxz = np.array(self.data_x.variables["Wxz"])[self.skip:,1:-1] * sci.atm * pa_to_Mpa
            Wyz = np.array(self.data_x.variables["Wyz"])[self.skip:,1:-1] * sci.atm * pa_to_Mpa

            Wxy_z = np.array(self.data_z.variables["Wxy"])[self.skip:] * sci.atm * pa_to_Mpa
            Wxz_z = np.array(self.data_z.variables["Wxz"])[self.skip:] * sci.atm * pa_to_Mpa
            Wyz_z = np.array(self.data_z.variables["Wyz"])[self.skip:] * sci.atm * pa_to_Mpa

            Wxy_chunkX = np.mean(Wxy, axis=(0,2))
            Wxz_chunkX = np.mean(Wxz, axis=(0,2))
            Wyz_chunkX = np.mean(Wyz, axis=(0,2))

            Wxy_chunkZ = np.mean(Wxy_z, axis=(0,1))
            Wxz_chunkZ = np.mean(Wxz_z, axis=(0,1))
            Wyz_chunkZ = np.mean(Wyz_z, axis=(0,1))

        except KeyError:
            pass

        # Diagonal components
        vir = np.array(self.data_x.variables["Virial"])[self.skip:,] * sci.atm * pa_to_Mpa
        vir_x = np.array(self.data_x.variables["Virial"])[self.skip:,1:-1] * sci.atm * pa_to_Mpa
        vir_z = np.array(self.data_z.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa

        vir_t = np.sum(vir, axis=(1,2)) #/ (3 * totVi)
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
        # print(f'Pressure difference is {pDiff} MPa')

        # Pressure gradient in the simulation domain
        pGrad = - pDiff / pd_length       # MPa/nm
        # print(f'Pressure gradient is {pGrad} MPa/nm')

        try:
            return {'Wxx_chunkX': Wxx_chunkX , 'Wxx_chunkZ': Wxx_chunkZ,
                    'Wyy_chunkX': Wyy_chunkX , 'Wyy_chunkZ': Wyy_chunkZ,
                    'Wzz_chunkX': Wzz_chunkX , 'Wzz_chunkZ': Wzz_chunkZ,
                    'vir_chunkX': vir_chunkX, 'vir_chunkZ': vir_chunkZ,
                    'vir_t': vir_t, 'pGrad': pGrad, 'pDiff':pDiff }
        except NameError:
            return {'vir_chunkX': vir_chunkX, 'virX_nd': vir_x,
                    'vir_chunkZ': vir_chunkZ,
                    'vir_t': vir_t, 'pGrad': pGrad, 'pDiff':pDiff }


    def sigwall(self, pump_size=0.2, pd=None, couette=None):
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

        # Couette (shearing one wall)
        if couette is not None:
            fx_wall = 0.5 * (fx_Lower - fx_Upper)
        # Poiseuille
        if pd is not None:
            fx_wall = 0.5 * (fx_Lower + fx_Upper)

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
                'sigxy_t':sigxy_t, 'pDiff':pDiff, 'sigzz_wall_out':sigzz_wall_out,
                'sigzz_wall_in':sigzz_wall_in}


    def transport(self, couette=None, pd=None, equilib=None):
        """
        Returns
        -------
        mu: float
            Dynamic viscosity
        """
        dd = derive_data(self.skip, self.infile_x, self.infile_z)
        vels = dd.velocity()

        if couette is not None:
            coeffs_fit = np.polyfit(vels['height_array_mod'], vels['vx_chunkZ_mod'], 1)
            sigxz_avg = np.mean(dd.sigwall(couette=1)['sigxz_t']) * 1e9      # mPa
            shear_rate = coeffs_fit[0] * 1e9
            mu = sigxz_avg / shear_rate
            print(f'Viscosity is {mu:.4f} mPa.s at Shear rate {shear_rate:e} s^-1')

        if pd is not None:
            # Get the viscosity
            sigxz_avg = np.mean(dd.sigwall(pd=1)['sigxz_t']) * 1e9      # mPa
            pgrad = dd.virial()['pGrad']            # MPa/m

            coeffs_fit = np.polyfit(vels['height_array_mod'], vels['vx_chunkZ_mod'], 2)
            mu = pgrad/(2 * coeffs_fit[0])          # mPa.s
            shear_rate = sigxz_avg / mu

            coeffs_fit_lo = np.polyfit(vels['height_array_mod'], dd.uncertainty()['lo'], 2)
            mu_lo = pgrad/(2 * coeffs_fit_lo[0])          # mPa.s
            shear_rate_lo = sigxz_avg / mu_lo

            coeffs_fit_hi = np.polyfit(vels['height_array_mod'], dd.uncertainty()['hi'], 2)
            mu_hi = pgrad/(2 * coeffs_fit_hi[0])          # mPa.s
            shear_rate_hi = sigxz_avg / mu_hi

            print(f'Viscosity is {mu:.4f} mPa.s at Shear rate {shear_rate:e} s^-1')
            print(f'Lower vx profile: Viscosity is {mu_lo:.4f} mPa.s at Shear rate {shear_rate_lo:e} s^-1')
            print(f'Upper vx profile: Viscosity is {mu_hi:.4f} mPa.s at Shear rate {shear_rate_hi:e} s^-1')

        if equilib is not None:
            # Equilibrium MD (based on ACF of the shear stress - Green-Kubo relation)
            cutoff = -1
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

        return {'shear_rate': shear_rate, 'mu': mu,
                'shear_rate_lo': shear_rate_lo, 'mu_lo': mu_lo,
                'shear_rate_hi': shear_rate_hi, 'mu_hi': mu_hi,}


    def temp(self):
        """
        Returns
        -------
        tempX : Avg. Shear stress in the walls along the length
        tempZ : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        temp_x = np.array(self.data_x.variables["Temperature"])[self.skip:,1:-1]
        temp_z = np.array(self.data_z.variables["Temperature"])[self.skip:]

        temp_t = np.mean(temp_x,axis=(1,2))
        tempX = np.mean(temp_x,axis=(0,2))
        tempZ = np.mean(temp_z,axis=(0,1))

        return {'tempX':tempX, 'tempZ':tempZ, 'temp_t':temp_t}


    def uncertainty(self):

        # chunk_vol = self.dx * 10 * self.Ly * 10 * np.mean(self.avg_bulk_height) * 10     # A^3
        # vir_x = np.array(self.data_x.variables["Virial"])[self.skip:,1:-1] *  sci.atm * pa_to_Mpa# / (3 * chunk_vol)
        # Block sampling of the virial and calculating the uncertainty
        dd = derive_data(self.skip, self.infile_x, self.infile_z)

        # arr = dd.virial()['virX_nd']
        arr = dd.velocity()['vx_nd']

        n = 100 # samples/block
        # blocks = sq.block_ND(len(self.time), arr, self.Nx-2, n)
        blocks = sq.block_ND(len(self.time), arr, self.Nz, n)
        blocks_mean = np.mean(blocks, axis=0)
        blocks = np.transpose(blocks)
        blocks = blocks[blocks_mean !=0]
        blocks = np.transpose(blocks)
        err =  sq.get_err(blocks)['uncertainty']
        lo, hi = sq.get_err(blocks)['Lo'], sq.get_err(blocks)['Hi']

        # chunk_A =  self.dx * self.Ly * 1e-18
        # fz_Upper = np.array(self.data_x.variables["Fz_Upper"])[self.skip:,1:-1] * kcalpermolA_to_N
        # fz_Lower = np.array(self.data_x.variables["Fz_Lower"])[self.skip:,1:-1] * kcalpermolA_to_N


        # # Block sampling of the wall stress and calculating the uncertainty
        # fz_Upper_blocks = sq.block_ND(len(self.time), fz_Upper, self.Nx-2, 100)
        # fz_Lower_blocks = sq.block_ND(len(self.time), fz_Lower, self.Nx-2, 100)
        # fz_Upper_err = sq.get_err(fz_Upper_blocks)['uncertainty']
        # fz_Lower_err = sq.get_err(fz_Lower_blocks)['uncertainty']

        # # The uncertain in each chunk is the Propagation of uncertain of L and U surfaces
        # cov_in_chunk = np.zeros(self.Nx-2)
        # for i in range(self.Nx-2):
        #     cov_in_chunk[i] = np.cov(fz_Upper_blocks[:,i], fz_Lower_blocks[:,i])[0,1]
        #
        # fz_err = np.sqrt(fz_Upper_err**2 + fz_Lower_err**2 - 2*cov_in_chunk)
        # sigzz_err = fz_err * pa_to_Mpa / chunk_A

        return {'err':err, 'lo':lo, 'hi':hi}#, 'sigzz_length_err':sigzz_err}


    def uncertainty_pDiff(self, pump_size=0.2):

        # Virial -----------------
        # Uncertainty of the measured pDiff
        dd = derive_data(self.skip, self.infile_x, self.infile_z)
        vir_out = dd.virial()['vir_out']
        vir_in = dd.virial()['vir_in']

        # sigzz_out = dd.sigwall()['sigzz_wall_out']
        # sigzz_in = dd.sigwall()['sigzz_wall_in']

        # blocks_out, blocks_in = sq.block_1D(vir_out, 100), sq.block_1D(vir_in, 100)
        blocks_out, blocks_in = sq.block_1D(sigzz_out, 100), sq.block_1D(sigzz_in, 100)

        # Pearson correlation coefficient (Gaussian distributed variables)
        # Check: https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
        corr, _ = spearmanr(blocks_out, blocks_in)


        vir_out_err, vir_in_err = sq.get_err(blocks_out)['uncertainty'], \
                                  sq.get_err(blocks_in)['uncertainty']

        # Propagation of uncertainty
        pDiff_err = np.sqrt(vir_in_err**2 + vir_out_err**2 - 2*np.cov(blocks_out, blocks_in)[0,1])

        # # Wall stress ----------------
        # blocksU_out, blocksU_in = sq.block_1D(upper_out, 100), sq.block_1D(upper_in, 100)
        # blocksL_out, blocksL_in = sq.block_1D(lower_out, 100), sq.block_1D(lower_in, 100)
        #
        # sigzzU_out_err, sigzzU_in_err = sq.get_err(blocksU_out)['uncertainty'] ,  \
        #                                 sq.get_err(blocksU_in)['uncertainty']
        # sigzzL_out_err, sigzzL_in_err = sq.get_err(blocksL_out)['uncertainty'] ,  \
        #                                 sq.get_err(blocksL_in)['uncertainty']
        #
        # sig_A, sig_B, sig_C, sig_D = sigzzU_out_err, sigzzU_in_err, sigzzL_out_err, sigzzL_in_err
        # covAB = np.cov(blocksU_out, blocksU_in)[0,1]
        # covAC = np.cov(blocksU_out, blocksL_out)[0,1]
        # covAD = np.cov(blocksU_out, blocksL_in)[0,1]
        # covBC = np.cov(blocksU_in, blocksL_out)[0,1]
        # covBD = np.cov(blocksU_in, blocksL_in)[0,1]
        # covCD = np.cov(blocksL_out, blocksL_in)[0,1]
        #
        # pDiff_err = np.sqrt(sig_A**2 + sig_B**2 + sig_C**2 + sig_D**2 - 2*covAB -
        #                     2*covAC - 2*covAD - 2*covBC - 2*covBD - 2*covCD)

        print(f"The measured pressure difference uncerainty is {pDiff_err:.2f} MPa \
        \nWith input-output correlation of {corr:.2f}")

        return {'pDiff_err':pDiff_err, 'correlation':corr}


    def dsf(self):
        """
        Returns
        -------
        skx : Avg. Shear stress in the walls along the length
        sky : Avg. Normal stress in the walls along the length
        """

        # wavevectors
        kx = np.array(self.data_x.variables["kx"])
        # ky = np.array(self.data_x.variables["ky"])

        sf_real = np.array(self.data_x.variables["sf"])
        sf_im = np.array(self.data_x.variables["sf_im"])

        # Structure factor
        sf = sf_real + 1j * sf_im
        sf = np.mean(sf_real, axis=0)


        # Intermediate Scattering Function -------------------------------

        # Fourier components of the density
        rho_kx_real = np.array(self.data_x.variables["rho_kx"])
        rho_kx_im = np.array(self.data_x.variables["rho_kx_im"])

        rho_kx = rho_kx_real + 1j * rho_kx_im
        rho_kx_conj = rho_kx_real - 1j * rho_kx_im

        a = np.mean(rho_kx, axis=0)
        b = np.mean(rho_kx_conj, axis=0)
        var = np.var(rho_kx, axis=0)
        ISFx = np.zeros([len(self.time),len(kx)])
        for i in range(len(kx)):
            C = np.correlate(rho_kx[:,i]-a[i], rho_kx_conj[:,i]-b[i], mode="full")
            C = C[C.size // 2:].real
            ISFx[:,i] = C #/ var[i]
        # we get the same with manual acf
        # ISFx = sq.acf(rho_kx)['non-norm'].real

        ISFx_mean = np.mean(ISFx, axis=0)

        # Fourier transform of the ISF gives the Dynamical structure factor
        DSFx = np.fft.fft(ISFx[:,5]).real
        # print(DSFx.shape)
        # DSFx_mean = np.mean(DSFx.real, axis=1)
        # print(DSFx.shape)
        # print(DSFx_mean)

        return {'kx':kx, 'sf':sf, 'ISFx':ISFx, 'DSFx': DSFx}

        # exit()


        # # Intermediate Scattering Function (ISF): ACF of Fourier components of density
        # ISFx = sq.numpy_acf(rho_kx)#['non-norm']
        # ISFx_mean = np.mean(ISFx, axis=0)

        # return {'kx':kx, 'ISFx':ISFx_mean, 'DSFx': DSFx_mean}


    def transverse_acf(self):

        mf = 72.15
        density = np.array(self.data_z.variables["Density"])[self.skip:1000] * (mf/sci.N_A) / (ang_to_cm**3)

        gridsize = np.array([self.Nx])
        # axes to perform FFT over
        fft_axes = np.where(gridsize > 1)[0] + 1
        print(fft_axes)

        # permutation of FFT output axes
        permutation = np.arange(3, dtype=int)
        permutation[fft_axes] = fft_axes[::-1]
        print(permutation)

        rho_tq = np.fft.fftn(density, axes=fft_axes).transpose(*permutation)
        print(rho_tq.shape)

        acf_rho = sq.acf_fft(rho_tq)
        print(acf_rho.shape)

        nmax = (min(gridsize[gridsize > 1]) - 1) // 2
        print(nmax)

        acf_rho_nc = np.zeros([len(self.time)-34000, 1, nmax], dtype=np.float32)

        for ax in fft_axes:
            # mask positive wavevectors
            pmask = list(np.ones(4, dtype=int) * 0)
            pmask[0] = slice(len(self.time-34000))
            pmask[ax] = slice(1, nmax + 1)
            pmask = tuple(pmask)

            print(pmask)

            # mask negative wavevectors
            nmask = list(np.ones(4, dtype=int) * 0)
            nmask[0] = slice(len(self.time-34000))
            nmask[ax] = slice(-1, -nmax - 1, -1)
            nmask = tuple(nmask)

            print(nmask)

            # fill output buffers with average of pos. and neg. wavevectors
            acf_rho_nc[:, ax-1, :] = (acf_rho[pmask] + acf_rho[nmask]) / 2

            print(acf_rho_nc.shape)


        # Fourier transform of the ACF > Spectrum density
        # spec_density = np.fft.fft(acf)
        # Inverse DFT
        # acf = np.fft.ifftn(spec_density,axes=(0,1))

    def trans(self, mf):

        jx = np.array(self.data_x.variables["Jx"])[self.skip:10000]
        # Fourier transform in the space dimensions
        jx_tq = np.fft.fftn(jx, axes=(1,2))

        acf = sq.acf(jx_tq)['norm']

        return {'a':acf[:, 0, 0].real}



#
