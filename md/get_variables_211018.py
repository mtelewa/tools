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
from scipy.stats import pearsonr, spearmanr


# Converions
ang_to_cm = 1e-7
nm_to_cm = 1e-7
A_per_fs_to_m_per_s = 1e5
g_to_kg = 1e-3
pa_to_Mpa = 1e-6
kcalpermolA_to_N = 6.947694845598684e-11
fs_to_ns = 1e-6


class derive_data:
    """
    Computes the Thermodynamic properties and the transport coefficients, as well as the fit data.
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

    def __init__(self, skip, infile_x, infile_z, mf, pumpsize):
        """
        parameters
        ----------
        skip: int
            Timesteps to skip in the sampling
        infile_x: str
            NetCDF file with the grid along the length
        infile_z: str
            NetCDF file with the grid along the gap height
        mf: float
            Molecular mass of the fluid
        pumpsize: float
            Multiple of the box length that represents the size of the pump
            It is used to calculate the pressure gradient.
        """

        self.skip = skip
        self.infile_x = infile_x
        self.infile_z = infile_z
        self.data_x = netCDF4.Dataset(self.infile_x)
        self.data_z = netCDF4.Dataset(self.infile_z)
        self.mf = mf
        if self.mf == 72.15: self.A_per_molecule = 5.
        if self.mf == 39.948 : self.A_per_molecule = 1.
        self.pumpsize = pumpsize
        self.Nf = len(self.data_x.dimensions["Nf"])     # No. of fluid atoms

        # # Variables
        # for varobj in self.data_x.variables.keys():
        #     print(varobj)
        # # Global Attributes
        # for name in self.data_x.ncattrs():
        #     print("Global attr {} = {}".format(name, getattr(self.data_x, name)))

        # Time
        self.time =  np.array(self.data_x.variables["Time"])
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
        if len(self.h)==0:
            print('Reduce the number of skipped steps!')
            exit()
        self.avg_gap_height = np.mean(self.h)
        # COM (in each timestep)
        com = np.array(self.data_x.variables["COM"])[self.skip:] / 10      # nm

        # Number of chunks
        self.Nx = self.data_x.dimensions['x'].size
        self.Nz = self.data_z.dimensions['z'].size

        # Chunk length
        dx = self.Lx/ self.Nx
        # Chunk and whole wall area (in m^2)
        self.chunk_A = dx * self.Ly* 1e-18
        self.wall_A = self.Lx * self.Ly * 1e-18

        # The length and height arrays for plotting
        self.length_array = np.arange(dx/2.0, self.Lx, dx)   #nm

        if self.avg_gap_height != 0:    # Simulation with walls
            dz = self.avg_gap_height / self.Nz
            self.height_array = np.arange(dz/2.0, self.avg_gap_height, dz)     #nm
            self.vol = self.Lx * self.Ly * self.avg_gap_height      # nm3
            # If the bulk height is given
            try:
                # USE THIS
                bulkStart = np.array(self.data_x.variables["Bulk_Start"])[self.skip:] / 10 #
                bulkEnd = np.array(self.data_x.variables["Bulk_End"])[self.skip:] / 10 #
                avg_bulkStart, avg_bulkEnd = np.mean(bulkStart), np.mean(bulkEnd)
                self.bulk_height_array = np.linspace(avg_bulkStart, avg_bulkEnd , self.Nz)     #nm

            except KeyError:
                pass

            # OLD way (Asymmetric virial profile along the height)
            try:
                # REMOVE THIS
                bulk_height = np.array(self.data_x.variables["Bulk_Height"])[self.skip:] / 10
                self.avg_bulk_height = np.mean(bulk_height)
                dz_bulk = self.avg_bulk_height / self.Nz
                bulkStart = bulk_height[0] + (dz_bulk/2.0)
                bulkEnd = bulk_height[0] + self.avg_bulk_height
                self.bulk_height_array = np.arange(bulkStart, bulkEnd , dz_bulk)     #nm
            except KeyError:
                pass

        else:       # Bulk simulation
            dz = self.Lz / self.Nz
            self.height_array = np.arange(dz/2.0, self.Lz, dz)     #nm
            self.vol = np.array(self.data_x.variables["Fluid_Vol"])[self.skip:] * 1e-3 # nm3

    # Thermodynamic properties-----------------------------------------------
    # -----------------------------------------------------------------------

    def velocity(self):
        """
        Returns
        -------
        height_array_mod : arr
            height with the zero-mean chunks removed
        vx_chunkZ_mod : arr
            velocity along the gap height with the zero-mean chunks removed
        """
        vx_full_x = np.array(self.data_x.variables["Vx"])[self.skip:] * A_per_fs_to_m_per_s  # m/s
        vx_full_z = np.array(self.data_z.variables["Vx"])[self.skip:] * A_per_fs_to_m_per_s  # m/s

        vx_chunkX = np.mean(vx_full_x, axis=(0,2))
        vx_chunkZ = np.mean(vx_full_z, axis=(0,1))
        vx_t = (np.mean(vx_full_x, axis=(1,2)) + np.mean(vx_full_z, axis=(1,2))) / 2.

        return {'vx_X':vx_chunkX, 'vx_Z': vx_chunkZ, 'vx_t': vx_t,
                'vx_full_x':vx_full_x, 'vx_full_z':vx_full_z}

    def mflux(self):
        """
        Returns
        -------
        jx_X : mass flux along the box length
        jx_Z : mass flux along the gap height
        jx_t : mass flux time-series
        """
        # Measure the flux and mass flow rate only in the stable region
        jx_stable = np.array(self.data_x.variables["mflux_stable"])[self.skip:] # g/(m^2.ns)
        mflowrate_stable = np.array(self.data_x.variables["mflow_rate_stable"])[self.skip:]
        # Measure the flux and mass flow rate only in the pump region
        jx_pump = np.array(self.data_x.variables["mflux_pump"])[self.skip:] # g/(m^2.ns)
        mflowrate_pump = np.array(self.data_x.variables["mflow_rate_pump"])[self.skip:]

        # Measure the mass flux in the full simulation domain
        jx_full_x = np.array(self.data_x.variables["Jx"])[self.skip:]
        jx_full_z = np.array(self.data_z.variables["Jx"])[self.skip:]
        jx_chunkX = np.mean(jx_full_x, axis=(0,2)) * (sci.angstrom/fs_to_ns) * (self.mf/sci.N_A) / (sci.angstrom**3)
        jx_chunkZ = np.mean(jx_full_z, axis=(0,1)) * (sci.angstrom/fs_to_ns) * (self.mf/sci.N_A) / (sci.angstrom**3)
        jx_t = (np.mean(jx_full_x, axis=(1,2)) + np.mean(jx_full_z, axis=(1,2))) / 2.
        jx_t *= (sci.angstrom/fs_to_ns) * (self.mf/sci.N_A) / (sci.angstrom**3)

        return {'jx_X': jx_chunkX, 'jx_Z': jx_chunkZ, 'jx_t': jx_t,
                'jx_full_x': jx_full_x, 'jx_full_z': jx_full_z,
                'jx_stable': jx_stable, 'mflowrate_stable':mflowrate_stable,
                'jx_pump': jx_pump, 'mflowrate_pump':mflowrate_pump}

    def density(self):
        """
        Returns
        -------
        den_X : Avg. bulk density along the length
        den_t : Avg. bulk density with time.
        den_Z : Avg. fluid density along the height
        """
        # Bulk Density ---------------------
        density_Bulk = np.array(self.data_x.variables["Density_Bulk"])[self.skip:] * (self.mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
        den_X = np.mean(density_Bulk, axis=0)    # g/cm^3

        if np.mean(self.h) == 0: # Bulk simulation
            # TODO: No.
            Nf = len(self.data_x.dimensions["Nf"]) / self.A_per_molecule     # No. of fluid molecules
            mass = Nm * (self.mf/sci.N_A)
            den_t = mass / (self.vol * nm_to_cm**3)    # g/cm3
        else:
            den_t = np.mean(density_Bulk, axis=1)

        # Fluid Density ---------------------
        den_full_z = np.array(self.data_z.variables["Density"])[self.skip:] * (self.mf/sci.N_A) / (ang_to_cm**3)
        den_Z = np.mean(den_full_z, axis=(0,1))     # g/cm^3

        return {'den_X': den_X, 'den_Z': den_Z, 'den_t': den_t,
                'den_full_x':density_Bulk, 'den_full_z':den_full_z}

    def virial(self):
        """
        Returns
        -------
        vir_X : Avg. Shear stress in the walls along the length
        vir_Z : Avg. Normal stress in the walls along the length
        vir_t : Avg. shear stress in the walls with time.
        """

        totVi = np.array(self.data_x.variables["Voronoi_volumes"])[self.skip:]

        # Off-Diagonal components
        try:
            Wxy_full_x = np.array(self.data_x.variables["Wxy"])[self.skip:] * sci.atm * pa_to_Mpa
            Wxz_full_x = np.array(self.data_x.variables["Wxz"])[self.skip:] * sci.atm * pa_to_Mpa
            Wyz_full_x = np.array(self.data_x.variables["Wyz"])[self.skip:] * sci.atm * pa_to_Mpa

            Wxy_full_z = np.array(self.data_z.variables["Wxy"])[self.skip:] * sci.atm * pa_to_Mpa
            Wxz_full_z = np.array(self.data_z.variables["Wxz"])[self.skip:] * sci.atm * pa_to_Mpa
            Wyz_full_z = np.array(self.data_z.variables["Wyz"])[self.skip:] * sci.atm * pa_to_Mpa

            Wxy_chunkX = np.mean(Wxy_full_x, axis=(0,2))
            Wxz_chunkX = np.mean(Wxz_full_x, axis=(0,2))
            Wyz_chunkX = np.mean(Wyz_full_x, axis=(0,2))

            Wxy_chunkZ = np.mean(Wxy_full_z, axis=(0,1))
            Wxz_chunkZ = np.mean(Wxz_full_z, axis=(0,1))
            Wyz_chunkZ = np.mean(Wyz_full_z, axis=(0,1))

            Wxy_t = (np.mean(Wxy_full_z, axis=(1,2)) + np.mean(Wxy_full_x, axis=(1,2))) / 2.
            Wxz_t = (np.mean(Wxz_full_z, axis=(1,2)) + np.mean(Wxz_full_x, axis=(1,2))) / 2.
            Wyz_t = (np.mean(Wyz_full_z, axis=(1,2)) + np.mean(Wyz_full_x, axis=(1,2))) / 2.

        except KeyError:
            pass

        # Diagonal components
        vir_full_x = np.array(self.data_x.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa
        vir_full_z = np.array(self.data_z.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa

        if np.mean(self.h) == 0: # Bulk simulation
            vir_full_x = np.array(self.data_x.variables["Virial"])[self.skip:, 10:-10, :] * sci.atm * pa_to_Mpa
            vir_full_z = np.array(self.data_z.variables["Virial"])[self.skip:, :, 10:-10] * sci.atm * pa_to_Mpa

        vir_t = (np.mean(vir_full_x, axis=(1,2)) + np.mean(vir_full_z, axis=(1,2))) / 2.
        vir_chunkX = np.mean(vir_full_x, axis=(0,2))
        vir_chunkZ = np.mean(vir_full_z, axis=(0,1))

        # pressure gradient ---------------------------------------
        pd_length = self.Lx - self.pumpsize*self.Lx      # nm
        # Virial pressure at the inlet and the outlet of the pump
        out_chunk, in_chunk = np.argmax(vir_chunkX[1:-1]), np.argmin(vir_chunkX[1:-1])
        # timeseries of the output and input chunks
        vir_out, vir_in = np.mean(vir_full_x[:, out_chunk]), np.mean(vir_full_x[:, in_chunk])
        # Pressure Difference  between inlet and outlet
        pDiff = vir_out - vir_in
        p_ratio = vir_out / vir_in
        # Pressure gradient in the simulation domain
        pGrad = - pDiff / pd_length       # MPa/nm

        try:
            return {'Wxy_X': Wxy_chunkX , 'Wxy_Z': Wxy_chunkZ, 'Wxy_t': Wxy_t,
                    'Wxy_full_x': Wxy_full_x, 'Wxy_full_z': Wxy_full_z,
                    'Wxz_X': Wxz_chunkX , 'Wxz_Z': Wxz_chunkZ, 'Wxz_t': Wxz_t,
                    'Wxz_full_x': Wxz_full_x, 'Wxz_full_z': Wxz_full_z,
                    'Wyz_X': Wyz_chunkX , 'Wyz_Z': Wyz_chunkZ, 'Wyz_t': Wyz_t,
                    'Wyz_full_x': Wyz_full_x,'Wyz_full_z': Wyz_full_z,
                    'vir_X': vir_chunkX, 'vir_Z': vir_chunkZ,
                    'vir_t': vir_t, 'pGrad': pGrad, 'pDiff':pDiff,
                    'p_ratio':p_ratio, 'vir_full_x': vir_full_x, 'vir_full_z': vir_full_z}
        except NameError:
            return {'vir_X': vir_chunkX, 'vir_Z': vir_chunkZ,
                    'vir_t': vir_t, 'pGrad': pGrad, 'pDiff':pDiff,
                    'p_ratio':p_ratio, 'vir_full_x': vir_full_x, 'vir_full_z': vir_full_z}


    def sigwall(self):
        """
        Returns
        -------
        sigxz_chunkX : Avg. Shear stress in the walls along the length
        sigzz_chunkX : Avg. Normal stress in the walls along the length
        sigxz_t : Avg. shear stress in the walls with time.
        sigzz_t : Avg. normal stress in the walls with time.
        """

        wall_height = 0.4711    # nm

        # in Newtons (Remove the last chunck)
        fx_Upper = np.array(self.data_x.variables["Fx_Upper"])[self.skip:] * kcalpermolA_to_N
        fy_Upper = np.array(self.data_x.variables["Fy_Upper"])[self.skip:] * kcalpermolA_to_N
        fz_Upper = np.array(self.data_x.variables["Fz_Upper"])[self.skip:] * kcalpermolA_to_N
        fx_Lower = np.array(self.data_x.variables["Fx_Lower"])[self.skip:] * kcalpermolA_to_N
        fy_Lower = np.array(self.data_x.variables["Fy_Lower"])[self.skip:] * kcalpermolA_to_N
        fz_Lower = np.array(self.data_x.variables["Fz_Lower"])[self.skip:] * kcalpermolA_to_N

        # Check if the signs of the average shear stress of the upper and lower walls is the same
        avg_sigxzU = np.mean(np.sum(fx_Upper,axis=1), axis=0)
        avg_sigxzL = np.mean(np.sum(fx_Lower,axis=1), axis=0)
        # If same: Shear-driven or superposed simulations
        if np.sign(avg_sigxzU) != np.sign(avg_sigxzL):
            fx_wall = 0.5 * (fx_Lower - fx_Upper)
        # Else: Equilibrium or loading, Pressure-driven simulations (walls are static in the x-direction)
        else:
            fx_wall = 0.5 * (fx_Lower + fx_Upper)

        fy_wall = 0.5 * (fy_Upper - fy_Lower)
        fz_wall = 0.5 * (fz_Upper - fz_Lower)

        # Stress tensort time series
        sigxz_t = np.sum(fx_wall,axis=1) * pa_to_Mpa / self.wall_A
        sigxy_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / (wall_height * self.Lx * 1e-18)
        sigyz_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / self.wall_A
        sigzz_t = np.sum(fz_wall,axis=1) * pa_to_Mpa / self.wall_A

        # Error calculation for the time series
        fxL, fxU = np.sum(fx_Lower,axis=1) , np.sum(fx_Upper,axis=1)
        sigxz_err = sq.prop_uncertainty(fxL, fxU)['err'] * pa_to_Mpa / self.wall_A

        fzL, fzU = np.sum(fz_Lower,axis=1) , np.sum(fz_Upper,axis=1)
        sigzz_err = sq.prop_uncertainty(fzL, fzU)['err'] * pa_to_Mpa / self.wall_A

        sigxz_chunkX = np.mean(fx_wall,axis=0) * pa_to_Mpa / self.chunk_A
        sigyz_chunkX = np.mean(fy_wall,axis=0) * pa_to_Mpa / self.chunk_A
        sigzz_chunkX = np.mean(fz_wall,axis=0) * pa_to_Mpa / self.chunk_A

        # pressure gradient ----------------------------
        pd_length = self.Lx - self.pumpsize*self.Lx      # nm

        out_chunk, in_chunk = np.argmax(sigzz_chunkX), np.argmin(sigzz_chunkX)

        upper_out, lower_out = fz_Upper[:,out_chunk] * pa_to_Mpa / self.chunk_A ,\
                               fz_Lower[:,out_chunk] * pa_to_Mpa / self.chunk_A
        upper_in, lower_in = fz_Upper[:,in_chunk] * pa_to_Mpa / self.chunk_A ,\
                             fz_Lower[:,in_chunk] * pa_to_Mpa / self.chunk_A

        sigzz_wall_out, sigzz_wall_in = 0.5 * (upper_out - lower_out), \
                                        0.5 * (upper_in - lower_in)

        pDiff = np.mean(sigzz_wall_out) - np.mean(sigzz_wall_in)
        pGrad = - pDiff / pd_length       # MPa/nm

        return {'sigxz_X':sigxz_chunkX, 'sigzz_X':sigzz_chunkX,
                'sigxz_t':sigxz_t, 'sigyz_t':sigyz_t, 'sigzz_t':sigzz_t,
                'sigxy_t':sigxy_t, 'pDiff':pDiff, 'sigzz_wall_out':sigzz_wall_out,
                'sigzz_wall_in':sigzz_wall_in, 'sigxz_err': sigxz_err,
                'sigzz_err':sigzz_err}

    def temp(self):
        """
        Returns
        -------
        temp_X : Avg. Shear stress in the walls along the length
        temp_Z : Avg. Normal stress in the walls along the length
        temp_t : Avg. shear stress in the walls with time.
        """

        temp_full_x = np.array(self.data_x.variables["Temperature"])[self.skip:]
        temp_full_z = np.array(self.data_z.variables["Temperature"])[self.skip:]

        tempX = np.mean(temp_full_x,axis=(0,2))
        tempZ = np.mean(temp_full_z,axis=(0,1))
        temp_t = np.mean(temp_full_x,axis=(1,2)) # Do not include the wild oscillations along the height

        # Inlet and outlet of the pump region
        temp_out = np.max(tempX)
        temp_in = np.min(tempX)
        temp_ratio = temp_out / temp_in

        pd_length = self.Lx - self.pumpsize*self.Lx      # nm
        temp_grad = - (temp_out - temp_in) / pd_length

        try:
            temp_full_x_solid = np.array(self.data_x.variables["Temperature_solid"])[self.skip:]
            tempX_solid = np.mean(temp_full_x_solid, axis=0)
            temp_t_solid = np.mean(temp_full_x_solid, axis=1)
        except KeyError:
            pass

        try:
            return {'temp_X':tempX, 'temp_Z':tempZ, 'temp_t':temp_t, 'temp_ratio':temp_ratio,
                    'temp_full_x': temp_full_x, 'temp_full_z': temp_full_z,
                    'temp_full_x_solid':temp_full_x_solid, 'tempX_solid':tempX_solid,
                    'temp_t_solid':temp_t_solid, temp_grad:'temp_grad'}
        except NameError:
            return {'temp_X':tempX, 'temp_Z':tempZ, 'temp_t':temp_t, 'temp_ratio':temp_ratio,
                    'temp_full_x': temp_full_x, 'temp_full_z': temp_full_z, temp_grad:'temp_grad'}


    def heat_flux(self):
        """
        Calculate the heat flux according to Irving-Kirkwood expression
        """

        # From post-processing we get kcal/mol * A/fs ---> J * m/s
        conv_to_Jmpers = (4184*1e-10)/(sci.N_A*1e-15)

        # Heat flux vector
        je_x = np.array(self.data_x.variables["JeX"])[self.skip:] * conv_to_Jmpers * 1/np.mean(self.vol* 1e-27)  # J/m2.s
        je_y = np.array(self.data_x.variables["JeY"])[self.skip:] * conv_to_Jmpers * 1/np.mean(self.vol* 1e-27)  # J/m2.s
        je_z = np.array(self.data_x.variables["JeZ"])[self.skip:] * conv_to_Jmpers * 1/np.mean(self.vol* 1e-27)  # J/m2.s

        return {'je_x':je_x, 'je_y':je_y, 'je_z':je_z}


    def vel_distrib(self):
        """
        Returns
        -------
        values: arr
            fluid velocities averaged over time for each atom/molecule
        probabilities : arr
            probabilities of these velocities
        """
        fluid_vx = np.array(self.data_x.variables["Fluid_Vx"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s
        fluid_vy = np.array(self.data_x.variables["Fluid_Vy"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s

        fluid_vx_lte = np.array(self.data_x.variables["Fluid_lte_Vx"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s
        fluid_vy_lte = np.array(self.data_x.variables["Fluid_lte_Vy"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s

        values_x, probabilities_x = sq.get_err(fluid_vx)['values'], sq.get_err(fluid_vx)['probs']
        values_y, probabilities_y = sq.get_err(fluid_vy)['values'], sq.get_err(fluid_vy)['probs']

        values_x_lte, probabilities_x_lte = sq.get_err(fluid_vx_lte)['values'], sq.get_err(fluid_vx_lte)['probs']
        values_y_lte, probabilities_y_lte = sq.get_err(fluid_vy_lte)['values'], sq.get_err(fluid_vy_lte)['probs']

        return {'vx_values': values_x, 'vx_prob': probabilities_x,
                'vy_values': values_y, 'vy_prob': probabilities_y,
                'vx_values_lte': values_x_lte, 'vx_prob_lte': probabilities_x_lte,
                'vy_values_lte': values_y_lte, 'vy_prob_lte': probabilities_y_lte}

    # Derived Quantities ----------------------------------------------------
    # -----------------------------------------------------------------------

    def transport(self):
        """
        Returns
        -------
        mu: float
            Dynamic viscosity
        """
        dd = derive_data(self.skip, self.infile_x, self.infile_z, self.mf, self.pumpsize)
        vels = dd.velocity()['vx_Z']

        # sd
        if self.pumpsize==0:
            coeffs_fit = np.polyfit(self.height_array[vels!=0], vels[vels!=0], 1)
            sigxz_avg = np.mean(dd.sigwall()['sigxz_t']) * 1e9      # mPa
            shear_rate = coeffs_fit[0] * 1e9
            mu = sigxz_avg / shear_rate

            coeffs_fit_lo = np.polyfit(self.height_array[vels!=0], sq.get_err(dd.velocity()['vx_full_z'])['lo'], 1)
            shear_rate_lo = coeffs_fit_lo[0] * 1e9
            mu_lo = sigxz_avg / shear_rate_lo

            coeffs_fit_hi = np.polyfit(self.height_array[vels!=0], sq.get_err(dd.velocity()['vx_full_z'])['hi'], 1)
            shear_rate_hi = coeffs_fit_hi[0] * 1e9
            mu_hi = sigxz_avg / shear_rate_hi

        # pd, sp
        if self.pumpsize!=0:
            # Get the viscosity
            sigxz_avg = np.mean(dd.sigwall()['sigxz_t']) * 1e9      # mPa
            pgrad = dd.virial()['pGrad']            # MPa/m

            coeffs_fit = np.polyfit(self.height_array[vels!=0], vels[vels!=0], 2)
            mu = pgrad/(2 * coeffs_fit[0])          # mPa.s
            shear_rate = sigxz_avg / mu

            coeffs_fit_lo = np.polyfit(self.height_array[vels!=0], sq.get_err(dd.velocity()['vx_full_z'])['lo'], 2)
            mu_lo = pgrad/(2 * coeffs_fit_lo[0])          # mPa.s
            shear_rate_lo = sigxz_avg / mu_lo

            coeffs_fit_hi = np.polyfit(self.height_array[vels!=0], sq.get_err(dd.velocity()['vx_full_z'])['hi'], 2)
            mu_hi = pgrad/(2 * coeffs_fit_hi[0])          # mPa.s
            shear_rate_hi = sigxz_avg / mu_hi

        return {'shear_rate': shear_rate, 'mu': mu,
                'shear_rate_lo': shear_rate_lo, 'mu_lo': mu_lo,
                'shear_rate_hi': shear_rate_hi, 'mu_hi': mu_hi}


    def lambda_gk(self):
        """
        Calculate thermal conductivity (lambda) from equilibrium MD
        (based on ACF of the of the heat flux - Green-Kubo relation)
        """

        dd = derive_data(self.skip, self.infile_x, self.infile_z, self.mf, self.pumpsize)
        T = np.mean(dd.temp()['temp_t']) # K
        prefac =  np.mean(self.vol*1e-27)/(sci.k*T**2)        # m3/(J*K)

        # GK relation to get lambda from Equilibrium
        lambda_x = prefac * np.sum(sq.acf(dd.heat_flux()['je_x'])['non-norm']* 1e-15)      # J/mKs
        lambda_y = prefac * np.sum(sq.acf(dd.heat_flux()['je_y'])['non-norm']* 1e-15)      # J/mKs
        lambda_z = prefac * np.sum(sq.acf(dd.heat_flux()['je_z'])['non-norm']* 1e-15)      # J/mKs
        lambda_tot = (lambda_x+lambda_y+lambda_z)/3.

        return lambda_tot


    def lambda_nemd(self):
        """
        Calculate thermal conductivity (lambda) from Fourier's law
        """

        dd = derive_data(self.skip, self.infile_x, self.infile_z, self.mf, self.pumpsize)
        T = np.mean(dd.temp()['temp_t']) # K

        je_x = dd.heat_flux()['je_x']
        lambda_x = -(je_x)/(dd.temp()['temp_grad'])

        return {'lambda_x':lambda_x}


    def viscosity_gk(self):
        """
        Calculate dynamic viscosity from equilibrium MD
        (based on ACF of the shear stress - Green-Kubo relation)
        """
        dd = derive_data(self.skip, self.infile_x, self.infile_z, self.mf, self.pumpsize)

        # TODO: Replace with fluid stress tensor
        sigxy_t = dd.sigwall()['sigxy_t'] * 1e9  # mPa
        sigxz_t = dd.sigwall()['sigxz_t'] * 1e9
        sigyz_t = dd.sigwall()['sigyz_t'] * 1e9

        temp = np.mean(dd.temp()['temp_t'])         # K

        prefac = self.vol* 1e-27 / (sci.k * temp )

        v_xy = prefac * np.sum(sq.acf(sigxy_t)['non-norm']) * 1e-15  # mPa.s
        v_xz = prefac * np.sum(sq.acf(sigxz_t)['non-norm']) * 1e-15  # mPa.s
        v_yz = prefac * np.sum(sq.acf(sigyz_t)['non-norm']) * 1e-15  # mPa.s

        viscosity = (v_xz+v_yz+v_xy)/3.

        return viscosity



    def slip_length(self):
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
        dd = derive_data(self.skip, self.infile_x, self.infile_z, self.mf, self.pumpsize)
        vels = dd.velocity()['vx_Z']

        if self.pumpsize==0:
            fit_data = funcs.fit(self.height_array[vels!=0], vels[vels!=0], 1)['fit_data']
        else:
            fit_data = funcs.fit(self.height_array[vels!=0], vels[vels!=0], 2)['fit_data']

        npoints = len(self.height_array[vels!=0])
        # Positions to inter/extrapolate
        xdata_left = np.linspace(-12, self.height_array[vels!=0][0], npoints)
        xdata_right = np.linspace(self.height_array[vels!=0][-1], 12 , npoints)

        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
        # do inter/extrapolation
        extrapolate = InterpolatedUnivariateSpline(self.height_array[vels!=0], fit_data, k=order)
        coeffs_extrapolate_left = np.polyfit(xdata_left, extrapolate(xdata_left), 1)
        coeffs_extrapolate_right = np.polyfit(xdata_right, extrapolate(xdata_right), 1)

        # Where the velocity profile vanishes
        root_left = np.roots(coeffs_extrapolate_left)
        root_right = np.roots(coeffs_extrapolate_right)

        # If the root is positive or very small, there is much noise in the velocity profile
        if root_left > 0 or np.abs(root_left) < 0.1 : root_left = 0

        # Slip length is the extrapolated length in addition to the depletion region: small
        # length where there are no atoms
        Ls = np.abs(root_left) + self.height_array[vels!=0][0]

        # Slip velocity according to Navier boundary
        Vs = Ls * sci.nano * dd.transport()['shear_rate']        # m/s

        return {'root_left':root_left, 'Ls':Ls, 'Vs':Vs,
                'xdata_left':xdata_left,
                'extrapolate_left':extrapolate(xdata_left),
                'root_right':root_right,
                'xdata_right':xdata_right,
                'extrapolate_right':extrapolate(xdata_right)}


    def sf(self):
        """
        Returns
        -------
        skx : Longitudnal structure factor
        sky : Transverse structure factor
        """

        # wavevectors
        kx = np.array(self.data_x.variables["kx"])
        ky = np.array(self.data_x.variables["ky"])

        # skip here is used to truncate the calculated SF
        sf_real = np.array(self.data_x.variables["sf"])[:self.skip]
        sf_x_real = np.array(self.data_x.variables["sf_x"])[:self.skip]
        sf_y_real = np.array(self.data_x.variables["sf_y"])[:self.skip]

        # Structure factor averaged over time for each k
        sf = np.mean(sf_real, axis=0)
        sf_time = np.mean(sf_real, axis=(1,2))
        sf_x = np.mean(sf_x_real, axis=0)
        sf_y = np.mean(sf_y_real, axis=0)

        # How to get S(k) in radial direction (In 2D from k=(kx,ky) and the sf=(sfx,sfy))
        # k_vals=[]
        # sf_r=[]
        # for i in range(len(kx)):
        #     for k in range(len(ky)):
        #         k_vals.append(np.sqrt(kx[i]**2+ky[k]**2))
        #         sf_r.append(np.sqrt(sf_x[i]**2+sf_y[k]**2))

        return {'kx':kx, 'ky':ky, 'sf':sf, 'sf_x':sf_x, 'sf_y':sf_y, 'sf_time':sf_time}

    def ISF(self):
        """
        Intermediate Scattering Function
        """

        # Fourier components of the density
        rho_k = np.array(self.data_x.variables["rho_k"])

        # a = np.mean(rho_k, axis=0)
        # var = np.var(rho_k, axis=0)

        ISF = sq.acf_conjugate(rho_k)['norm']
        print(ISF.shape)

        # ISF = np.zeros([len(self.time),len(kx)])
        # for i in range(len(kx)):
        #     C = np.correlate(rho_kx[:,i]-a[i], rho_kx_conj[:,i]-b[i], mode="full")
        #     C = C[C.size // 2:].real
        #     ISF[:,i] = C #/ var[i]
        # we get the same with manual acf
        # ISFx = sq.acf(rho_kx)['non-norm'].real

        # ISFx_mean = np.mean(ISFx, axis=0)

        # Fourier transform of the ISF gives the Dynamical structure factor
        # DSFx = np.fft.fft(ISFx[:,5]).real
        # DSFx_mean = np.mean(DSFx.real, axis=1)

        # Intermediate Scattering Function (ISF): ACF of Fourier components of density
        # ISFx = sq.numpy_acf(rho_kx)#['non-norm']
        # ISFx_mean = np.mean(ISFx, axis=0)

        return {'ISF':ISF}


    def transverse_acf(self):

        density = np.array(self.data_z.variables["Density"])[self.skip:1000] * (self.mf/sci.N_A) / (ang_to_cm**3)

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

    def trans(self):

        jx = np.array(self.data_x.variables["Jx"])[self.skip:10000]
        # Fourier transform in the space dimensions
        jx_tq = np.fft.fftn(jx, axes=(1,2))

        acf = sq.acf(jx_tq)['norm']

        return {'a':acf[:, 0, 0].real}




    # def uncertainty(self, qtty):
    #
    #     # Block sampling and calculating the uncertainty
    #     n = 100 # samples/block
    #
    #     dd= derive_data(self.skip, self.infile_x, self.infile_z, self.mf, self.pumpsize)
    #
    #     # if qtty == 'vir_X': blocks = sq.block_ND(len(self.time[self.skip:]), dd.virial()['vir_X'], self.Nx, n)
    #     # if qtty == 'den_X': blocks = sq.block_ND(len(self.time[self.skip:]), dd.density()['density_Bulk'], self.Nx, n)
    #     # if qtty == 'vx_Z':
    #     #     arr = np.array(self.data_z.variables["Vx"])[self.skip:] * A_per_fs_to_m_per_s  # m/s
    #     #     blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nz, n)
    #
    #     # Blocks shape is (time/n, Nchunks)
    #     err =  sq.get_err(blocks)['uncertainty']
    #     lo, hi = sq.get_err(blocks)['Lo'], sq.get_err(blocks)['Hi']
    #
    #     # Discard chunks with zero time average
    #     # Get the time average in each chunk
    #     # blocks_mean = np.mean(blocks, axis=0)
    #     # blocks = np.transpose(blocks)
    #     # blocks = blocks[blocks_mean !=0]
    #     # blocks = np.transpose(blocks)
    #
    #     return {'err':err, 'lo':lo, 'hi':hi}


    # def prop_uncertainty(self, qtty):
    #
    #     n = 100 # samples/block
    #
    #     if qtty=='sigzz_X':
    #         qtty1 = np.array(self.data_x.variables["Fz_Upper"])[self.skip:] * kcalpermolA_to_N * pa_to_Mpa / self.chunk_A
    #         qtty2 = np.array(self.data_x.variables["Fz_Lower"])[self.skip:] * kcalpermolA_to_N * pa_to_Mpa / self.chunk_A
    #
    #     if qtty=='sigxz_X':
    #         qtty1 = np.array(self.data_x.variables["Fx_Upper"])[self.skip:] * kcalpermolA_to_N * pa_to_Mpa / self.chunk_A
    #         qtty2 = np.array(self.data_x.variables["Fx_Lower"])[self.skip:] * kcalpermolA_to_N * pa_to_Mpa / self.chunk_A
    #
    #     # Blocks with dimension (time, chunk)
    #     qtty1_blocks = sq.block_ND(len(self.time[self.skip:]), qtty1, self.Nx, n)
    #     qtty2_blocks = sq.block_ND(len(self.time[self.skip:]), qtty2, self.Nx, n)
    #
    #     # Get the error in each chunk
    #     qtty1_err = sq.get_err(qtty1_blocks)['uncertainty']
    #     qtty2_err = sq.get_err(qtty2_blocks)['uncertainty']
    #
    #     # The uncertainty in each chunk is the Propagation of uncertainty of L and U surfaces
    #     # Get the covariance in the chunk
    #     cov_in_chunk = np.zeros(self.Nx)
    #     for i in range(self.Nx):
    #         cov_in_chunk[i] = np.cov(qtty1_blocks[:,i], qtty2_blocks[:,i])[0,1]
    #
    #     # Needs to be corrected with a positive sign of the variance for sigxz in PD
    #     err = np.sqrt(0.5**2*qtty1_err**2 + 0.5**2*qtty2_err**2 - 2*0.5*0.5*cov_in_chunk)
    #     avg = 0.5 * (np.mean(qtty1, axis=0) - np.mean(qtty2, axis=0))
    #     lo = avg - err
    #     hi = avg + err
    #
    #     return {'err':err, 'lo':lo, 'hi':hi}
