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
from scipy.optimize import curve_fit
from scipy.integrate import simpson, trapezoid
from math import isclose
from operator import itemgetter
import extract_thermo
import numpy.ma as ma

# np.seterr(all='raise')

# Conversions
ang_to_cm = 1e-8
nm_to_cm = 1e-7
g_to_kg = 1e-3
pa_to_Mpa = 1e-6
fs_to_ns = 1e-6
Angstromperfs_to_mpers = 1e5
kcalpermolA_to_N = 6.947694845598684e-11

class ExtractFromTraj:

    def __init__(self, skip, infile_x, infile_z, mf, pumpsize):
        """
        parameters
        ----------
        skip: int, Timesteps to skip in the sampling
        infile_x: str, NetCDF file with the grid along the length
        infile_z: str, NetCDF file with the grid along the gap height
        mf: float, Molecular mass of the fluid
        pumpsize: float, Multiple of the box length that represents the size of the pump.
        """

        self.skip = skip
        self.infile_x = infile_x
        self.infile_z = infile_z
        self.data_x = netCDF4.Dataset(self.infile_x)
        self.data_z = netCDF4.Dataset(self.infile_z)
        self.mf = mf
        if self.mf == 39.948 : self.A_per_molecule = 1. # Lennard-Jones
        if self.mf == 44.09 : self.A_per_molecule = 3  # propane
        if self.mf == 72.15: self.A_per_molecule = 5.   # Pentane
        if self.mf == 100.21 : self.A_per_molecule = 7  # heptane
        if self.mf == 442.83 : self.A_per_molecule = 30  # squalane
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
        self.Lz = dim["Lz"] / 10      # nm

        # Number of chunks
        self.Nx = self.data_x.dimensions['x'].size
        self.Nz = self.data_z.dimensions['z'].size

        try:    # Simulation with walls
            # Gap heights (time,) in nm
            self.h = np.array(self.data_x.variables["Height"])[self.skip:] / 10
            if len(self.h) == 0:
                raise ValueError('The array is empty! Reduce the skipped timesteps.')
            # Average gap height
            self.avg_gap_height = np.mean(self.h)
            try:       # Converging-Diverging channels
                self.h_conv = np.array(self.data_x.variables["Height_conv"])[self.skip:] / 10
                self.h_div = np.array(self.data_x.variables["Height_div"])[self.skip:] / 10
            except KeyError:
                pass
            # Bulk height (time,)
            bulkStart = np.array(self.data_x.variables["Bulk_Start"])[self.skip:] / 10
            bulkEnd = np.array(self.data_x.variables["Bulk_End"])[self.skip:] / 10
            # Time-average
            avg_bulkStart, avg_bulkEnd = np.mean(bulkStart), np.mean(bulkEnd)
            # Center of Mass (time,)
            com = np.array(self.data_x.variables["COM"])[self.skip:] / 10

            # The length array
            dx = self.Lx/ self.Nx
            self.length_array = np.arange(dx/2.0, self.Lx, dx)   #nm

            # Chunk and whole wall areas (in m^2)
            self.chunk_A = dx * self.Ly * 1e-18
            self.wall_A = self.Lx * self.Ly * 1e-18

            # The total gap height array
            dz = self.avg_gap_height / self.Nz
            self.height_array = np.arange(dz/2.0, self.avg_gap_height, dz)

            # The bulk region gap height array
            self.bulk_height_array = np.linspace(avg_bulkStart, avg_bulkEnd , self.Nz)

            # self.vol = self.Lx * self.Ly * self.avg_gap_height      # nm^3

        except KeyError:    # Bulk simulation
            self.avg_gap_height = 0
            dx = self.Lx/ self.Nx
            self.length_array = np.arange(dx/2.0, self.Lx, dx)

            dz = self.Lz / self.Nz
            self.height_array = np.arange(dz/2.0, self.Lz, dz)
            self.vol = np.array(self.data_x.variables["Fluid_Vol"])[self.skip:] * 1e-3      # nm^3


    def mask_invalid_zeros(self, array):
        """
        Mask zeros and nans generated from post-processing script in bins with no atoms on fine grids
        Parameters:
        -----------
        array of shape (time,Nx,Nz)
        Returns:
        --------
        array of same shape with zeros and nans masked
        """
        mask_invalid = np.ma.masked_invalid(array)
        masked_array = ma.masked_where(mask_invalid == 0, mask_invalid)
        # If array is all masked that means values are all zeros (i.e. property was not computed during simulation)
        if np.ma.count_masked(array) == array.size:
            masked_array = np.zeros(array.shape)

        return masked_array

    # Thermodynamic properties-----------------------------------------------
    # -----------------------------------------------------------------------

    def velocity(self):
        """
        Computes the center of mass (streaming or x-component) velocity of the fluid
        Units: m/s

        Returns
        -------
        vx_X : arr (Nx,), velocity of whole fluid along the length
        vx_Z : arr (Nz,), velocity of fluid in the stable region along the gap height
        vx_t : arr (time,), velocity time-series
        vx_R : arr (Nz,), velocity in 5 regions along the stream
        """

        # Velocity of the whole fluid along the stream
        try:    # Simulation with walls
            vx_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Vx_whole"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
        except KeyError: # Bulk simulation
            vx_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Vx"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
        # Measured away from the pump (Stable region)
        vx_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Vx"])[self.skip:]) * Angstromperfs_to_mpers  # m/s

        vx_chunkX = np.mean(vx_full_x, axis=(0,2))
        vx_chunkZ = np.mean(vx_full_z, axis=(0,1))
        vx_t = np.mean(vx_full_x, axis=(1,2))

        # First and last 2 chunks have large uncertainties
        if np.ma.count_masked(vx_chunkZ)!=0:
            first_non_masked = np.ma.flatnotmasked_edges(vx_chunkZ)[0]
            last_non_masked = np.ma.flatnotmasked_edges(vx_chunkZ)[1]
            vx_chunkZ[first_non_masked], vx_chunkZ[first_non_masked+1], \
            vx_chunkZ[last_non_masked], vx_chunkZ[last_non_masked-1] = np.nan, np.nan, np.nan, np.nan

        try: # For the 5-regions grid
            vx_R1_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Vx_R1"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
            vx_R2_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Vx_R2"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
            vx_R3_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Vx_R3"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
            vx_R4_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Vx_R4"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
            vx_R5_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Vx_R5"])[self.skip:]) * Angstromperfs_to_mpers  # m/s

            vx_R1 = np.mean(vx_R1_z, axis=(0,1))
            vx_R2 = np.mean(vx_R2_z, axis=(0,1))
            vx_R3 = np.mean(vx_R3_z, axis=(0,1))
            vx_R4 = np.mean(vx_R4_z, axis=(0,1))
            vx_R5 = np.mean(vx_R5_z, axis=(0,1))

        except KeyError:
            vx_R1, vx_R2, vx_R3, vx_R4, vx_R5 = 0, 0, 0, 0, 0

        return {'vx_X':vx_chunkX, 'vx_Z': vx_chunkZ, 'vx_t': vx_t,
                'vx_full_x':vx_full_x, 'vx_full_z':vx_full_z,
                'vx_R1':vx_R1, 'vx_R2':vx_R2, 'vx_R3':vx_R3, 'vx_R4':vx_R4, 'vx_R5':vx_R5}


    def density(self):
        """
        Computes the mass density of the fluid
        Units: g/cm^3

        Returns
        -------
        den_X : arr (Nx,), Bulk density along the length
        den_Z : arr (Nz,), Fluid density along the gap height
        den_t : arr (time,), Bulk density time series
        """

        # Bulk Density ---------------------
        if self.avg_gap_height !=0:
            density_Bulk = self.mask_invalid_zeros(np.array(self.data_x.variables["Density_Bulk"])[self.skip:]) / (sci.N_A * ang_to_cm**3)
            den_X = np.mean(density_Bulk, axis=0)
        else:
            density_Bulk = self.mask_invalid_zeros(np.array(self.data_x.variables["Density"])[self.skip:]) / (sci.N_A * ang_to_cm**3)
            den_X = np.mean(density_Bulk, axis=(0,2))

        # Fluid Density ---------------------
        den_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Density"])[self.skip:]) / (sci.N_A * ang_to_cm**3)
        den_Z = np.mean(den_full_z, axis=(0,1))

        if np.mean(self.avg_gap_height) != 0: # Walls simulation
            den_t = np.mean(density_Bulk, axis=1)
        else:   # Bulk
            Nm = len(self.data_x.dimensions["Nf"]) / self.A_per_molecule     # No. of fluid molecules
            mass = Nm * (self.mf/sci.N_A)       # g
            den_t = mass / (self.vol * nm_to_cm**3)     # g/cm3

        return {'den_X': den_X, 'den_Z': den_Z, 'den_t': den_t,
                'den_full_x':density_Bulk, 'den_full_z':den_full_z}


    def mflux(self):
        """
        Computes the x-component of the mass flux and the mass flow rate of the fluid
        Units: g/(m^2.ns) and g/ns for the mass flux and mass flow rate, respectively

        Returns
        -------
        jx_X : arr (Nx,), mass flux along the length
        jx_Z : arr (Nz,), mass flux along the gap height
        jx_t : arr (time,), mass flux time-series
        jx_stable : arr (time,), mass flux time-series only in the stable region (defined in LAMMPS post-processing script)
        jx_pump : arr (time,), mass flux time-series only in the pump region (defined in LAMMPS post-processing script)

        mflowrate_X : arr (Nx,), mass flow rate along the length
        mflowrate_t : arr (time,), mass flow rate time-series
        mflowrate_stable : arr (time,), mass flow rate time-series only in the stable region (defined in LAMMPS post-processing script)
        mflowrate_pump : arr (time,), mass flow rate time-series only in the pump region (defined in LAMMPS post-processing script)
        """

        # Measure the flux and mass flow rate only in the stable region
        jx_stable = np.array(self.data_x.variables["mflux_stable"])[self.skip:] # g/(m^2.ns)
        mflowrate_stable = np.array(self.data_x.variables["mflow_rate_stable"])[self.skip:]
        # Measure the flux and mass flow rate only in the pump region
        jx_pump = np.array(self.data_x.variables["mflux_pump"])[self.skip:] # g/(m^2.ns)
        mflowrate_pump = np.array(self.data_x.variables["mflow_rate_pump"])[self.skip:]

        # Measure the mass flux in the full simulation domain
        jx_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Jx"])[self.skip:])
        jx_chunkX = np.mean(jx_full_x, axis=(0,2)) / (sci.N_A * fs_to_ns * sci.angstrom**2)

        jx_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Jx"])[self.skip:])
        jx_chunkZ = np.mean(jx_full_z, axis=(0,1)) / (sci.N_A * fs_to_ns * sci.angstrom**2)

        jx_t = np.sum(jx_full_x, axis=(1,2)) / (sci.N_A * fs_to_ns * sci.angstrom**2)

        # Mass flow rate in the fluid domain
        mflowrate_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["mdot"])[self.skip:])
        mflowrate_chunkX = np.mean(mflowrate_full_x, axis=(0,2)) / (sci.N_A * fs_to_ns)

        mflowrate_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["mdot"])[self.skip:])
        mflowrate_chunkZ = np.mean(mflowrate_full_x, axis=(0,2)) / (sci.N_A * fs_to_ns)

        # Time-averaged mass flow rate in the whole domain
        mflowrate_t = np.sum(mflowrate_full_x, axis=(1,2)) / (sci.N_A * fs_to_ns)

        return {'jx_X': jx_chunkX, 'jx_Z': jx_chunkZ, 'jx_t': jx_t,
                'jx_full_x': jx_full_x, 'jx_full_z': jx_full_z,
                'jx_stable': jx_stable, 'mflowrate_stable':mflowrate_stable,
                'jx_pump': jx_pump, 'mflowrate_pump':mflowrate_pump, 'mflowrate_X':mflowrate_chunkX,
                'mflowrate_t':mflowrate_t, 'mflowrate_full_x': mflowrate_full_x}


    def mflowrate_hp(self):
        """
        Computes the continuum prediction (Hagen Poiseuille) for mass flow rate
        Can be used to initialize FC simulations without running FF simulations apriori

        Units: g/(m^2.ns)
        """

        avg_gap_height = self.avg_gap_height * 1e-9      # m
        bulk_den_avg = np.mean(self.density()['den_t']) * 1e3     # kg/m3
        pGrad = np.absolute(self.virial()['pGrad']) * 1e6 * 1e9  # Pa/m
        eta = self.viscosity_nemd()['eta'] * 1e-3            # Pa.s
        Ly = self.Ly * 1e-9
        kgpers_to_gperns =  1e-9 * 1e3

        # Without slip
        mflowrate_hp = (bulk_den_avg * Ly * pGrad * avg_gap_height**3 * kgpers_to_gperns) / (12 * eta)

        # With slip
        mflowrate_hp_slip = mflowrate_hp + (bulk_den_avg * Ly * kgpers_to_gperns * \
                            np.mean(self.slip_length()['Vs']) * avg_gap_height)

        if self.pumpsize == 0: raise ValueError('Pump size is 0. There is no pressure gradient!')

        return {'mflowrate_hp':mflowrate_hp, 'mflowrate_hp_slip':mflowrate_hp_slip}


    def virial(self):
        """
        Computes the virial tensor as well as the trace of the tensor i.e. the
        virial pressure of the fluid according the Irving-Kirkwood expression
        Units: MPa

        Returns
        -------
        vir_X : arr (Nx,), virial pressure along the length
        vir_Z : arr (Nz,), virial pressure along the gap height
        vir_t : arr (time,), virial pressure time-series
        W<ab>_X : arr (Nx,), ab virial tensor component along the length where ab are cartesian coordinates
        W<ab>_Z : arr (Nz,), ab virial tensor component along the gap height
        W<ab>_t : arr (time,), ab virial tensor component time-series

        """

        # Diagonal components (bulk)
        Wxx_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wxx"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wyy_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wyy"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wzz_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wzz"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wxx_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Wxx"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wyy_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Wyy"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wzz_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Wzz"])[self.skip:]) * sci.atm * pa_to_Mpa
        # Off-Diagonal components (Whole fluid)
        Wxy_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wxy"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wxz_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wxz"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wyz_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wyz"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wxy_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Wxy"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wxz_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Wxz"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wyz_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Wyz"])[self.skip:]) * sci.atm * pa_to_Mpa

        if self.avg_gap_height != 0:
            # Averaging along time and height
            Wxx_chunkX = np.mean(Wxx_full_x, axis=(0,2))
            Wyy_chunkX = np.mean(Wyy_full_x, axis=(0,2))
            Wzz_chunkX = np.mean(Wzz_full_x, axis=(0,2))
            Wxy_chunkX = np.mean(Wxy_full_x, axis=(0,2))
            Wxz_chunkX = np.mean(Wxz_full_x, axis=(0,2))
            Wyz_chunkX = np.mean(Wyz_full_x, axis=(0,2))
            # Averaging along time and length
            Wxx_chunkZ = np.mean(Wxx_full_z, axis=(0,1))
            Wyy_chunkZ = np.mean(Wyy_full_z, axis=(0,1))
            Wzz_chunkZ = np.mean(Wzz_full_z, axis=(0,1))
            Wxy_chunkZ = np.mean(Wxy_full_z, axis=(0,1))
            Wxz_chunkZ = np.mean(Wxz_full_z, axis=(0,1))
            Wyz_chunkZ = np.mean(Wyz_full_z, axis=(0,1))
            # Averaging along length and height
            Wxx_t = np.mean(Wxx_full_z, axis=(1,2))
            Wyy_t = np.mean(Wyy_full_z, axis=(1,2))
            Wzz_t = np.mean(Wzz_full_z, axis=(1,2))
            Wxy_t = np.mean(Wxy_full_z, axis=(1,2))
            Wxz_t = np.mean(Wxz_full_z, axis=(1,2))
            Wyz_t = np.mean(Wyz_full_z, axis=(1,2))

            if self.pumpsize == 0: # Compute virial pressure from the three diagonal components
                print('Virial computed from the three components')
                vir_full_x = -(Wxx_full_x + Wyy_full_x + Wzz_full_x) / 3.
                vir_full_z = -(Wxx_full_z + Wyy_full_z + Wzz_full_z) / 3.
            else:  # Compute virial pressure from the y-component (perp. to wall-perp. to flow direction)
                print('Virial computed from the y-component')
                vir_full_x = - Wyy_full_x
                vir_full_z = - Wyy_full_z

            vir_chunkX = np.mean(vir_full_x, axis=(0,2))
            vir_chunkZ = np.mean(vir_full_z, axis=(0,1))
            vir_t = np.mean(vir_full_x, axis=(1,2))

        if self.avg_gap_height == 0:
            # Averaging along time and height
            Wxx_chunkX = np.mean(Wxx_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            Wyy_chunkX = np.mean(Wyy_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            Wzz_chunkX = np.mean(Wzz_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            Wxy_chunkX = np.mean(Wxy_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            Wxz_chunkX = np.mean(Wxz_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            Wyz_chunkX = np.mean(Wyz_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            # Averaging along time and length
            Wxx_chunkZ = np.mean(Wxx_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)
            Wyy_chunkZ = np.mean(Wyy_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)
            Wzz_chunkZ = np.mean(Wzz_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)
            Wxy_chunkZ = np.mean(Wxy_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)
            Wxz_chunkZ = np.mean(Wxz_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)
            Wyz_chunkZ = np.mean(Wyz_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)

            Wxx_t = np.sum(Wxx_full_z, axis=(1,2)) / (self.vol*1e3)
            Wyy_t = np.sum(Wyy_full_z, axis=(1,2)) / (self.vol*1e3)
            Wzz_t = np.sum(Wzz_full_z, axis=(1,2)) / (self.vol*1e3)
            Wxy_t = np.sum(Wxy_full_z, axis=(1,2)) / (self.vol*1e3)
            Wxz_t = np.sum(Wxz_full_z, axis=(1,2)) / (self.vol*1e3)
            Wyz_t = np.sum(Wyz_full_z, axis=(1,2)) / (self.vol*1e3)

            vir_full_x = -(Wxx_full_x + Wyy_full_x + Wzz_full_x) / 3.
            vir_full_z = -(Wxx_full_z + Wyy_full_z + Wzz_full_z) / 3.

            vir_chunkX = np.mean(vir_full_x, axis=(0,2)) / (np.mean(self.vol)*1e3 / self.Nx)
            vir_chunkZ = np.mean(vir_full_z, axis=(0,1)) / (np.mean(self.vol)*1e3 / self.Nz)
            vir_t = -(Wxx_t + Wyy_t + Wzz_t) / 3.

        vir_fluctuations = sq.get_err(vir_full_x)['var']

        # pressure gradient ---------------------------------------
        pd_length = self.Lx - self.pumpsize * self.Lx      # nm
        # Virial pressure at the inlet and the outlet of the pump
        out_chunk, in_chunk = np.argmax(vir_chunkX), np.argmin(vir_chunkX)
        # timeseries of the output and input chunks
        vir_out, vir_in = np.mean(vir_full_x[:, out_chunk]), np.mean(vir_full_x[:, in_chunk])
        # Pressure Difference  between inlet and outlet
        pDiff = vir_out - vir_in
        # Pressure gradient in the simulation domain
        if pd_length !=0:
            pGrad = - pDiff / pd_length       # MPa/nm
        else:
            pGrad = 0

        # # Remove first and last chuunks
        # first_non_masked = np.ma.flatnotmasked_edges(vir_chunkX)[0]
        # last_non_masked = np.ma.flatnotmasked_edges(vir_chunkX)[1]
        # vir_chunkX[first_non_masked], vir_chunkX[last_non_masked] = np.nan, np.nan

        return {'Wxx_X': Wxx_chunkX , 'Wxx_Z': Wxx_chunkZ, 'Wxx_t': Wxx_t,
                'Wxx_full_x': Wxx_full_x, 'Wxx_full_z': Wxx_full_z,
                'Wyy_X': Wyy_chunkX , 'Wyy_Z': Wyy_chunkZ, 'Wyy_t': Wyy_t,
                'Wyy_full_x': Wyy_full_x, 'Wyy_full_z': Wyy_full_z,
                'Wzz_X': Wzz_chunkX , 'Wzz_Z': Wzz_chunkZ, 'Wzz_t': Wzz_t,
                'Wzz_full_x': Wzz_full_x,'Wzz_full_z': Wzz_full_z,
                'Wxy_X': Wxy_chunkX , 'Wxy_Z': Wxy_chunkZ, 'Wxy_t': Wxy_t,
                'Wxy_full_x': Wxy_full_x, 'Wxy_full_z': Wxy_full_z,
                'Wxz_X': Wxz_chunkX , 'Wxz_Z': Wxz_chunkZ, 'Wxz_t': Wxz_t,
                'Wxz_full_x': Wxz_full_x, 'Wxz_full_z': Wxz_full_z,
                'Wyz_X': Wyz_chunkX , 'Wyz_Z': Wyz_chunkZ, 'Wyz_t': Wyz_t,
                'Wyz_full_x': Wyz_full_x,'Wyz_full_z': Wyz_full_z,
                'vir_X': vir_chunkX, 'vir_Z': vir_chunkZ,
                'vir_t': vir_t, 'pGrad': pGrad, 'pDiff':pDiff,
                'vir_full_x': vir_full_x, 'vir_full_z': vir_full_z}


    def sigwall(self):
        """
        Computes the stress tensor in the solid walls from the mechanical defintion
        Units: MPa

        Returns
        -------
        sigxz_chunkX : arr (Nx,), Shear stress along the length
        sigzz_chunkX :  arr (Nx,), Normal stress along the length
        sigxz_t :  arr (time,), Shear stress time-series
        sigzz_t :  arr (time,), Normal stress times-series
        """

        wall_height = 0.4711 * 2   # nm

        # in Newtons (Remove the last chunck)
        fx_Upper = np.array(self.data_x.variables["Fx_Upper"])[self.skip:] * kcalpermolA_to_N
        fy_Upper = np.array(self.data_x.variables["Fy_Upper"])[self.skip:] * kcalpermolA_to_N
        fz_Upper = np.array(self.data_x.variables["Fz_Upper"])[self.skip:] * kcalpermolA_to_N
        fx_Lower = np.array(self.data_x.variables["Fx_Lower"])[self.skip:] * kcalpermolA_to_N
        fy_Lower = np.array(self.data_x.variables["Fy_Lower"])[self.skip:] * kcalpermolA_to_N
        fz_Lower = np.array(self.data_x.variables["Fz_Lower"])[self.skip:] * kcalpermolA_to_N

        # Check if the signs of the average shear stress of the upper and lower walls is the same
        avg_FxU = np.mean(np.sum(fx_Upper,axis=1), axis=0)
        avg_FxL = np.mean(np.sum(fx_Lower,axis=1), axis=0)
        # If same: Shear-driven or superposed simulations
        if np.sign(avg_FxU) != np.sign(avg_FxL):
            fx_wall = 0.5 * (fx_Lower - fx_Upper)
        # Else: Equilibrium or loading, Pressure-driven simulations (walls are static in the x-direction)
        else:
            fx_wall = 0.5 * (fx_Lower + fx_Upper)

        fy_wall = 0.5 * (fy_Upper - fy_Lower)
        fz_wall = 0.5 * (fz_Upper - fz_Lower)

        # Stress tensort time series
        sigxz_t = np.sum(fx_wall,axis=1) * pa_to_Mpa / self.wall_A
        sigyz_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / self.wall_A
        sigzz_t = np.sum(fz_wall,axis=1) * pa_to_Mpa / self.wall_A

        sigxx_t = np.sum(fx_wall,axis=1) * pa_to_Mpa / (wall_height * self.Lx * 1e-18)
        sigxy_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / (wall_height * self.Lx * 1e-18)
        sigyy_t = np.sum(fy_wall,axis=1) * pa_to_Mpa / (wall_height * self.Ly * 1e-18)

        # Error calculation for the time series
        fxL, fxU = np.sum(fx_Lower,axis=1) , np.sum(fx_Upper,axis=1)
        fzL, fzU = np.sum(fz_Lower,axis=1) , np.sum(fz_Upper,axis=1)

        # Calculation of stress in the chunks
        sigxz_chunkX = np.mean(fx_wall,axis=0) * pa_to_Mpa / self.chunk_A
        sigyz_chunkX = np.mean(fy_wall,axis=0) * pa_to_Mpa / self.chunk_A
        sigzz_chunkX = np.mean(fz_wall,axis=0) * pa_to_Mpa / self.chunk_A

        # Error calculation for each chunk
        sigxz_err_t = sq.prop_uncertainty(fxL, fxU)['err'] * pa_to_Mpa / self.wall_A
        sigxz_lo_t = sq.prop_uncertainty(fxL, fxU)['lo'] * pa_to_Mpa / self.wall_A
        sigxz_hi_t = sq.prop_uncertainty(fxL, fxU)['hi'] * pa_to_Mpa / self.wall_A

        sigzz_err_t = sq.prop_uncertainty(fzL, fzU)['err'] * pa_to_Mpa / self.wall_A
        sigzz_lo_t = sq.prop_uncertainty(fzL, fzU)['lo'] * pa_to_Mpa / self.wall_A
        sigzz_hi_t = sq.prop_uncertainty(fzL, fzU)['hi'] * pa_to_Mpa / self.wall_A

        sigxz_err = sq.prop_uncertainty(fx_Lower, fx_Upper)['err'] * pa_to_Mpa / self.chunk_A
        sigxz_lo = sq.prop_uncertainty(fx_Lower, fx_Upper)['lo'] * pa_to_Mpa / self.chunk_A
        sigxz_hi = sq.prop_uncertainty(fx_Lower, fx_Upper)['hi'] * pa_to_Mpa / self.chunk_A

        sigzz_err = sq.prop_uncertainty(fz_Lower, fz_Upper)['err'] * pa_to_Mpa / self.chunk_A
        sigzz_lo = sq.prop_uncertainty(fz_Lower, fz_Upper)['lo'] * pa_to_Mpa / self.chunk_A
        sigzz_hi = sq.prop_uncertainty(fz_Lower, fz_Upper)['hi'] * pa_to_Mpa / self.chunk_A

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
                'sigzz_wall_in':sigzz_wall_in,
                'sigxz_err': sigxz_err, 'sigxz_lo': sigxz_lo, 'sigxz_hi': sigxz_hi,
                'sigzz_err': sigzz_err, 'sigzz_lo': sigzz_lo, 'sigzz_hi': sigzz_hi,
                'sigxz_err_t': sigxz_err_t, 'sigxz_lo_t': sigxz_lo_t, 'sigxz_hi_t': sigxz_hi_t,
                'sigzz_err_t': sigzz_err_t, 'sigzz_lo_t' :sigzz_lo_t, 'sigzz_hi_t': sigzz_hi_t}

    def temp(self):
        """
        Computes the scalar kinetic temperature of the fluid as well as the
        x,y,z components of the temperature vector
        Units: K

        Returns
        -------
        temp_X : arr (Nx,), temperature along the length
        temp_Z : arr (Nz,), temperature along the gap height
        temp_t : arr (time,), temperature time-series
        tempX, tempY, tempZ : temperature x,y,z components from the correspondng velocity components
        """

        temp_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Temperature"])[self.skip:])
        tempX = np.mean(temp_full_x, axis=(0,2))

        temp_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["Temperature"])[self.skip:])
        tempZ = np.mean(temp_full_z, axis=(0,1))
        # First and last 2 chunks have large uncertainties
        if np.ma.count_masked(tempZ)!=0:
            first_non_masked = np.ma.flatnotmasked_edges(tempZ)[0]
            last_non_masked = np.ma.flatnotmasked_edges(tempZ)[1]
            tempZ[first_non_masked], tempZ[first_non_masked+1], \
            tempZ[last_non_masked], tempZ[last_non_masked-1] = np.nan, np.nan, np.nan, np.nan

        # print(np.isnan(temp_full_z).any())

        temp_t = np.mean(temp_full_x, axis=(1,2))

        try:
            # Temp in X-direction
            tempX_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["TemperatureX"])[self.skip:])  # Along length
            tempX_len = np.mean(tempX_full_x,axis=(0,2))
            tempX_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["TemperatureX"])[self.skip:])  # Along height
            tempX_height = np.mean(tempX_full_z,axis=(0,1))
            tempX_t = np.mean(tempX_full_x,axis=(1,2))

            # Temp in Y-direction
            tempY_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["TemperatureY"])[self.skip:])
            tempY_len = np.mean(tempY_full_x,axis=(0,2))
            tempY_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["TemperatureY"])[self.skip:])
            tempY_height = np.mean(tempY_full_z,axis=(0,1))
            tempY_t = np.mean(tempY_full_x,axis=(1,2))

            # Temp in Y-direction
            tempZ_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["TemperatureZ"])[self.skip:])
            tempZ_len = np.mean(tempZ_full_x,axis=(0,2))
            tempZ_full_z = self.mask_invalid_zeros(np.array(self.data_z.variables["TemperatureZ"])[self.skip:])
            tempZ_height = np.mean(tempZ_full_z,axis=(0,1))
            tempZ_t = np.mean(tempZ_full_x,axis=(1,2))

            # Temperature in the solid
            temp_full_x_solid = self.mask_invalid_zeros(np.array(self.data_x.variables["Temperature_solid"])[self.skip:])
            temp_full_z_solid = self.mask_invalid_zeros(np.array(self.data_z.variables["Temperature_solid"])[self.skip:])
            if np.mean(temp_full_x_solid, axis=(0,1)) != None:
                tempS_len = np.mean(temp_full_x_solid, axis=0)
                tempS_t = np.mean(temp_full_x_solid, axis=1)
            else:
                tempS_len, tempS_height, tempS_t= 0, 0, 0

            # Inlet and outlet of the pump region
            temp_bulk = np.mean(tempZ[40:-40])
            temp_walls = np.mean(tempS_t)

            temp_grad = (temp_walls - temp_bulk) / (0.5 * self.avg_gap_height * 1e-9)    # K/m

        except KeyError:
            tempX_full_x, tempX_full_z, tempX_len, tempX_height, tempX_t = 0,0,0,0,0
            tempY_full_x, tempY_full_z, tempY_len, tempY_height, tempY_t = 0,0,0,0,0
            tempZ_full_x, tempZ_full_z, tempZ_len, tempZ_height, tempZ_t = 0,0,0,0,0
            temp_full_x_solid, temp_full_z_solid, tempS_t, tempS_len = 0,0,0,0
            temp_grad = 1

        return {'temp_X':tempX, 'temp_Z':tempZ, 'temp_t':temp_t,
                'temp_full_x': temp_full_x, 'temp_full_z': temp_full_z,
                'temp_full_x_solid':temp_full_x_solid, 'temp_full_z_solid':temp_full_z_solid,
                'tempS_len':tempS_len, 'tempS_t':tempS_t, 'temp_grad':temp_grad,
                'tempX_full_x':tempX_full_x, 'tempX_full_z':tempX_full_z, 'tempX_len':tempX_len,
                'tempX_height':tempX_height, 'tempX_t':tempX_t,
                'tempY_full_x':tempY_full_x, 'tempY_full_z':tempY_full_z, 'tempY_len':tempY_len,
                'tempY_height':tempY_height, 'tempY_t':tempY_t,
                'tempZ_full_x':tempZ_full_x, 'tempZ_full_z':tempZ_full_z, 'tempZ_len':tempZ_len,
                'tempZ_height':tempZ_height, 'tempZ_t':tempZ_t}

    # Static and Dynamic properties------------------------------------------
    # -----------------------------------------------------------------------

    def vel_distrib(self):
        """
        Computes the x,y,z velocity distributions of from
            * Thermal velocity (computed in a 5x5x5 Angstrom^3 region n the box center) in simulations with Walls
            * Thermal velocity (computed in the whole fluid domain) in bulk simulations
        Units: m/s

        Returns
        -------
        values: arr, fluid velocities averaged over time for each atom/molecule
        probabilities : arr, probabilities of these velocities
        """

        try:    # bulk
            fluid_vx = np.array(self.data_x.variables["Fluid_Vx"]) *  Angstromperfs_to_mpers       # m/s
            fluid_vy = np.array(self.data_x.variables["Fluid_Vy"]) *  Angstromperfs_to_mpers       # m/s
            fluid_vz = np.array(self.data_x.variables["Fluid_Vz"]) *  Angstromperfs_to_mpers       # m/s
        except KeyError:    # walls
            fluid_vx = np.array(self.data_x.variables["Fluid_lte_Vx"]) *  Angstromperfs_to_mpers       # m/s
            fluid_vy = np.array(self.data_x.variables["Fluid_lte_Vy"]) *  Angstromperfs_to_mpers       # m/s
            fluid_vz = np.array(self.data_x.variables["Fluid_lte_Vz"]) *  Angstromperfs_to_mpers       # m/s

        fluid_v = np.sqrt(fluid_vx**2+fluid_vy**2+fluid_vz**2)

        values_x, probabilities_x = sq.get_err(fluid_vx)['values'], sq.get_err(fluid_vx)['probs']
        values_y, probabilities_y = sq.get_err(fluid_vy)['values'], sq.get_err(fluid_vy)['probs']
        values_z, probabilities_z = sq.get_err(fluid_vz)['values'], sq.get_err(fluid_vz)['probs']

        return {'vx_values': values_x, 'vx_prob': probabilities_x,
                'vy_values': values_y, 'vy_prob': probabilities_y,
                'vz_values': values_z, 'vz_prob': probabilities_z,
                'fluid_v':fluid_v}


    def green_kubo(self, arr1, arr2, arr3, t_corr):
        """
        Computes the Green-Kubo transport coefficients from the Autocorrelation functions of
        3 input arrays (arr1, arr2, arr3).
        The integral is up to time t_corr.

        Units: input array unit ** 2
        """

        # Correlation time is 20000 steps (100e3 fs)
        time_origins = [0]#, 60000, 100000, 150000] # correspond to 100e3, 300e3, 500e3 fs

        acf_xy = np.zeros((len(time_origins), t_corr))
        acf_xz = np.zeros((len(time_origins), t_corr))
        acf_yz = np.zeros((len(time_origins), t_corr))

        for i,t in enumerate(time_origins):
            # The Convolution Autocorrelation functions
            C_xy = np.correlate(arr1[t:], arr1[t:], 'full')
            C_xz = np.correlate(arr2[t:], arr2[t:], 'full')
            C_yz = np.correlate(arr3[t:], arr3[t:], 'full')
            # The statistical ACF
            # C_stat = np.array([1]+[np.corrcoef(Wxz_t[:-i], Wxz_t[i:])[0,1] for i in range(1, t_corr)])
            print('Numpy correlation done!')

            # Slice part of the positive part of the ACF
            acf_xy[i,:] = C_xy[len(C_xy)//2 : len(C_xy)//2 + t_corr] / (len(C_xy)//2)#/ C_xy[len(C_xy)//2]
            acf_xz[i,:] = C_xz[len(C_xz)//2 : len(C_xz)//2 + t_corr] / (len(C_xz)//2)#/ C_xz[len(C_xz)//2]
            acf_yz[i,:] = C_xy[len(C_yz)//2 : len(C_yz)//2 + t_corr] / (len(C_yz)//2)#/ C_yz[len(C_yz)//2]

        acf_xy_avg = np.mean(acf_xy, axis=0)
        acf_xz_avg = np.mean(acf_xz, axis=0)
        acf_yz_avg = np.mean(acf_yz, axis=0)
        acf = (acf_xy + acf_xz + acf_yz) / 3.
        # time = np.arange(0, t_corr)*5
        # np.savetxt('acf.txt', np.c_[time, acf],  delimiter=' ', header='time   acf')

        gamma_xy = np.array([simpson(acf_xy_avg[:n+1]) for n in range(len(acf_xy_avg) - 1)]) #np.array([trapezoid(acf_xy_avg[:n+1]) for n in range(len(acf_xy_avg)-1)]) #
        gamma_xz = np.array([simpson(acf_xz_avg[:n+1]) for n in range(len(acf_xz_avg) - 1)]) #np.array([trapezoid(acf_xz_avg[:n+1]) for n in range(len(acf_xz_avg)-1)]) #
        gamma_yz = np.array([simpson(acf_yz_avg[:n+1]) for n in range(len(acf_yz_avg) - 1)]) #np.array([trapezoid(acf_yz_avg[:n+1]) for n in range(len(acf_yz_avg)-1)]) #

        gamma = (gamma_xy + gamma_xz + gamma_yz) / 3.
        # np.savetxt('gamma.txt', np.c_[self.time[1:t_corr], gamma],  delimiter=' ', header='time   viscosity')
        print('Integral done!')

        return {'acf': acf, 'gamma':gamma,
                'gamma_xz': gamma_xz, 'gamma_xy': gamma_xy, 'gamma_yz': gamma_yz}


    def viscosity_gk_log(self, logfile, tcorr):
        """
        Computes the dynamic viscosity from an equilibrium MD simulation (from the log file)
        based on Green-Kubo relation - Autocorrelation function of the shear stress

        Several time origins (t0) are considered by moving the wave along time with np correlate
        correlation time (tcorr) is the time up to which the acf and its integral (viscosity) is computed

        Units: mPa.s
        """

        # Thermodynamic variables printed every
        logdata = extract_thermo.extract_data(logfile, self.skip)
        data = logdata[0]
        thermo_variables = logdata[1]

        thermo_out = data[:,thermo_variables.index('Step')][-1] - data[:,thermo_variables.index('Step')][-2]

        vol = self.Lx * self.Ly * self.Lz
        temp = np.mean(data[:,thermo_variables.index('c_tempFluid')])   # K
        pressure = (np.mean(data[:,thermo_variables.index('Pxx')]) +\
                    np.mean(data[:,thermo_variables.index('Pyy')]) +\
                    np.mean(data[:,thermo_variables.index('Pzz')])) / 3.
        print(f'Press:{pressure*0.101325:.2f} MPa and temp:{temp:.2f} K')

        prefac = sci.atm**2 * 1e3 * np.mean(vol) * 1e-27 * thermo_out * 1e-15 / (sci.k * temp)

        # Virial off-diagonal components
        Wxy_t = data[:,thermo_variables.index('Pxy')]
        Wxz_t = data[:,thermo_variables.index('Pxz')]
        Wyz_t = data[:,thermo_variables.index('Pyz')]

        # Get the viscosity along t_corr
        eta = self.green_kubo(Wxy_t, Wxz_t, Wyz_t, tcorr)['gamma'] * prefac

        return {'eta': eta, 'thermo_out': thermo_out}


    def viscosity_gk_traj(self, tcorr):
        """
        Computes the dynamic viscosity from an equilibrium MD simulation (from the trajectory)
        based on Green-Kubo relation - Autocorrelation function of the shear stress

        Several time origins (t0) are considered by moving the wave along time with np correlate
        correlation time (tcorr) is the time up to which the acf and its integral (viscosity) is computed

        Units: mPa.s
        """

        # Thermodynamic variables printed every
        thermo_out = 5 #self.time[1] - self.time[0]

        temp = np.mean(self.temp()['temp_t'])         # K
        prefac = sci.atm**2 * 1e3 * np.mean(self.vol) * 1e-27 * thermo_out * 1e-15 / (sci.k * temp)

        Wxy_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wxy"])[self.skip:])  # atm * A3
        Wxz_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wxz"])[self.skip:])
        Wyz_full_x = self.mask_invalid_zeros(np.array(self.data_x.variables["Wyz"])[self.skip:])

        # Virial off-diagonal components
        Wxy_t = np.sum(Wxy_full_x, axis=(1,2), dtype=np.float64) / (np.mean(self.vol)*1e3)     # atm
        Wxz_t = np.sum(Wxz_full_x, axis=(1,2), dtype=np.float64) / (np.mean(self.vol)*1e3)
        Wyz_t = np.sum(Wyz_full_x, axis=(1,2), dtype=np.float64) / (np.mean(self.vol)*1e3)

        # Get the viscosity along t_corr
        eta = self.green_kubo(Wxy_t, Wxz_t, Wyz_t, tcorr)['gamma'] * prefac

        return {'eta': eta}


    def viscosity_nemd(self):
        """
        Computes the dynamic viscosity and shear rate (and the uncertainties)
        from the corresponding velocity profile of a non-equilibrium MD simulation

        Units: mPa.s and s^-1 for the dynamic viscosity and shear rate, respectively
        """

        # Remove the first and last chunk along the gap height (high uncertainty)
        height = self.height_array
        velocity = np.ma.masked_invalid(self.velocity()['vx_Z'])
        velocity_full = self.velocity()['vx_full_z']

        # sd
        if self.pumpsize==0:
            coeffs_fit = np.ma.polyfit(height, velocity, 1)
            sigxz_avg = np.mean(self.sigwall()['sigxz_t']) * 1e9      # mPa
            shear_rate = coeffs_fit[0] * 1e9        # s^-1
            eta = sigxz_avg / shear_rate

            coeffs_fit_lo = np.ma.polyfit(height, sq.get_err(velocity_full)['lo'], 1)
            shear_rate_lo = coeffs_fit_lo[0] * 1e9
            eta_lo = sigxz_avg / shear_rate_lo

            coeffs_fit_hi = np.ma.polyfit(height, sq.get_err(velocity_full)['hi'], 1)
            shear_rate_hi = coeffs_fit_hi[0] * 1e9
            eta_hi = sigxz_avg / shear_rate_hi

        # pd, sp
        if self.pumpsize!=0:
            # Get the viscosity
            sigxz_avg = np.mean(self.sigwall()['sigxz_t']) * 1e9      # mPa
            pgrad = self.virial()['pGrad']            # MPa/m

            # Shear rate at the walls
            x=0
            coeffs_fit = np.ma.polyfit(height, velocity, 2)
            shear_rate = (coeffs_fit[0]* 2 * x - coeffs_fit[0] * self.avg_gap_height) * 1e9
            eta = sigxz_avg / shear_rate                # mPa.s

            # TODO: Fix this
            coeffs_fit_lo = np.ma.polyfit(height, sq.get_err(velocity_full)['lo'], 2)
            shear_rate_lo = (coeffs_fit[0]* 2 * x - coeffs_fit[0] * self.avg_gap_height) * 1e9
            eta_lo = sigxz_avg / shear_rate_lo          # mPa.s

            coeffs_fit_hi = np.ma.polyfit(height, sq.get_err(velocity_full)['hi'], 2)
            shear_rate_hi = (coeffs_fit[0]* 2 * x - coeffs_fit[0] * self.avg_gap_height) * 1e9
            eta_hi = sigxz_avg / shear_rate_hi          # mPa.s

        return {'shear_rate': shear_rate, 'eta': eta,
                'shear_rate_lo': shear_rate_lo, 'eta_lo': eta_lo,
                'shear_rate_hi': shear_rate_hi, 'eta_hi': eta_hi}


    def conductivity_gk_log(self, logfile, tcorr):
        """
        Computes the thermal conductivity (conductivity) from equilibrium MD
        based on Green-Kubo relation - Autocorrelation function of the of the heat flux.
        The heat flux vector components were computed according to Irving-Kirkwood expression
        during the simulation

        Units: J/mKs
        """

        logdata = extract_thermo.extract_data(logfile, self.skip)
        data = logdata[0]
        thermo_variables = logdata[1]

        thermo_out = data[:,thermo_variables.index('Step')][-1] - data[:,thermo_variables.index('Step')][-2]

        vol = self.Lx * self.Ly * self.Lz
        temp = np.mean(data[:,thermo_variables.index('c_tempFluid')])   # K

        # From post-processing we get kcal/mol * A/fs ---> J * m/s
        conv_to_Jmpers = (4184*1e-10)/(sci.N_A*1e-15)

        # Heat flux vector
        je_x = data[:,thermo_variables.index('v_jex')] * conv_to_Jmpers / np.mean(vol* 1e-27)  # J/m2.s
        je_y = data[:,thermo_variables.index('v_jey')] * conv_to_Jmpers / np.mean(vol* 1e-27)  # J/m2.s
        je_z = data[:,thermo_variables.index('v_jez')] * conv_to_Jmpers / np.mean(vol* 1e-27)  # J/m2.s

        temp = np.mean(data[:,thermo_variables.index('c_tempFluid')])   # K
        prefac =  1e-15 * np.mean(vol) * 1e-27 * thermo_out / (sci.k * temp**2)        # m3.s/(J*K)

        # Get the viscosity along t_corr
        lambda_time = self.green_kubo(je_x, je_y, je_z, tcorr)['gamma'] * prefac # W/mK

        return {'lambda_time': lambda_time, 'thermo_out': thermo_out}


    def heat_flux(self):
        """
        Computes the heat flux vector components according to Irving-Kirkwood expression

        Units: J/(m^2.s)
        """

        # From post-processing we get kcal/mol * A/fs ---> J * m/s
        conv_to_Jmpers = (4184*1e-10)/(sci.N_A*1e-15)

        # Heat flux vector
        je_x = self.mask_invalid_zeros(np.array(self.data_x.variables["JeX"])[self.skip:]) * conv_to_Jmpers / np.mean(self.vol* 1e-27)  # J/m2.s
        je_y = self.mask_invalid_zeros(np.array(self.data_x.variables["JeY"])[self.skip:]) * conv_to_Jmpers / np.mean(self.vol* 1e-27)  # J/m2.s
        je_z = self.mask_invalid_zeros(np.array(self.data_x.variables["JeZ"])[self.skip:]) * conv_to_Jmpers / np.mean(self.vol* 1e-27)  # J/m2.s

        return {'je_x':je_x, 'je_y':je_y, 'je_z':je_z}


    def conductivity_ecouple(self, logfile):
        """
        Computes the heat flow rate from the slope of the energy removed by the thermostat with time in NEMD
        LAMMPS keyword is Ecouple.
        and the corresponding z-component of thermal conductivity vector from Fourier's law

        Units: J/mKs and J/s for the thermal conductivty and heat flow rate, respectively
        """

        logdata = extract_thermo.extract_data(logfile, self.skip)
        data = logdata[0]
        thermo_variables = logdata[1]

        # Ecouple from the thermostat:
        delta_energy = data[:,thermo_variables.index('f_nvt')] # kcal/mol
        cut_ds = self.skip + 5000 # Consider the energy slope in a window of 10 ns
        cut_log = 5000
        # Heat flow rate: slope of the energy with time
        qdot, _  = np.polyfit(self.time[self.skip:cut_ds], delta_energy[:cut_log], 1)  # (kcal/mol) / fs
        qdot *= 4184 / (sci.N_A * 1e-15)    # J/s
        qdot_continuum = self.viscosity_nemd()['eta'] * 1e-3 * \
               self.viscosity_nemd()['shear_rate']**2 * 2 * self.wall_A * self.avg_gap_height * 1e-9

        # Heat flux
        j =  qdot / (2 * self.wall_A) # J/(m2.s)

        temp_grad = self.temp()['temp_grad']     # K/m
        print(f'Tgrad = {temp_grad:e} K/m')
        conductivity_z = -j / temp_grad

        return {'conductivity_z':conductivity_z, 'qdot':qdot, 'qdot_continuum':qdot_continuum}


    def conductivity_IK(self):
        """
        Computes the heat flow rate from the Irving-Kirkwood expression in NEMD
        and the corresponding z-component of thermal conductivity vector from Fourier's law

        Units: J/mKs and J/s for the thermal conductivty and heat flow rate, respectively
        """

        je_z = -self.heat_flux()['je_z']   # J/m2.s
        temp_grad = self.temp()['temp_grad']   # K/m
        print(f'Tgrad = {temp_grad:e} K/m')
        conductivity_z = -je_z / temp_grad

        # CORRECT: The wall area (the heat flux was calculated only in the stable region)
        stable_region_area = 0.4 * self.wall_A
        qdot = np.mean(je_z) * stable_region_area  # W

        qdot_continuum = self.viscosity_nemd()['eta'] * 1e-3 \
                    * self.viscosity_nemd()['shear_rate']**2 * stable_region_area \
                    * self.avg_gap_height * 1e-9

        conductivity_continuum = - self.viscosity_nemd()['eta'] * 1e-3 \
                    * self.viscosity_nemd()['shear_rate']**2 \
                    * self.avg_gap_height * 1e-9 / (temp_grad)   # W/(m.K)

        return {'conductivity_z':conductivity_z, 'conductivity_continuum':conductivity_continuum,\
                'qdot':qdot, 'qdot_continuum':qdot_continuum}


    def slip_length(self):
        """
        Computes the slip length and velocity (Slip velocity * Shear rate) from
        linear extrapolation of the velocity profile to zero velocity and returns
        the extrapolated data for plotting

        Units: nm and m/s for length and velocity, respectively
        """

        # Remove the first and last chunk along the gap height (high uncertainty)
        height = self.height_array
        velocity = np.ma.masked_invalid(self.velocity()['vx_Z'])

        if self.pumpsize==0:
            fit_data = funcs.fit(height, velocity, 1)['fit_data']
        else:
            fit_data = funcs.fit(height, velocity, 2)['fit_data']

        npoints = len(self.height_array[velocity!=0])
        # Positions to inter/extrapolate
        xdata_left = np.linspace(-1, height[0], npoints)
        xdata_right = np.linspace(height[-1], 12 , npoints)

        # spline order: 1 linear, 2 quadratic, 3 cubic ...
        order = 1
        # do inter/extrapolation
        extrapolate = InterpolatedUnivariateSpline(height, fit_data, k=order)
        coeffs_extrapolate_left = np.polyfit(xdata_left, extrapolate(xdata_left), 1)
        coeffs_extrapolate_right = np.polyfit(xdata_right, extrapolate(xdata_right), 1)

        # Where the velocity profile vanishes
        root_left = np.roots(coeffs_extrapolate_left)
        root_right = np.roots(coeffs_extrapolate_right)

        # If the root is positive or very small, there is much noise in the velocity profile
        if root_left > 0 or np.abs(root_left) < 0.1 : root_left = 0

        # Slip length is the extrapolated length from the velocity profile fit
        b = np.abs(root_left)
        # Slip velocity
        Vs = velocity[1]
        # # Slip velocity according to Navier boundary
        # Vs = b * sci.nano * self.viscosity_nemd()['shear_rate']        # m/s

        return {'root_left':root_left, 'b':b, 'Vs':Vs,
                'xdata_left':xdata_left,
                'extrapolate_left':extrapolate(xdata_left),
                'root_right':root_right,
                'xdata_right':xdata_right,
                'extrapolate_right':extrapolate(xdata_right)}


    def coexistence_densities(self):
        """
        Computes the Liquid and vapor densities densities in equilibrium
        Liquid/Vapor interface simulations

        Units: g/cm^3
        """

        density = self.density()['den_Z']

        rho_liquid = np.max(density)
        rho_vapour = np.min(self.density()['den_Z'][10:-10])

        return {'rho_l':rho_liquid, 'rho_v':rho_vapour}


    def reynolds_num(self):
        """
        Computes the Reynolds number of the flow
        """

        rho = np.mean(self.density()['den_t']) * 1e3 # kg/m3
        u = np.mean(self.velocity()['vx_t']) # m/s
        h = self.avg_gap_height * 1e-9 # m
        eta = self.viscosity_nemd()['eta'] * 1e-3 # Pa.s

        Re = rho * u * h / eta

        return Re


    def cavitation_num(self, pl, pv):
        """
        Computes the cavitation number of the flow
        """

        rho = np.max(self.density()['den_X']) * 1e3 # Liquid density (kg/m3)
        u = self.velocity()['vx_X'] # m/s

        K = (pl-pv)/(0.5*rho*u**2)

        return K


    def surface_tension(self):
        """
        Computes the surface tension (γ) from an equilibrium Liquid/Vapor interface
        The pressure tensor is computed from the "virial" method using IK expression

        Units: mN/m
        """

        virxx, viryy, virzz = -self.virial()['Wxx_t'], -self.virial()['Wyy_t'], -self.virial()['Wzz_t']   # MPa
        if self.avg_gap_height !=0: # Walls
            gamma = 0.5 * self.avg_gap_height * (virzz - (0.5*(virxx+viryy)) ) * 1e6 * 1e-9
        else:
            gamma = 0.5 * self.Lz * (virzz - (0.5*(virxx+viryy)) ) * 1e6 * 1e-9

        return {'gamma':gamma}
