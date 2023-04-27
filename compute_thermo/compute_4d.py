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

    def __init__(self, skip, infile, mf, pumpsize):
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
        self.infile = infile
        self.data = netCDF4.Dataset(self.infile)
        self.mf = mf
        if self.mf == 39.948 : self.A_per_molecule = 1. # Lennard-Jones
        if self.mf == 44.09 : self.A_per_molecule = 3  # propane
        if self.mf == 72.15: self.A_per_molecule = 5.   # Pentane
        if self.mf == 100.21 : self.A_per_molecule = 7  # heptane
        if self.mf == 442.83 : self.A_per_molecule = 30  # squalane
        self.pumpsize = pumpsize
        # self.Nf = len(self.data.dimensions["Nf"])     # No. of fluid atoms

        # Time
        self.time =  np.array(self.data.variables["Time"])
        if not self.time.size:
            raise ValueError('The array is empty! Reduce the skipped timesteps.')

        # Box spatial dimensions ------------------------------------------
        dim = self.data.__dict__
        self.Lx = dim["Lx"] / 10      # nm
        self.Ly = dim["Ly"] / 10      # nm
        self.Lz = dim["Lz"] / 10      # nm

        # Number of chunks
        self.Nx = self.data.dimensions['x'].size
        self.Ny = self.data.dimensions['y'].size
        self.Nz = self.data.dimensions['z'].size

        # Gap heights (time,) in nm
        self.h = np.array(self.data_x.variables["Height"])[self.skip:] / 10
        if len(self.h) == 0:
            raise ValueError('The array is empty! Reduce the skipped timesteps.')
        # Average gap height
        self.avg_gap_height = np.mean(self.h)
        self.h_conv = np.array(self.data_x.variables["Height_conv"])[self.skip:] / 10
        self.h_div = np.array(self.data_x.variables["Height_div"])[self.skip:] / 10
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

        # The width array
        dy = self.Ly/ self.Ny

        # The total gap height array
        dz = self.avg_gap_height / self.Nz
        self.height_array = np.arange(dz/2.0, self.avg_gap_height, dz)

        # The bulk region gap height array
        self.bulk_height_array = np.linspace(avg_bulkStart, avg_bulkEnd , self.Nz)


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

        return masked_array

    def remove_first_last(self, array):
        """
        Remove the first and last spatial bins
        """

        if np.ma.count_masked(array)!=0:
            first_non_masked = np.ma.flatnotmasked_edges(array)[0]
            last_non_masked = np.ma.flatnotmasked_edges(array)[1]
            array[first_non_masked], array[last_non_masked] = np.nan, np.nan
        else:
            array[0], array[-1] = np.nan, np.nan

        return array

    # Thermodynamic properties-----------------------------------------------
    # -----------------------------------------------------------------------

    def velocity(self):
        """
        Computes the center of mass (streaming or x-component) velocity of the fluid
        Units: m/s

        Returns
        -------
        vx_X : arr (Nx,), velocity along the length
        vx_Z : arr (Nz,), velocity along the gap height
        vx_t : arr (time,), velocity time-series
        vx_R : arr (Nz,), velocity in different regions along the stream
        """

        # Measured in the whole fluid
        data = self.mask_invalid_zeros(np.array(self.data.variables["Vx_whole"])[self.skip:]) * Angstromperfs_to_mpers  # m/s
        # Measured away from the pump (Stable region)
        # vx_full_stable = np.array(self.data.variables["Vx"])[self.skip:] * Angstromperfs_to_mpers  # m/s

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


    def mflux(self):
        """
        Computes the x-component of the mass flux and the mass flow rate of the fluid
        Units: g/(m^2.ns) and g/s for the mass flux and mass flow rate, respectively

        Returns
        -------
        jx_X : arr (Nx,), mass flux along the length
        jx_Z : arr (Nz,), mass flux along the gap height
        jx_t : arr (time,), mass flux time-series
        jx_stable : arr (time,), mass flux time-series only in the stable region (defined in LAMMPS post-processing script)
        jx_pump : arr (time,), mass flux time-series only in the pump region (defined in LAMMPS post-processing script)
        """

        # Measure the mass flux in the full simulation domain
        data = self.mask_invalid_zeros(np.array(self.data.variables["Jx"])[self.skip:]) / (sci.N_A * fs_to_ns * sci.angstrom**2)

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


    def mflowrate(self):
        """
        Computes the x-component of the mass flow rate of the fluid
        Units: g/ns

        Returns
        -------
        mflowrate_X : arr (Nx,), mass flow rate along the length
        mflowrate_t : arr (time,), mass flow rate time-series
        mflowrate_stable : arr (time,), mass flow rate time-series only in the stable region (defined in LAMMPS post-processing script)
        mflowrate_pump : arr (time,), mass flow rate time-series only in the pump region (defined in LAMMPS post-processing script)
        """

        # Measure the mass flux in the full simulation domain
        data = self.mask_invalid_zeros(np.array(self.data.variables["mdot"])[self.skip:]) / (sci.N_A * fs_to_ns)

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


    def density(self):
        """
        Computes the mass density of the fluid
        Units: g/cm^3

        Returns
        -------
        den_X : arr (Nx,), BULK density along the length
        den_Z : arr (Nz,), fluid density along the gap height
        den_t : arr (time,), BULK density time-series
        """

        data = self.mask_invalid_zeros(np.array(self.data.variables["Density"])[self.skip:]) / (sci.N_A * ang_to_cm**3)

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


    def density_bulk(self):
        """
        Computes the mass density of the fluid
        Units: g/cm^3

        Returns
        -------
        den_X : arr (Nx,), BULK density along the length
        den_Z : arr (Nz,), fluid density along the gap height
        den_t : arr (time,), BULK density time-series
        """

        data = self.mask_invalid_zeros(np.array(self.data.variables["Density_Bulk"])[self.skip:]) / (sci.N_A * ang_to_cm**3)

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


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

        # Diagonal components
        Wxx_full = self.mask_invalid_zeros(np.array(self.data.variables["Wxx"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wyy_full = self.mask_invalid_zeros(np.array(self.data.variables["Wyy"])[self.skip:]) * sci.atm * pa_to_Mpa
        Wzz_full = self.mask_invalid_zeros(np.array(self.data.variables["Wzz"])[self.skip:]) * sci.atm * pa_to_Mpa

        if np.isclose(np.mean(Wxx_full, axis=(0,1,2,3)), np.mean(Wyy_full, axis=(0,1,2,3)), rtol=0.1, atol=0.0): # Incompressible flow
            print('Virial computed from the three components')
            data = -(Wxx_full + Wyy_full + Wzz_full) / 3.
        else:  # Compressible flow
            print('Virial computed from the y-component')
            data = - Wyy_full

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        if self.avg_gap_height == 0:
            data = self.mask_invalid_zeros(np.array(self.data.variables["Virial"])[self.skip:]) * sci.atm * pa_to_Mpa
            data_xz = self.remove_first_last(np.mean(data, axis=(0,2))) / (self.vol*1e3)
            data_xy = self.remove_first_last(np.mean(data, axis=(0,3))) / (self.vol*1e3)
            data_xt = self.remove_first_last(np.mean(data, axis=(2,3))) / (self.vol*1e3)
            data_yz = self.remove_first_last(np.mean(data, axis=(0,1))) / (self.vol*1e3)
            data_yt = self.remove_first_last(np.mean(data, axis=(1,3))) / (self.vol*1e3)
            data_zt = self.remove_first_last(np.mean(data, axis=(1,2))) / (self.vol*1e3)

        return {'data':data, 'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


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

        data = self.mask_invalid_zeros(np.array(self.data.variables["Temperature"])[self.skip:])

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}


    def temp_solid(self):
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

        data = self.mask_invalid_zeros(np.array(self.data.variables["Temperature_solid"])[self.skip:])

        data_xz = self.remove_first_last(np.mean(data, axis=(0,2)))
        data_xy = self.remove_first_last(np.mean(data, axis=(0,3)))
        data_xt = self.remove_first_last(np.mean(data, axis=(2,3)))
        data_yz = self.remove_first_last(np.mean(data, axis=(0,1)))
        data_yt = self.remove_first_last(np.mean(data, axis=(1,3)))
        data_zt = self.remove_first_last(np.mean(data, axis=(1,2)))

        return {'data_xz':data_xz, 'data_xy':data_xy, 'data_xt':data_xt,
                'data_yz':data_yz, 'data_yt':data_yt, 'data_zt':data_zt}
