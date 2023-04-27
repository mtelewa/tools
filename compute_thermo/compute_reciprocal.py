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
        self.nx = self.data.dimensions['nx'].size
        self.ny = self.data.dimensions['ny'].size

    # Thermodynamic properties-----------------------------------------------
    # -----------------------------------------------------------------------

    def sf(self):
        """
        Computes the longitudnal (skx) and transverse (sky) structure factors for the liquid and solid
        """

        # wavevectors
        kx = np.array(self.data.variables["kx"])
        ky = np.array(self.data.variables["ky"])

        # For the fluid
        sf_real = np.array(self.data.variables["sf"])[self.skip:]
        # Structure factor averaged over time for each k
        sf = np.mean(sf_real, axis=0)
        sf_time = np.mean(sf_real, axis=(1,2))

        # For the solid
        sf_real_solid = np.array(self.data.variables["sf_solid"])[self.skip:]
        sf_solid = np.mean(sf_real_solid, axis=0)

        return {'kx':kx, 'ky':ky, 'sf':sf, 'sf_time':sf_time, 'sf_solid':sf_solid}


    def ISF(self):
        """
        Computes the intermediate scattering function
        TODO: Check that it works properly!
        """

        # Fourier components of the density
        rho_k = np.array(self.data.variables["rho_k"])
        ISF = sq.acf_conjugate(rho_k)['norm']

        # a = np.mean(rho_k, axis=0)
        # var = np.var(rho_k, axis=0)


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


    # Time-correlation functions ------------------------------------------------

    def transverse_acf(self):
        """
        Computes the transverse Autocorrelation function
        TODO: Check that it works properly!
        """

        density = np.array(self.data.variables["Density"])[self.skip:1000] * (self.mf/sci.N_A) / (ang_to_cm**3)

        gridsize = np.array([self.Nx])
        # axes to perform FFT over
        fft_axes = np.where(gridsize > 1)[0] + 1

        # permutation of FFT output axes
        permutation = np.arange(3, dtype=int)
        permutation[fft_axes] = fft_axes[::-1]

        rho_tq = np.fft.fftn(density, axes=fft_axes).transpose(*permutation)

        acf_rho = sq.acf_fft(rho_tq)

        nmax = (min(gridsize[gridsize > 1]) - 1) // 2

        acf_rho_nc = np.zeros([len(self.time)-34000, 1, nmax], dtype=np.float32)

        for ax in fft_axes:
            # mask positive wavevectors
            pmask = list(np.ones(4, dtype=int) * 0)
            pmask[0] = slice(len(self.time-34000))
            pmask[ax] = slice(1, nmax + 1)
            pmask = tuple(pmask)

            # mask negative wavevectors
            nmask = list(np.ones(4, dtype=int) * 0)
            nmask[0] = slice(len(self.time-34000))
            nmask[ax] = slice(-1, -nmax - 1, -1)
            nmask = tuple(nmask)

            # fill output buffers with average of pos. and neg. wavevectors
            acf_rho_nc[:, ax-1, :] = (acf_rho[pmask] + acf_rho[nmask]) / 2

        # Fourier transform of the ACF > Spectrum density
        # spec_density = np.fft.fft(acf)
        # Inverse DFT
        # acf = np.fft.ifftn(spec_density,axes=(0,1))

    def trans(self):
        """
        Computes the transverse Autocorrelation function ?!
        TODO: Check that it works properly!
        """

        jx = np.array(self.data.variables["Jx"])[self.skip:10000]
        # Fourier transform in the space dimensions
        jx_tq = np.fft.fftn(jx, axes=(1,2))

        acf = sq.acf(jx_tq)['norm']

        return {'a':acf[:, 0, 0].real}
