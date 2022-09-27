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
from math import isclose

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
        # Old simulations have no Lz dimension (Only gap height)
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
            try:
                bulkStart = np.array(self.data_x.variables["Bulk_Start"])[self.skip:] / 10 #
                bulkEnd = np.array(self.data_x.variables["Bulk_End"])[self.skip:] / 10 #
                avg_bulkStart, avg_bulkEnd = np.mean(bulkStart), np.mean(bulkEnd)
                self.bulk_height_array = np.linspace(avg_bulkStart, avg_bulkEnd , self.Nz)     #nm

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
        Computes the center of mass (streaming or x-component) velocity of the fluid
        Units: m/s

        Returns
        -------
        vx_X : arr (Nx,), velocity along the length
        vx_Z : arr (Nz,), velocity along the gap height
        vx_t : arr (time,), velocity time-series
        vx_R : arr (Nz,), velocity in different regions along the stream
        """

        vx_full_x = np.array(self.data_x.variables["Vx"])[self.skip:] * Angstromperfs_to_mpers  # m/s
        vx_full_z = np.array(self.data_z.variables["Vx"])[self.skip:] * Angstromperfs_to_mpers  # m/s

        try:
            vx_full_x_whole = np.array(self.data_x.variables["Vx_whole"])[self.skip:] * Angstromperfs_to_mpers  # m/s
        except KeyError:
            vx_full_x_whole = 0

        vx_chunkX = np.mean(vx_full_x, axis=(0,2))
        vx_chunkZ = np.mean(vx_full_z, axis=(0,1))
        vx_t = np.mean(vx_full_x, axis=(1,2))

        try:
            vx_R1_z = np.array(self.data_z.variables["Vx_R1"])[self.skip:] * Angstromperfs_to_mpers  # m/s
            vx_R2_z = np.array(self.data_z.variables["Vx_R2"])[self.skip:] * Angstromperfs_to_mpers  # m/s
            vx_R3_z = np.array(self.data_z.variables["Vx_R3"])[self.skip:] * Angstromperfs_to_mpers  # m/s
            vx_R4_z = np.array(self.data_z.variables["Vx_R4"])[self.skip:] * Angstromperfs_to_mpers  # m/s
            vx_R5_z = np.array(self.data_z.variables["Vx_R5"])[self.skip:] * Angstromperfs_to_mpers  # m/s

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
        jx_full_x = np.array(self.data_x.variables["Jx"])[self.skip:]
        jx_full_z = np.array(self.data_z.variables["Jx"])[self.skip:]
        jx_chunkX = np.mean(jx_full_x, axis=(0,2)) * (sci.angstrom/fs_to_ns) * (self.mf/sci.N_A) / (sci.angstrom**3)
        jx_chunkZ = np.mean(jx_full_z, axis=(0,1)) * (sci.angstrom/fs_to_ns) * (self.mf/sci.N_A) / (sci.angstrom**3)
        jx_t = np.sum(jx_full_x, axis=(1,2)) #) + np.mean(jx_full_z, axis=(1,2))) / 2.
        jx_t *= (sci.angstrom/fs_to_ns) * (self.mf/sci.N_A) / (sci.angstrom**3)

        # Measure the mass flow rate whole domain
        try:
            mflowrate_full_x = np.array(self.data_x.variables["mdot"])[self.skip:] * Angstromperfs_to_mpers
            mflowrate_chunkX = (self.mf/sci.N_A) * np.mean(mflowrate_full_x, axis=(0,2)) * 1e-9 / (self.Lx*1e-9/self.Nx)
            # Time-averaged mass flow rate in the whole domain
            mflowrate_t = (self.mf/sci.N_A) * np.sum(mflowrate_full_x, axis=(1,2)) / (self.Lx*1e-9)
        except KeyError:
            mflowrate_full_x, mflowrate_chunkX, mflowrate_t = 0 ,0 ,0

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
        mu = self.transport()['mu'] * 1e-3            # Pa.s
        Ly = self.Ly * 1e-9
        kgpers_to_gperns =  1e-9 * 1e3

        # Without slip
        mflowrate_hp = (bulk_den_avg * Ly * pGrad * avg_gap_height**3 * kgpers_to_gperns) / (12 * mu)

        # With slip
        mflowrate_hp_slip = mflowrate_hp + (bulk_den_avg * Ly * kgpers_to_gperns * \
                            np.mean(self.slip_length()['Vs']) * avg_gap_height)

        if self.pumpsize == 0:
            mflowrate_hp = (bulk_den_avg * Ly * np.float(input('pGrad:')) * avg_gap_height**3 * kgpers_to_gperns) / (12 * mu)

        return {'mflowrate_hp':mflowrate_hp, 'mflowrate_hp_slip':mflowrate_hp_slip}


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

        # Bulk Density ---------------------
        density_Bulk = np.array(self.data_x.variables["Density_Bulk"])[self.skip:] * (self.mf/sci.N_A) / (ang_to_cm**3)    # g/cm^3
        den_X = np.mean(density_Bulk, axis=0)    # g/cm^3

        if np.mean(self.h) == 0: # Bulk simulation
            Nm = len(self.data_x.dimensions["Nf"]) / self.A_per_molecule     # No. of fluid molecules
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

        # Voronoi volume
        try:
            totVi = np.array(self.data_x.variables["Voronoi_volumes"])[self.skip:]
        except KeyError:
            pass

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

            Wxy_t = np.mean(Wxy_full_x, axis=(1,2))
            Wxz_t = np.mean(Wxz_full_z, axis=(1,2))
            Wyz_t = np.mean(Wyz_full_z, axis=(1,2))

        except KeyError:
            Wxy_full_x, Wxz_full_x, Wyz_full_x, Wxy_full_z, Wxz_full_z, Wyz_full_z, Wxy_chunkX, Wxz_chunkX, Wyz_chunkX, \
            Wxy_chunkZ, Wxz_chunkZ, Wyz_chunkZ, Wxy_t, Wxz_t, Wyz_t = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

        # Diagonal components (separately)
        try:
            Wxx_full_x = np.array(self.data_x.variables["Wxx"])[self.skip:] * sci.atm * pa_to_Mpa # Units: MPa if walls and MPa.A3 if bulk
            Wyy_full_x = np.array(self.data_x.variables["Wyy"])[self.skip:] * sci.atm * pa_to_Mpa
            Wzz_full_x = np.array(self.data_x.variables["Wzz"])[self.skip:] * sci.atm * pa_to_Mpa

            Wxx_full_z = np.array(self.data_z.variables["Wxx"])[self.skip:] * sci.atm * pa_to_Mpa
            Wyy_full_z = np.array(self.data_z.variables["Wyy"])[self.skip:] * sci.atm * pa_to_Mpa
            Wzz_full_z = np.array(self.data_z.variables["Wzz"])[self.skip:] * sci.atm * pa_to_Mpa

            # Only for systems with walls
            Wxx_chunkX = np.mean(Wxx_full_x, axis=(0,2))
            Wyy_chunkX = np.mean(Wyy_full_x, axis=(0,2))
            Wzz_chunkX = np.mean(Wzz_full_x, axis=(0,2))

            Wxx_chunkZ = np.mean(Wxx_full_z, axis=(0,1))
            Wyy_chunkZ = np.mean(Wyy_full_z, axis=(0,1))
            Wzz_chunkZ = np.mean(Wzz_full_z, axis=(0,1))

            if np.mean(self.h) != 0:
                Wxx_t = np.mean(Wxx_full_x, axis=(1,2))
                Wyy_t = np.mean(Wyy_full_z, axis=(1,2))
                Wzz_t = np.mean(Wzz_full_z, axis=(1,2))
                # Diagonal components
                # If in LAMMPS we switched off the flow direction to compute the virial
                # and used only the y-direction (perp. to flow perp. to loading), then
                # we conisder only that direction in the virial calcualtion..
                if isclose(np.mean(Wxx_full_x, axis=(0,1,2)), np.mean(Wyy_full_x, axis=(0,1,2)), rel_tol=0.2, abs_tol=0.0):
                    vir_full_x = -(Wxx_full_x + Wyy_full_x + Wzz_full_x) / 3.
                    vir_full_z = -(Wxx_full_z + Wyy_full_z + Wzz_full_z) / 3.
                else:
                    vir_full_x = -Wyy_full_x
                    vir_full_z = -Wyy_full_z

                vir_t = np.mean(vir_full_x, axis=(1,2))

                vir_chunkX = np.mean(vir_full_x, axis=(0,2))
                vir_chunkZ = np.mean(vir_full_z, axis=(0,1))
                vir_fluctuations = sq.get_err(vir_full_x)['var']

            else: # Bulk simulation (Overall pressure)
                Wxx_t = np.sum(Wxx_full_x, axis=(1)) / (self.vol*1e3)
                Wyy_t = np.sum(Wyy_full_x, axis=(1)) / (self.vol*1e3)
                Wzz_t = np.sum(Wzz_full_x, axis=(1)) / (self.vol*1e3)

                # TODO: Correct these by diving by volume of the slab in each time step (in post-processing script)
                vir_full_x = -(Wxx_full_x + Wyy_full_x + Wzz_full_x) / 3.
                vir_full_z = -(Wxx_full_z + Wyy_full_z + Wzz_full_z) / 3.
                vir_chunkX, vir_chunkZ  = np.mean(vir_full_x, axis=(0,2)), np.mean(vir_full_z, axis=(0,1))

                vir_t = -(Wxx_t + Wyy_t + Wzz_t) / 3.
                # pGrad, pDiff, p_ratio = 0,0,0

        # The virial pressure (scalar) was computed in the post-processing script
        except KeyError:
            vir_full_x = np.array(self.data_x.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa
            vir_full_z = np.array(self.data_z.variables["Virial"])[self.skip:] * sci.atm * pa_to_Mpa
            vir_chunkX = np.mean(vir_full_x, axis=(0,2))
            vir_chunkZ = np.mean(vir_full_z, axis=(0,1))
            vir_t = np.mean(vir_full_x, axis=(1,2))

            Wxx_full_x, Wyy_full_x, Wzz_full_x, Wxx_full_z, Wyy_full_z, Wzz_full_z, Wxx_chunkX, Wyy_chunkX, Wzz_chunkX, \
            Wxx_chunkZ, Wyy_chunkZ, Wzz_chunkZ, Wxx_t, Wyy_t, Wzz_t = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0


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
                'p_ratio':p_ratio, 'vir_full_x': vir_full_x, 'vir_full_z': vir_full_z}


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
        fzL, fzU = np.sum(fz_Lower,axis=1) , np.sum(fz_Upper,axis=1)
        sigxz_err_t = sq.prop_uncertainty(fxL, fxU)['err'] * pa_to_Mpa / self.wall_A
        sigxz_lo_t = sq.prop_uncertainty(fxL, fxU)['lo'] * pa_to_Mpa / self.wall_A
        sigxz_hi_t = sq.prop_uncertainty(fxL, fxU)['hi'] * pa_to_Mpa / self.wall_A

        sigzz_err_t = sq.prop_uncertainty(fzL, fzU)['err'] * pa_to_Mpa / self.wall_A
        sigzz_lo_t = sq.prop_uncertainty(fzL, fzU)['lo'] * pa_to_Mpa / self.wall_A
        sigzz_hi_t = sq.prop_uncertainty(fzL, fzU)['hi'] * pa_to_Mpa / self.wall_A

        # Calculation of stress in the chunks
        sigxz_chunkX = np.mean(fx_wall,axis=0) * pa_to_Mpa / self.chunk_A
        sigyz_chunkX = np.mean(fy_wall,axis=0) * pa_to_Mpa / self.chunk_A
        sigzz_chunkX = np.mean(fz_wall,axis=0) * pa_to_Mpa / self.chunk_A

        # Error calculation for each chunk
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

        temp_full_x = np.array(self.data_x.variables["Temperature"])[self.skip:]
        temp_full_z = np.array(self.data_z.variables["Temperature"])[self.skip:]

        tempX = np.mean(temp_full_x,axis=(0,2))
        tempZ = np.mean(temp_full_z,axis=(0,1))
        temp_t = np.mean(temp_full_x,axis=(1,2)) # Do not include the wild oscillations along the height

        # Inlet and outlet of the pump region
        temp_bulk = np.max(tempZ)
        temp_walls = 300 # K #TODO: should be Computesd
        temp_ratio = temp_walls / temp_bulk
        # print(f'Temp. at the walls {temp_walls} and in the bulk {temp_bulk}')
        # pd_length = self.Lx - self.pumpsize * self.Lx      # nm
        # pump_length = self.pumpsize * self.Lx      # nm

        temp_grad = (temp_walls - temp_bulk) * 1e9 / 1     # K/m        Here: assumed that the bulk temp is 1 nm away from the walls.

        try:
            temp_full_x_solid = np.array(self.data_x.variables["Temperature_solid"])[self.skip:]
            temp_full_z_solid = np.array(self.data_z.variables["Temperature_solid"])[self.skip:]
            tempS_len = np.mean(temp_full_x_solid, axis=0) # np.mean(temp_full_x_solid, axis=(0,2))  #
            tempS_height = 0 # np.mean(temp_full_z_solid, axis=(0,1))
            tempS_t = np.mean(temp_full_x_solid, axis=1) #np.mean(temp_full_x_solid, axis=(1,2))  #
        except KeyError:
            temp_full_x_solid, temp_full_z_solid, tempS_t, tempS_len, tempS_height = 0,0,0,0,0
            # pass

        try:
            # Temp in X-direction
            tempX_full_x = np.array(self.data_x.variables["TemperatureX"])[self.skip:]  # Along length
            tempX_full_z = np.array(self.data_z.variables["TemperatureX"])[self.skip:]  # Along height
            tempX_len = np.mean(tempX_full_x,axis=(0,2))
            tempX_height = np.mean(tempX_full_z,axis=(0,1))
            tempX_t = np.mean(tempX_full_x,axis=(1,2)) # Do not include the wild oscillations along the height

            # Temp in Y-direction
            tempY_full_x = np.array(self.data_x.variables["TemperatureY"])[self.skip:]
            tempY_full_z = np.array(self.data_z.variables["TemperatureY"])[self.skip:]
            tempY_len = np.mean(tempY_full_x,axis=(0,2))
            tempY_height = np.mean(tempY_full_z,axis=(0,1))
            tempY_t = np.mean(tempY_full_x,axis=(1,2)) # Do not include the wild oscillations along the height

            # Temp in Y-direction
            tempZ_full_x = np.array(self.data_x.variables["TemperatureZ"])[self.skip:]
            tempZ_full_z = np.array(self.data_z.variables["TemperatureZ"])[self.skip:]
            tempZ_len = np.mean(tempZ_full_x,axis=(0,2))
            tempZ_height = np.mean(tempZ_full_z,axis=(0,1))
            tempZ_t = np.mean(tempZ_full_x,axis=(1,2)) # Do not include the wild oscillations along the height

        except KeyError:
            tempX_full_x, tempX_full_z, tempX_len, tempX_height, tempX_t = 0,0,0,0,0
            tempY_full_x, tempY_full_z, tempY_len, tempY_height, tempY_t = 0,0,0,0,0
            tempZ_full_x, tempZ_full_z, tempZ_len, tempZ_height, tempZ_t = 0,0,0,0,0

        # try:
        return {'temp_X':tempX, 'temp_Z':tempZ, 'temp_t':temp_t, 'temp_ratio':temp_ratio,
                'temp_full_x': temp_full_x, 'temp_full_z': temp_full_z,
                'temp_full_x_solid':temp_full_x_solid, 'temp_full_z_solid':temp_full_z_solid,
                'tempS_len':tempS_len, 'tempS_height':tempS_height,
                'tempS_t':tempS_t, 'temp_grad':temp_grad,
                'tempX_full_x':tempX_full_x, 'tempX_full_z':tempX_full_z, 'tempX_len':tempX_len,
                'tempX_height':tempX_height, 'tempX_t':tempX_t,
                'tempY_full_x':tempY_full_x, 'tempY_full_z':tempY_full_z, 'tempY_len':tempY_len,
                'tempY_height':tempY_height, 'tempY_t':tempY_t,
                'tempZ_full_x':tempZ_full_x, 'tempZ_full_z':tempZ_full_z, 'tempZ_len':tempZ_len,
                'tempZ_height':tempZ_height, 'tempZ_t':tempZ_t}
        # except NameError:
        #     return {'temp_X':tempX, 'temp_Z':tempZ, 'temp_t':temp_t, 'temp_ratio':temp_ratio,
        #             'temp_full_x': temp_full_x, 'temp_full_z': temp_full_z, 'temp_grad':temp_grad}


    # Static and Dynamic properties------------------------------------------
    # -----------------------------------------------------------------------

    def vel_distrib(self):
        """
        Computes the x,y,z velocity distributions of from
            * Streaming velocity (computed in the whole fluid) and
            * Thermal velocity (computed in a 5x5x5 Angstrom^3 region n the box center)
        Units: m/s

        Returns
        -------
        values: arr, fluid velocities averaged over time for each atom/molecule
        probabilities : arr, probabilities of these velocities
        """

        fluid_vx = np.array(self.data_x.variables["Fluid_Vx"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s
        fluid_vy = np.array(self.data_x.variables["Fluid_Vy"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s
        fluid_vz = np.array(self.data_x.variables["Fluid_Vz"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s

        fluid_vx_thermal = np.array(self.data_x.variables["Fluid_lte_Vx"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s
        fluid_vy_thermal = np.array(self.data_x.variables["Fluid_lte_Vy"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s
        fluid_vz_thermal = np.array(self.data_x.variables["Fluid_lte_Vz"])[self.skip:] *  A_per_fs_to_m_per_s       # m/s

        values_x, probabilities_x = sq.get_err(fluid_vx)['values'], sq.get_err(fluid_vx)['probs']
        values_y, probabilities_y = sq.get_err(fluid_vy)['values'], sq.get_err(fluid_vy)['probs']
        values_z, probabilities_z = sq.get_err(fluid_vz)['values'], sq.get_err(fluid_vz)['probs']

        values_x_thermal, probabilities_x_thermal = sq.get_err(fluid_vx_thermal)['values'], sq.get_err(fluid_vx_thermal)['probs']
        values_y_thermal, probabilities_y_thermal = sq.get_err(fluid_vy_thermal)['values'], sq.get_err(fluid_vy_thermal)['probs']
        values_z_thermal, probabilities_z_thermal = sq.get_err(fluid_vz_thermal)['values'], sq.get_err(fluid_vz_thermal)['probs']

        return {'vx_values': values_x, 'vx_prob': probabilities_x,
                'vy_values': values_y, 'vy_prob': probabilities_y,
                'vz_values': values_z, 'vz_prob': probabilities_z,
                'vx_values_thermal': values_x_thermal, 'vx_prob_thermal': probabilities_x_thermal,
                'vy_values_thermal': values_y_thermal, 'vy_prob_thermal': probabilities_y_thermal,
                'vz_values_thermal': values_z_thermal, 'vz_prob_thermal': probabilities_z_thermal}


    def viscosity_gk(self):
        """
        Computes the dynamic viscosity from an equilibrium MD simulation
        based on Green-Kubo relation - Autocorrelation function of the shear stress

        Units: mPa.s
        """
        # TODO: Replace with fluid stress tensor
        sigxy_t = self.sigwall()['sigxy_t'] * 1e9  # mPa
        sigxz_t = self.sigwall()['sigxz_t'] * 1e9
        sigyz_t = self.sigwall()['sigyz_t'] * 1e9

        temp = np.mean(self.temp()['temp_t'])         # K

        prefac = self.vol* 1e-27 / (sci.k * temp )

        v_xy = prefac * np.sum(sq.acf(sigxy_t)['non-norm']) * 1e-15  # mPa.s
        v_xz = prefac * np.sum(sq.acf(sigxz_t)['non-norm']) * 1e-15  # mPa.s
        v_yz = prefac * np.sum(sq.acf(sigyz_t)['non-norm']) * 1e-15  # mPa.s

        viscosity = (v_xz+v_yz+v_xy)/3.

        return {'mu': viscosity}


    def transport(self):
        """
        Computes the dynamic viscosity and shear rate (and the uncertainties)
        from the corresponding velocity profile of a non-equilibrium MD simulation

        Units: mPa.s and s^-1 for the dynamic viscosity and shear rate, respectively
        """

        vels = self.velocity()['vx_Z']

        # sd
        if self.pumpsize==0:
            coeffs_fit = np.polyfit(self.height_array[vels!=0], vels[vels!=0], 1)
            sigxz_avg = np.mean(self.sigwall()['sigxz_t']) * 1e9      # mPa
            shear_rate = coeffs_fit[0] * 1e9
            mu = sigxz_avg / shear_rate

            coeffs_fit_lo = np.polyfit(self.height_array[vels!=0], sq.get_err(self.velocity()['vx_full_z'])['lo'], 1)
            shear_rate_lo = coeffs_fit_lo[0] * 1e9
            mu_lo = sigxz_avg / shear_rate_lo

            coeffs_fit_hi = np.polyfit(self.height_array[vels!=0], sq.get_err(self.velocity()['vx_full_z'])['hi'], 1)
            shear_rate_hi = coeffs_fit_hi[0] * 1e9
            mu_hi = sigxz_avg / shear_rate_hi

        # pd, sp
        if self.pumpsize!=0:
            # Get the viscosity
            sigxz_avg = np.mean(self.sigwall()['sigxz_t']) * 1e9      # mPa
            pgrad = self.virial()['pGrad']            # MPa/m

            coeffs_fit = np.polyfit(self.height_array[vels!=0], vels[vels!=0], 2)
            mu = pgrad/(2 * coeffs_fit[0])          # mPa.s
            shear_rate = sigxz_avg / mu

            coeffs_fit_lo = np.polyfit(self.height_array[vels!=0], sq.get_err(self.velocity()['vx_full_z'])['lo'], 2)
            mu_lo = pgrad/(2 * coeffs_fit_lo[0])          # mPa.s
            shear_rate_lo = sigxz_avg / mu_lo

            coeffs_fit_hi = np.polyfit(self.height_array[vels!=0], sq.get_err(self.velocity()['vx_full_z'])['hi'], 2)
            mu_hi = pgrad/(2 * coeffs_fit_hi[0])          # mPa.s
            shear_rate_hi = sigxz_avg / mu_hi

        return {'shear_rate': shear_rate, 'mu': mu,
                'shear_rate_lo': shear_rate_lo, 'mu_lo': mu_lo,
                'shear_rate_hi': shear_rate_hi, 'mu_hi': mu_hi}


    def heat_flux(self):
        """
        Computes the heat flux vector components according to Irving-Kirkwood expression

        Units: J/(m^2.s)
        """

        # From post-processing we get kcal/mol * A/fs ---> J * m/s
        conv_to_Jmpers = (4184*1e-10)/(sci.N_A*1e-15)

        # Heat flux vector
        je_x = np.array(self.data_x.variables["JeX"])[self.skip:] * conv_to_Jmpers * 1/np.mean(self.vol* 1e-27)  # J/m2.s
        je_y = np.array(self.data_x.variables["JeY"])[self.skip:] * conv_to_Jmpers * 1/np.mean(self.vol* 1e-27)  # J/m2.s
        je_z = np.array(self.data_x.variables["JeZ"])[self.skip:] * conv_to_Jmpers * 1/np.mean(self.vol* 1e-27)  # J/m2.s

        return {'je_x':je_x, 'je_y':je_y, 'je_z':je_z}


    def lambda_gk(self):
        """
        Computes the thermal conductivity (lambda) from equilibrium MD
        based on Green-Kubo relation - Autocorrelation function of the of the heat flux

        Units: J/mKs
        """

        T = np.mean(self.temp()['temp_t']) # K
        prefac =  np.mean(self.vol*1e-27)/(sci.k*T**2)        # m3/(J*K)

        # GK relation to get lambda from Equilibrium
        lambda_x = prefac * np.sum(sq.acf(self.heat_flux()['je_x'])['non-norm']* 1e-15)      # J/mKs
        lambda_y = prefac * np.sum(sq.acf(self.heat_flux()['je_y'])['non-norm']* 1e-15)      # J/mKs
        lambda_z = prefac * np.sum(sq.acf(self.heat_flux()['je_z'])['non-norm']* 1e-15)      # J/mKs
        lambda_tot = (lambda_x+lambda_y+lambda_z)/3.

        return {'lambda_tot':lambda_tot}


    def lambda_ecouple(self, thermo_out):
        """
        Computes the heat flow rate from the slope of the energy removed by the thermostat with time in NEMD
        LAMMPS keyword is Ecouple.
        and the corresponding z-component of thermal conductivity vector from Fourier's law

        Units: J/mKs and J/s for the thermal conductivty and heat flow rate, respectively
        """

        # Ecouple from the thermostat:
        delta_energy = np.loadtxt(thermo_out, skiprows=1, dtype=float)[:,2] # kcal/mol
        cut = self.skip + 10000 # Consider the energy slope in a window of 10 ns

        # Heat flow rate: slope of the energy with time
        qdot, _  = np.polyfit(self.time[self.skip:cut], delta_energy[self.skip:cut], 1)  # (kcal/mol) / fs
        qdot *= 4184 / (sci.N_A * 1e-15)     # J/s
        qdot_continuum = self.transport()['mu'] * 1e-3 * self.transport()['shear_rate']**2 * 2 * self.wall_A * 1e-18 * self.avg_gap_height * 1e-9

        # Heat flux
        j =  qdot / (2 * self.wall_A) # J/(m2.s)

        temp_grad = self.temp()['temp_grad']     # K/m
        print(f'Tgrad = {temp_grad:e} K/m')
        lambda_z = -j / temp_grad

        return {'lambda_z':lambda_z, 'qdot':qdot, 'qdot_continuum':qdot_continuum}


    def lambda_IK(self):
        """
        Computes the heat flow rate from the Irving-Kirkwood expression in NEMD
        and the corresponding z-component of thermal conductivity vector from Fourier's law

        Units: J/mKs and J/s for the thermal conductivty and heat flow rate, respectively
        """

        je_z = -self.heat_flux()['je_z']   # J/m2.s
        temp_grad = self.temp()['temp_grad']   # K/m

        qdot = np.mean(je_z) * self.wall_A * 1e-18  # W
        qdot_continuum = self.transport()['mu'] * 1e-3 * self.transport()['shear_rate']**2 * self.wall_A * 1e-18 * self.avg_gap_height * 1e-9 / 2.

        lambda_z = -je_z / temp_grad
        lambda_continuum = - self.transport()['mu'] * 1e-3 * self.transport()['shear_rate']**2 * self.avg_gap_height * 1e-9 / (2 * temp_grad)   # W/(m.K)

        return {'lambda_z':lambda_z, 'lambda_continuum':lambda_continuum, 'qdot':qdot, 'qdot_continuum':qdot_continuum}


    def slip_length(self):
        """
        Computes the slip length and velocity (Slip velocity * Shear rate) from
        linear extrapolation of the velocity profile to zero velocity and returns
        the extrapolated data for plotting

        Units: nm and m/s for length and velocity, respectively
        """

        vels = self.velocity()['vx_Z']

        if self.pumpsize==0:
            fit_data = funcs.fit(self.height_array[vels!=0], vels[vels!=0], 1)['fit_data']
        else:
            fit_data = funcs.fit(self.height_array[vels!=0], vels[vels!=0], 2)['fit_data']

        npoints = len(self.height_array[vels!=0])
        # Positions to inter/extrapolate
        xdata_left = np.linspace(-1, self.height_array[vels!=0][0], npoints)
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
        Vs = Ls * sci.nano * self.transport()['shear_rate']        # m/s

        return {'root_left':root_left, 'Ls':Ls, 'Vs':Vs,
                'xdata_left':xdata_left,
                'extrapolate_left':extrapolate(xdata_left),
                'root_right':root_right,
                'xdata_right':xdata_right,
                'extrapolate_right':extrapolate(xdata_right)}


    def coexistence_densities(self):
        """
        Computes the Liquid and vapor densities densities in equilibrium Liquid/Vapor interface simulations

        Units: g/cm^3
        """

        density = self.density()['den_Z']

        rho_liquid = np.max(density)
        rho_vapour = np.min(self.density()['den_Z'][10:-10])

        return {'rho_l':rho_liquid, 'rho_v':rho_vapour}


    def surface_tension(self):
        """
        Computes the surface tension (γ) from an equilibrium Liquid/Vapor interface
        The pressure tensor is computed from the "virial" method using IK expression

        Units: mN/m
        """

        virxx, viryy, virzz = -self.virial()['Wxx_t'], -self.virial()['Wyy_t'], -self.virial()['Wzz_t']   # MPa
        if self.avg_gap_height !=0: # Walls
            gamma = self.avg_gap_height * (virzz - (0.5*(virxx+viryy)) ) * 1e6 * 1e-9
        else:
            gamma = 0.5 * self.Lz * (virzz - (0.5*(virxx+viryy)) ) * 1e6 * 1e-9

        return {'gamma':gamma}


    def reynolds_num(self):
        """
        Computes the Reynolds number of the flow
        """

        rho = np.mean(self.density()['den_t']) * 1e3 # kg/m3
        u = np.mean(self.velocity()['vx_t']) # m/s
        h = self.avg_gap_height * 1e-9 # m
        mu = self.transport()['mu'] * 1e-3 # Pa.s

        Re = rho * u * h / mu

        return Re


    def sf(self):
        """
        Computes the longitudnal (skx) and transverse (sky) structure factors for the liquid and solid
        """

        # wavevectors
        kx = np.array(self.data_x.variables["kx"])
        ky = np.array(self.data_x.variables["ky"])

        sf_real = np.array(self.data_x.variables["sf"])[self.skip:]
        sf_x_real = np.array(self.data_x.variables["sf_x"])[self.skip:]
        sf_y_real = np.array(self.data_x.variables["sf_y"])[self.skip:]

        # Structure factor averaged over time for each k
        sf = np.mean(sf_real, axis=0)
        sf_time = np.mean(sf_real, axis=(1,2))
        sf_x = np.mean(sf_x_real, axis=0)
        sf_y = np.mean(sf_y_real, axis=0)

        # For the solid
        # sf_real_solid = np.array(self.data_x.variables["sf_solid"])[self.skip:]
        # sf_x_real_solid = np.array(self.data_x.variables["sf_x_solid"])[self.skip:]
        # sf_y_real_solid = np.array(self.data_x.variables["sf_y_solid"])[self.skip:]

        # sf_solid = np.mean(sf_real_solid, axis=0)
        # sf_x_solid = np.mean(sf_x_real_solid, axis=0)
        # sf_y_solid = np.mean(sf_y_real_solid, axis=0)

        # How to get S(k) in radial direction (In 2D from k=(kx,ky) and the sf=(sfx,sfy))
        # k_vals=[]
        # sf_r=[]
        # for i in range(len(kx)):
        #     for k in range(len(ky)):
        #         k_vals.append(np.sqrt(kx[i]**2+ky[k]**2))
        #         sf_r.append(np.sqrt(sf_x[i]**2+sf_y[k]**2))

        return {'kx':kx, 'ky':ky, 'sf':sf, 'sf_x':sf_x, 'sf_y':sf_y, 'sf_time':sf_time}
                # 'sf_solid':sf_solid, 'sf_x_solid':sf_x_solid, 'sf_y_solid':sf_y_solid}

    def ISF(self):
        """
        Computes the intermediate scattering function
        TODO: Check that it works properly!
        """

        # Fourier components of the density
        rho_k = np.array(self.data_x.variables["rho_k"])
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


    def transverse_acf(self):
        """
        Computes the transverse Autocorrelation function
        TODO: Check that it works properly!
        """

        density = np.array(self.data_z.variables["Density"])[self.skip:1000] * (self.mf/sci.N_A) / (ang_to_cm**3)

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

        jx = np.array(self.data_x.variables["Jx"])[self.skip:10000]
        # Fourier transform in the space dimensions
        jx_tq = np.fft.fftn(jx, axes=(1,2))

        acf = sq.acf(jx_tq)['norm']

        return {'a':acf[:, 0, 0].real}