#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, logging
import matplotlib.pyplot as plt
from operator import itemgetter
from plot_settings import Initialize, Modify
from scipy.integrate import odeint
import scipy.constants as sci
import compute_thermo as ct
import yaml
import funcs
import netCDF4
import extract_thermo
from diff_tvr import DiffTVR

# Logger Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import golbal plot configuration
plt.style.use('imtek')

kcalpermolA_to_N = 6.9477e-11

class PlotFromTxt:

    def __init__(self, skip, filename, txtfiles, configfile):

        self.skip = skip
        self.filename = filename
        self.txts = txtfiles
        self.configfile = configfile

        # Read the yaml file
        try:
            with open(configfile, 'r') as f:
                self.config = yaml.safe_load(f)
        except TypeError:
            logging.error('Insert template file!')

        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())


    def plot_data(self, ax, x, y):
        """
        Plots the data
        """

        ax.plot(x, y)
        # if input('Plot time-average?') == 'y':
        #     ax.axhline(y=np.mean(y))


    def extract_plot(self, *arr_to_plot):
        """
        Extracts data from log files does the plotting.
        Uses the Modify class from 'plot_settings' module.
        """

        variables = arr_to_plot[0]
        nrows = self.config['nrows']

        # if self.filename.startswith('log'): # Plot from LAMMPS log files
        for i in self.txts:
            n=0    # subplot
            data = extract_thermo.extract_data(i, self.skip)[0]

            if 'energy' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('E (Kcal/mol.A)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('TotEng')] # ns, Kcal/mol
                self.plot_data(self.axes_array[n], xdata, ydata)
                if nrows>1: n+=1
            if 'npump' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('No. of Atoms')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_nAtomsPump')] # ns, count
                self.plot_data(self.axes_array[n], xdata, ydata)
                if nrows>1: n+=1
            if 'fpump' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (pN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fpump')]*kcalpermolA_to_N*1e12 # ns, pN
                self.plot_data(self.axes_array[n], xdata, ydata)
                if nrows>1: n+=1
            if 'fw' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (nN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fw')]*1e9 # ns, nN
                self.plot_data(self.axes_array[n], xdata, ydata)
                if nrows>1: n+=1
            if 'fp' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (nN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fp')]*1e9 # ns, nN
                self.plot_data(self.axes_array[n], xdata, ydata)
                if nrows>1: n+=1
            if 'fin' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (nN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fin')]*1e9 # ns, nN
                self.plot_data(self.axes_array[n], xdata, ydata)
                if nrows>1: n+=1
            if 'fout' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (nN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fout')]*1e9 # ns, nN
                self.plot_data(self.axes_array[n], xdata, ydata)
            if 'fmomnet' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (pN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fmomnet')]*kcalpermolA_to_N*1e12 # ns, pN
                self.plot_data(self.axes_array[n], xdata, ydata)

            os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def press_md_cont(self):
        """
        Extracts data from 'pressure-profile-md.txt' and plots it.
        Used to compare pressure profiles from MD and continuum mainly for conv-div geometry.
        Uses the Modify class from 'plot_settings' module.
        """

        for idx, val in enumerate(self.txts):
            n=0    # subplot
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)

            xdata = data[:,0]           # nm
            if data[:,x_col][0]>10: #TODO : make md file start from zero
                xdata = (data[:,0] - data[:,0][0]) * x_pre

            ydata = data[:,1]           # MPa

            self.plot_data(self.axes_array[n], xdata, ydata)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def radius(self, R0, V0, pl, datasets_x, datasets_z, start, stop, dt, method):
        """
        Extracts data from 'radius.txt' plots it.
        Used to compare bubble radius from MD and compare with the Rayleigh-Plesset equation.
        Uses the Modify class from 'plot_settings' module.
        """
        # surface curvature correction
        tolman_length = 5e-10          # meter

        # simualtion-specific
        vol = 1.39269 * 1e6 * 1e-24    # From ovito surface mesh in cm3   #1.168 thin  #3.50801 thick
        rho = 600                      # From the density-length profile of the NEMD simulation in kg/m3
        temp = 300                     # K
        tau_w = 0.1 * 1e-9             # s

        # material-specific (pentane)
        pv = 1e5                       # Calculated from Pv from Liquid-vapor equilibrium interface simulations in Pa
        eta = 0.5e-3                   # From equilib MD at 300 K in Pa.s
        gamma = 0.0125                 # From equilibrium MD at 300 K in N/m
        m_molecular = 72.15 * 1e-3     # Moleuclar mass of n-pentane   (kg/mol)
        m = m_molecular / sci.N_A      # Mass of one molecule (kg)

        BC = [R0, V0]                 # Bubble radius in Angstrom and interface velocity in m/s
        m3_to_cm3 = 1e6
        n = rho * sci.N_A / (m_molecular * m3_to_cm3) # number density in cm^-3
        n_m3 = rho * sci.N_A / m_molecular            # number density in m^-3

        self.axes_array[0].set_xlabel('Time (ns)')
        self.axes_array[0].set_ylabel('R(t) ($\AA$)')

        for idx, val in enumerate(self.txts):
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)
            # get = ct.ExtractFromTraj(self.skip, datasets_x[idx], datasets_z[idx], 1, 1)
            # pl = get.virial()['vir_t'] * 1e6    # Virial pressure, also the applied external Pressure in Pa
            # pl_avg = np.mean(pl)   # Time-averaged virial pressure in Pa

            time_s = data[:, 0] * 1e-15         # s
            time = data[:, 0] * 1e-6            # ns
            time_fs = data[:, 0]                # fs

            if method=='static':        # Young-Laplace
                nsteps = 3 # reduce noise by further sampling every nsteps
                radius_sampled_every = np.int((data[:, 0][1] - data[:, 0][0]) / (1/nsteps*dt))
                pl_sampled = np.mean(pl.reshape(-1, radius_sampled_every), axis=1)

                time_cont = time[::nsteps]
                time_cont = time_cont[:len(time_cont)-1]
                radius_cont = 2*gamma/(pv-pl_sampled) * 1e10

                time = time[:len(time)-1]
                radius_MD = data[:,1][:len(time)] # Angstrom

                # Get the radius after curvature correction
                a = pv - pl_sampled
                b = -2 * gamma
                c = 4 * gamma * tolman_length
                radius_cont_corrected = funcs.solve_quadratic(a,b,c)[1] * 1e10

                # Fill with zeros if bubble radius is negative
                for idx,val in enumerate(radius_cont):
                    if val<0:
                        radius_cont[idx]=0
                for idx,val in enumerate(radius_cont_corrected):
                    if val<0:
                        radius_cont_corrected[idx]=0

                self.plot_data(self.axes_array[0], time_cont[start:stop], radius_cont[start:stop])
                self.plot_data(self.axes_array[0], time_cont[start:stop], radius_cont_corrected[start:stop])
                self.plot_data(self.axes_array[0], time[start:stop], radius_MD[start:stop])

            if method=='dynamic':       # Rayleigh-Plesset
                radius_MD = data[:,1]        # Angstrom
                sol = odeint(funcs.RP, BC, time_s, args=(rho, pv, pl))
                # # sol = odeint(funcs.RP_full, BC, time_s, args=(rho, pv, pl, gamma, eta))

                radius_cont = np.roll(sol[:,0],start) * 1e10  # Angstrom

                # Fill with zeros after bubble collapse
                for idx,val in enumerate(radius_cont[start:stop]):
                    if val>=100 or val<0:
                        radius_cont[start:stop][idx]=0

                self.plot_data(self.axes_array[0], time[start:stop], radius_cont[start:stop])
                self.plot_data(self.axes_array[0], time[start:stop], radius_MD[start:stop])

                # # Nucleation Rate (CNT)
                prefac = n * np.sqrt(2 * gamma / (np.pi * m))
                j_cnt = prefac * np.exp( -16 * np.pi * gamma**3 / (3 * sci.k * temp * (25e6 - pv)**2) )   # cm^3.s^-1
                print(f'Nucleation rate CNT: {j_cnt:.2e} cm^-3.s^-1')

                # Nucleation Rate (MD)
                j_md = 1/(tau_w * vol)
                print(f'Nucleation rate in MD: {j_md:.2e} cm^-3.s^-1')

                # Hydrodynamic collapse time
                tau = 0.91 * radius_cont[start:stop][0] * 1e-10 * np.sqrt(m * n_m3 / pl) * 1e9
                print(f'Hydrodynamic Collapse time: {tau:.4f} ns')

                if len(self.axes_array)>1:
                    self.axes_array[0].set_xlabel(None)
                    self.axes_array[1].set_xlabel('Time (ns)')
                    self.axes_array[1].set_ylabel('$\dot{R}$(t) (m/s)')

                    velocity_RP = np.gradient(radius_cont, time_fs)
                    # velocity_MD = np.gradient(radius_MD, time_fs)

                    self.plot_data(self.axes_array[1], time[start:stop][1:-1], velocity_RP[start:stop][1:-1]*1e5)
                    # self.plot_data(self.axes_array[1], time[start:stop][1:-1], velocity_MD[start:stop][1:-1]*1e5)

        try:
            Modify(time, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def GK(self):
        """
        Extracts data from 'eta.txt' or 'lambda.txt' and plots it.
        Used to check for convergence of viscosity and thermal conductivity obtained
        from Green-Kubo relations.
        Uses the Modify class from 'plot_settings' module.
        """
        self.axes_array[0].set_xlabel('Time (ns)')
        self.axes_array[0].set_ylabel('$\eta$ (mPa.s)')

        x_pre, y_pre, x_col, y_col = 1e-6, 1, 0, 1

        xdata = data[:,x_col] * x_pre
        ydata = data[:,y_col] * y_pre

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
