#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from operator import itemgetter
from plot_settings import Initialize, Modify
from scipy.integrate import odeint
import scipy.constants as sci
import compute_real as ct
import yaml
import funcs
import netCDF4
import extract_thermo
from diff_tvr import DiffTVR
import sample_quality as sq

# Logger Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import golbal plot configuration
# plt.style.use('imtek')
plt.style.use('thesis')
# Specify the path to the font file
font_path = '/usr/share/fonts/truetype/LinLibertine_Rah.ttf'
# Register the font with Matplotlib
fm.fontManager.addfont(font_path)
# Set the font family for all text elements
plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'
plt.rcParams['mathtext.it'] = 'Linux Libertine:italic'
plt.rcParams['mathtext.bf'] = 'Linux Libertine:bold'

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

        ax.plot(x, y, markersize=8)
        # if input('Plot time-average?') == 'y':
        #     ax.axhline(y=np.mean(y))


    def plot_uncertainty(self, ax, x, y, xerr, yerr):
        """
        Plots the uncertainty of the data
        """
        if self.config['err_caps'] is not None:
            markers, caps, bars= ax.errorbar(x, y, xerr=xerr, yerr=yerr, color='k',
                                        capsize=3.5, markersize=8, lw=2, alpha=0.8)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        if self.config['err_fill'] is not None:
            lo, hi = sq.get_err(yerr)['lo'], sq.get_err(yerr)['hi']
            ax.fill_between(x, lo, hi, color=color, alpha=0.4)


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
            thermo_variables = extract_thermo.extract_data(i, self.skip)[1]

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
                self.axes_array[0].set_ylabel('Force (nN)')
                npump = data[:,thermo_variables.index('v_nAtomsPump')]
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fpump')]*kcalpermolA_to_N*npump*1e9       #kcalpermolA_to_N*1e12 # ns, pN
                self.plot_data(self.axes_array[n], xdata, ydata)
                self.axes_array[0].axhline(y=np.mean(ydata))
                if nrows>1: n+=1
            if 'fw' in variables:
                self.axes_array[0].set_xlabel('Time (ns)')
                self.axes_array[0].set_ylabel('Force (nN)')
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_fw')]*1e9 # ns, nN
                self.plot_data(self.axes_array[n], xdata, ydata)
                self.axes_array[0].axhline(y=np.mean(ydata))
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
                xdata, ydata = data[:,0]*1e-6, data[:,thermo_variables.index('v_net_mom_flux')]*kcalpermolA_to_N*npump*1e9 #kcalpermolA_to_N*1e12 # ns, pN
                self.plot_data(self.axes_array[n], xdata, ydata)

            os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def nvt_eos(self):

        self.axes_array[0].set_xlabel('Temperature (K)')
        self.axes_array[0].set_ylabel('Pressure (MPa)')

        gromos_aa=[]
        gromos_ua=[]
        trappe_ua=[]

        for i in self.txts:
            if 'GROMOS-AA' in i: gromos_aa.append(i)
            if 'GROMOS-UA' in i: gromos_ua.append(i)
            if 'TraPPE-UA' in i: trappe_ua.append(i)

        pressure1, temperature1 = [], []
        for i in gromos_aa:
            n=0    # subplot
            data = extract_thermo.extract_data(i, self.skip)[0]
            thermo_variables = extract_thermo.extract_data(i, self.skip)[1]

            press = data[:,thermo_variables.index('Press')]
            temp = data[:,thermo_variables.index('Temp')]

            pressure1.append(np.mean(press))
            temperature1.append(np.mean(temp))

        pressure2, temperature2 = [], []
        for i in gromos_ua:
            n=0    # subplot
            data = extract_thermo.extract_data(i, self.skip)[0]
            thermo_variables = extract_thermo.extract_data(i, self.skip)[1]

            press = data[:,thermo_variables.index('Press')]
            temp = data[:,thermo_variables.index('Temp')]

            pressure2.append(np.mean(press))
            temperature2.append(np.mean(temp))

        pressure3, temperature3 = [], []
        for i in trappe_ua:
            n=0    # subplot
            data = extract_thermo.extract_data(i, self.skip)[0]
            thermo_variables = extract_thermo.extract_data(i, self.skip)[1]

            press = data[:,thermo_variables.index('Press')]
            temp = data[:,thermo_variables.index('Temp')]

            pressure3.append(np.mean(press))
            temperature3.append(np.mean(temp))

        temperature_exp = [313, 343, 373, 443, 483, 523, 543, 573]
        pressure_exp = [60.50, 86.80, 109.80, 160.10, 187.40, 213.00, 225.30, 243.90]

        if gromos_aa: self.plot_data(self.axes_array[n], temperature1, np.array(pressure1)*0.101325)
        if gromos_ua: self.plot_data(self.axes_array[n], temperature2, np.array(pressure2)*0.101325)
        if trappe_ua: self.plot_data(self.axes_array[n], temperature3, np.array(pressure3)*0.101325)
        self.plot_data(self.axes_array[n], temperature_exp, pressure_exp)

        try:
            Modify(temperature3, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def press_md_cont(self):
        """
        Extracts data from 'pressure-profile-md.txt' and plots it.
        Used to compare pressure profiles from MD and continuum mainly for conv-div geometry.
        Uses the Modify class from 'plot_settings' module.
        """
        self.axes_array[0].set_xlabel('Length (nm)')
        self.axes_array[0].set_ylabel('Pressure (MPa)')

        for idx, val in enumerate(self.txts):
            n=0    # subplot
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)

            xdata = data[:,0]           # nm
            if data[:,0][0]>1: #TODO : make md file start from zero
                xdata = (data[:,0] - data[:,0][0]) #* x_pre

            ydata = data[:,1]           # MPa

            self.plot_data(self.axes_array[n], xdata, ydata)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def virial(self):
        """
        Extracts data from 'virial-chunkZ.txt' and plots it.
        Uses the Modify class from 'plot_settings' module.
        """
        self.axes_array[0].set_xlabel('Length (nm)')
        self.axes_array[0].set_ylabel('Pressure (MPa)')

        # for idx, val in enumerate(self.txts):
        n=0    # subplot
        data = np.loadtxt(self.txts[0], skiprows=self.skip, dtype=float)

        xdata = data[:,0]           # nm
        ydata = data[:,1]           # MPa

        self.plot_data(self.axes_array[n], xdata, ydata)

        # for idx, val in enumerate(self.txts):
        # data = np.loadtxt(self.txts[1], skiprows=self.skip, dtype=float)
        #
        # xdata = np.roll(data[:,0],64)           # nm
        # ydata = data[:,1]           # MPa
        #
        # self.plot_data(self.axes_array[n], xdata, ydata)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def radii(self):
        """
        Extracts data from 'radius_cavity_full.txt' and plots it.
        """
        self.axes_array[0].set_xlabel('Time (ns)')
        self.axes_array[0].set_ylabel(r'Radius $R \, \mathrm{(\AA)}$')

        R0 = 0.1e-10
        V0 = [5.0e3,8.5e3,15e3]
        pl = [73.564e3,74.32e3,76.88e3]

        for idx, val in enumerate(self.txts):
            n=0    # subplot
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)

            xdata = data[:,0]           # nm
            if data[:,0][0]>1: #TODO : make md file start from zero
                xdata = (data[:,0] - data[:,0][0]) #* x_pre

            ydata = data[:,1]

            self.plot_data(self.axes_array[n], xdata * 1e-6, ydata)

            time_s = xdata * 1e-15

            # From RP
            # material-specific (pentane)
            pv = 73e3                       # Calculated from Pv from Liquid-vapor equilibrium interface simulations in Pa
            temp = 300                     # K
            rho = 600                      # From the density-length profile of the NEMD simulation in kg/m3
            gamma = 0.0125                 # Planar surface tension from equilibrium MD at 300 K in N/m
            m_molecular = 72.15 * 1e-3     # Moleuclar mass of n-pentane   (kg/mol)

            BC = [R0, V0[idx]]                 # Bubble radius in Angstrom and interface velocity in m/s
            sol = odeint(funcs.RP, BC, time_s, args=(rho, pv, pl[idx], gamma))

            # radius_cont = np.roll(sol[:,0], start) * 1e10  # Angstrom
            radius_cont = sol[:,0] * 1e10  # Angstrom

            # Fill with zeros after bubble collapse
            for idx,val in enumerate(radius_cont):
                if val>=100 or val<0:
                    radius_cont[idx]=0

            self.plot_data(self.axes_array[0], xdata * 1e-6, radius_cont)


        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def temp(self):
        """
        Extracts data from 'temp-height.txt' and plots it.
        Uses the Modify class from 'plot_settings' module.
        """
        self.axes_array[0].set_xlabel('Gap height $h/h_0$')
        # self.axes_array[0].set_xlabel('Height (nm)')
        self.axes_array[0].set_ylabel('Temperature (K)')

        for idx, val in enumerate(self.txts):
            n=0    # subplot
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)

            xdata = data[:,0]           # dimensionless length
            ydata = np.ma.masked_where(data[:,1] == 0, data[:,1])           # K
            yfit = data[:,2]

            self.plot_data(self.axes_array[n], xdata, ydata)
            self.plot_data(self.axes_array[n], xdata, yfit)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def vx(self):
        """
        Extracts data from 'vx-height.txt' and plots it.
        Uses the Modify class from 'plot_settings' module.
        """
        self.axes_array[0].set_xlabel('Gap height $h/h_0$')
        self.axes_array[0].set_ylabel('Velocity $u$ (m/s)')

        for idx, val in enumerate(self.txts):
            n=0    # subplot
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)

            xdata = data[:,0]           # dimensionless length
            ydata = np.ma.masked_where(data[:,1] == 0, data[:,1])           # K
            yfit = data[:,2]

            self.plot_data(self.axes_array[n], xdata, ydata)
            self.plot_data(self.axes_array[n], xdata, yfit)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig




    def radius(self, R0, V0, pl, datasets, start, stop, dt, method):
        """
        Extracts data from 'radius.txt' plots it.
        Used to compare bubble radius from MD and compare with the Rayleigh-Plesset equation.
        Uses the Modify class from 'plot_settings' module.
        """

        self.axes_array[0].set_xlabel('Time (ns)')
        self.axes_array[0].set_ylabel(r'Radius $R \, \mathrm{(\AA)}$')

        for idx, val in enumerate(self.txts):
            data = np.loadtxt(self.txts[idx], skiprows=0, dtype=float)

            time_fs = data[:, 0]             # fs
            time_s = time_fs * 1e-15         # s
            time = time_fs * 1e-6            # ns

            if method=='static':        # Young-Laplace
                # Get the radius from MD ------------------------------------------
                time = time[:len(time)-1]
                radius_MD = data[:,1][:len(time)] # Angstrom
                self.plot_data(self.axes_array[0], time[start:stop], radius_MD[start:stop])

                # Get the radius from Young-Laplace relation ----------------------
                pv = 73e3                       # Calculated from Pv from Liquid-vapor equilibrium interface simulations in Pa
                gamma = 0.0125                 # Planar surface tension from equilibrium MD at 300 K in N/m

                # Get the liquid pressure
                get = ct.ExtractFromTraj(self.skip, datasets[idx], 1, 1)
                # Average pressure away from bubble (varies according to simulation)
                pl = get.virial()['data'][:,10:21,17:28,:] * 1e6        # Sphere
                # pl = get.virial()['data'][:,35:42,:,:] * 1e6        # Cylinder
                pl_avg = np.mean(pl, axis=(0,1,2,3))

                radius_cont = 2*gamma/(pv-pl_avg) * 1e10
                self.ax.axhline(y=radius_cont)

                # # Get the radius after curvature correction ----------------------
                a = pv - pl_avg
                b = -2 * gamma
                c = 4 * gamma * 5.02e-10    # Sphere
                # c = 2 * gamma * 5.02e-10    # Cylinder
                radius_cont_corrected = funcs.solve_quadratic(a,b,c)[1] * 1e10
                self.ax.axhline(y=radius_cont_corrected)

            elif method=='dynamic':       # Rayleigh-Plesset
                m3_to_cm3 = 1e6

                # From MD simualtions
                radius_MD = data[:,1]        # Angstrom

                # From RP
                # material-specific (pentane)
                pv = 73e3                       # Calculated from Pv from Liquid-vapor equilibrium interface simulations in Pa
                temp = 300                     # K
                rho = 600                      # From the density-length profile of the NEMD simulation in kg/m3
                gamma = 0.0125                 # Planar surface tension from equilibrium MD at 300 K in N/m
                m_molecular = 72.15 * 1e-3     # Moleuclar mass of n-pentane   (kg/mol)
                # eta = 0.5e-3                   # From equilib MD at 300 K in Pa.s

                n = rho * sci.N_A / (m_molecular * m3_to_cm3) # number density in cm^-3
                n_m3 = rho * sci.N_A / m_molecular            # number density in m^-3
                m = m_molecular / sci.N_A      # Mass of one molecule (kg)

                BC = [R0, V0]                 # Bubble radius in Angstrom and interface velocity in m/s
                sol = odeint(funcs.RP, BC, time_s, args=(rho, pv, pl))

                # radius_cont = np.roll(sol[:,0], start) * 1e10  # Angstrom
                radius_cont = sol[:,0] * 1e10  # Angstrom

                # Fill with zeros after bubble collapse
                for idx,val in enumerate(radius_cont[start:stop]):
                    if val>=100 or val<0:
                        radius_cont[start:stop][idx]=0

                self.plot_data(self.axes_array[0], time[start:stop], radius_cont[start:stop])
                self.plot_data(self.axes_array[0], time[start:stop], radius_MD[start:stop])

                # Bubble velocity
                if len(self.axes_array)>1:
                    self.axes_array[0].set_xlabel(None)
                    self.axes_array[1].set_xlabel('Time (ns)')
                    self.axes_array[1].set_ylabel('Velocity $\dot{R}$ (m/s)')

                    velocity_RP = np.gradient(radius_cont, time_fs)
                    self.plot_data(self.axes_array[1], time[start:stop][1:-1], velocity_RP[start:stop][1:-1]*1e5)

                    # velocity_MD = np.gradient(radius_MD, time_fs)
                    # self.plot_data(self.axes_array[1], time[start:stop][1:-1], velocity_MD[start:stop][1:-1]*1e5)

                # # Nucleation Rate (CNT)
                prefac = n * np.sqrt(2 * gamma / (np.pi * m))
                print(  np.exp( - 16 * np.pi * gamma**3 / (3 * sci.k * temp * (pv - pl)**2)))
                j_cnt = prefac * np.exp( - 16 * np.pi * gamma**3 / (3 * sci.k * temp * (pv - pl)**2) )   # cm^3.s^-1
                print(f'Nucleation rate CNT: {j_cnt:.2e} cm^-3.s^-1')

                # Nucleation Rate (MD)
                vol = 1.39269 * 1e6 * 1e-24    # From ovito surface mesh in cm3   #1.168 thin  #3.50801 thick
                tau_w = 0.1 * 1e-9             # s
                j_md = 1/(tau_w * vol)
                print(f'Nucleation rate in MD: {j_md:.2e} cm^-3.s^-1')

                # Hydrodynamic collapse time
                # tau = 0.91 * radius_cont[start:stop][0] * 1e-10 * np.sqrt(rho / (pv-pl)) * 1e9
                # print(f'Hydrodynamic Collapse time: {tau:.4f} ns')

            else:
                self.plot_data(self.axes_array[0], time, data[:,1])

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
        self.axes_array[0].set_xlabel('Time $t$ (ns)')
        self.axes_array[0].set_ylabel('Viscosity $\eta$ (mPa.s)')

        data_0 = np.loadtxt(self.txts[0], skiprows=self.skip, dtype=float)[:,0]
        ydata = np.zeros((len(self.txts),len(data_0)))

        for idx, val in enumerate(self.txts):
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)
            xdata = data[:,0] * 1e-6
            ydata[idx,:] = data[:,1] * 1 #* 1.5

            self.plot_data(self.axes_array[0], xdata, ydata[idx,:])

        ydata_mean = np.mean(ydata, axis=0)
        self.plot_data(self.axes_array[0], xdata, ydata_mean)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig


    def transport_press(self):
        """
        Extracts data from 'p-eta.txt' or 'p-lambda.txt' and plots it.
        Used to plot viscosity or thermal conductivity obtained
        from Green-Kubo relations with pressure.
        Uses the Modify class from 'plot_settings' module.
        """
        self.axes_array[0].set_xlabel('Pressure (MPa)')
        if 'pvisco' in self.txts[0]: self.axes_array[0].set_ylabel('Viscosity $\eta/\eta_0$')
        if 'pconduct' in self.txts[0]: self.axes_array[0].set_ylabel('Conductivity $\lambda$ (W/mK)')

        for idx, val in enumerate(self.txts):
            data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)
            xdata = data[:,0]
            ydata = data[:,1]
            ydata_exp = data[:,4]

            # self.plot_data(self.axes_array[0], xdata, ydata)
            self.plot_uncertainty(self.axes_array[0], xdata, ydata, xerr=None, yerr=[data[:,1]-data[:,2],data[:,3]-data[:,1]])
            self.plot_data(self.axes_array[0], xdata, ydata_exp)

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
