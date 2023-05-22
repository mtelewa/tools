#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys, os, logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from operator import itemgetter
import scipy.constants as sci
from scipy.optimize import curve_fit
import yaml
import funcs
import sample_quality as sq
from compute_real import ExtractFromTraj as dataset
from plot_settings import Initialize, Modify
from matplotlib.ticker import ScalarFormatter
import numpy.ma as ma
import scipy.stats as stats

# Uncomment to change Matplotlib backend
# mpl.use('TkAgg')

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

mpl.rcParams.update({'lines.markersize': 8})

#           0             1                   2                    3
labels=('Height (nm)','Length (nm)', 'Time (ns)', r'Density $\rho$ (g/${\mathrm{cm^3}}$)',
#           4                                           5             6                    7
        r'${\mathrm{j_x}}$ (g/${\mathrm{m^2}}$.ns)', 'Velocity $u$ (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\mathrm{(Force)}}$ (pN)', r'${\mathrm{dP / dx}}$ (MPa/nm)',
#           10                                          11
        r'${\mathrm{\dot{m}}} \times 10^{-18}}$ (g/ns)', r'Shear rate ${\mathrm{\dot{\gamma}} (s^{-1})}$',
#           12
        r'Viscosity ${\mathrm{\eta}}$ (mPa.s)', r'$N_{\mathrm{\pump}}$', r'Energy (Kcal/mol)', '$R(t)$ (${\AA}$)')

colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'seagreen','darkslategrey']

class PlotGeneral:

    def __init__(self, skip, datasets_x, datasets_z, mf, configfile, pumpsize):

        self.skip = skip
        self.mf = mf
        self.datasets_x = datasets_x
        self.datasets_z = datasets_z
        self.configfile = configfile
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)
        self.pumpsize = pumpsize
        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())

        first_dataset = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        self.Nx = len(first_dataset.length_array)
        self.Nz = len(first_dataset.height_array)
        self.time = first_dataset.time
        self.Ly = first_dataset.Ly


    def plot_uncertainty(self, ax, x, y, xerr, yerr):
        """
        Plots the uncertainty of the data
        """
        if self.config['err_caps'] is not None:
            if len(np.asarray(yerr).shape)>1:   # calculate uncertainty array
                yerr = sq.get_err(arr)['uncertainty']
            else:   # uncertainty array is already given
                xerr, yerr = xerr, yerr

            markers, caps, bars= ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                                        capsize=3.5, markersize=8, lw=2, alpha=0.8)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        if self.config['err_fill'] is not None:
            lo, hi = sq.get_err(yerr)['lo'], sq.get_err(yerr)['hi']
            ax.fill_between(x, lo, hi, color=color, alpha=0.4)


    def v_distrib(self):
        """
        Plots the thermal velocity distributions for the three velocity components to check
        for local thermodynamic equilibrium (LTE) condition (zero skewness and kurtosis)
        """
        ax = self.axes_array[0]
        ax.set_xlabel('Velocity (m/s)')
        # ax.set_ylabel('Probability $f(v)$')

        first_dataset = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        T =  np.mean(first_dataset.temp()['temp_t'])
        # vx = np.array(first_dataset.vel_distrib()['vx_values_thermal'])
        # vy = np.array(first_dataset.vel_distrib()['vy_values_thermal'])
        # vz = np.array(first_dataset.vel_distrib()['vz_values_thermal'])
        v = np.array(first_dataset.vel_distrib()['fluid_v'])
        v = ma.masked_where(v == 0, v)

        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            vx_values = data.vel_distrib()['vx_values']
            vx_prob = data.vel_distrib()['vx_prob']

            vy_values = data.vel_distrib()['vy_values']
            vy_prob = data.vel_distrib()['vy_prob']

            vz_values = data.vel_distrib()['vz_values']
            vz_prob = data.vel_distrib()['vz_prob']

            ax.plot(vx_values, vx_prob)
            ax.plot(vy_values, vy_prob)
            ax.plot(vz_values, vz_prob)
            # Maxwell-Boltzmann distribution
            maxwell = stats.maxwell
            para = maxwell.fit(v)
            params = maxwell.fit(v,floc=0)
            mean, var, skew, kurt = maxwell.stats(moments='mvsk')
            x = np.linspace(0, 1.2, 100)
            ax.plot(x+para[0], maxwell.pdf(x, *params), lw=3)

        Modify(vx_values, self.fig, self.axes_array, self.configfile)


    def colorFader(self, c1, c2, mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


    def v_evolution(self):
        """
        Plots the streaming velocity profile in 5 regions along the flow direction
        This is mainly to observe the compressibility of the fluid along the stream
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[5])

        # Color gradient
        c1 = 'tab:blue' #blue
        c2 = 'tab:red' #red
        n = 4
        colors = []
        for x in range(n+1):
            colors.append(self.colorFader(c1,c2,x/n))

        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            h = data.height_array
            # vx_R1 =  data.velocity()['vx_R1']
            vx_R2 =  data.velocity()['vx_R2']
            vx_R3 =  data.velocity()['vx_R3']
            vx_R4 =  data.velocity()['vx_R4']
            vx_R5 =  data.velocity()['vx_R5']

            # ax.plot(h[vx_R1!=0][1:-1], vx_R1[vx_R1!=0][1:-1], color=colors[0])
            ax.plot(h[vx_R2!=0][1:-1], vx_R2[vx_R2!=0][1:-1], color=colors[1])
            ax.plot(h[vx_R3!=0][1:-1], vx_R3[vx_R3!=0][1:-1], color=colors[2])
            ax.plot(h[vx_R4!=0][1:-1], vx_R4[vx_R4!=0][1:-1], color=colors[3])
            ax.plot(h[vx_R5!=0][1:-1], vx_R5[vx_R5!=0][1:-1], color=colors[4])
            # fit_R1 = funcs.fit(h[vx_R1!=0][1:-1], vx_R1[vx_R1!=0][1:-1], 2)
            # ax.plot(h[vx_R1!=0][1:-1], fit_R1['fit_data'], color=colors[0])
            # fit_R2 = funcs.fit(h[vx_R2!=0][1:-1], vx_R2[vx_R2!=0][1:-1], 2)
            # ax.plot(h[vx_R2!=0][1:-1], fit_R2['fit_data'], color=colors[1])
            # fit_R3 = funcs.fit(h[vx_R3!=0][1:-1], vx_R3[vx_R3!=0][1:-1], 2)
            # ax.plot(h[vx_R3!=0][1:-1], fit_R3['fit_data'], color=colors[2])
            # fit_R4 = funcs.fit(h[vx_R4!=0][1:-1], vx_R4[vx_R4!=0][1:-1], 2)
            # ax.plot(h[vx_R4!=0][1:-1], fit_R4['fit_data'], color=colors[3])
            # fit_R5 = funcs.fit(h[vx_R5!=0][1:-1], vx_R5[vx_R5!=0][1:-1], 2)
            # ax.plot(h[vx_R5!=0][1:-1], fit_R5['fit_data'], color=colors[4])

        Modify(h, self.fig, self.axes_array, self.configfile)


    def pdiff_pumpsize(self):
        """
        Plots the pressure difference ΔP from the fluid virial pressure and the wall σ_zz
        with the pump size
        """
        ax = self.axes_array[0]
        ax.set_xlabel('Normalized pump length')

        pump_size = []
        vir_pdiff, sigzz_pdiff, vir_err, sigzz_err = [], [], [], []
        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            pump_size.append(self.pumpsize)
            vir_pdiff.append(data.virial(pump_size[i])['pDiff'])
            vir_err.append(data.virial(pump_size[i])['pDiff_err'])
            sigzz_pdiff.append(data.sigwall(pump_size[i])['pDiff'])
            sigzz_err.append(data.sigwall(pump_size[i])['pDiff_err'])

        markers_a, caps, bars= ax.errorbar(pump_size, vir_pdiff, None, yerr=vir_err, ls=lt, fmt=mark,
                label='Virial (Fluid)', capsize=1.5, markersize=1.5, alpha=1)
        markers2, caps2, bars2= ax.errorbar(pump_size, sigzz_pdiff, None, yerr=sigzz_err, ls=lt, fmt='x',
                label='$\sigma_{zz}$ (Solid)', capsize=1.5, markersize=3, alpha=1)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        [bar2.set_alpha(0.5) for bar2 in bars2]
        [cap2.set_alpha(0.5) for cap2 in caps2]


    def pgrad_mflowrate(self):
        """
        Plots the pressure gradient on the x-axis and the mass flow rate on the y-axis
        To compare with the Hagen-Poiseuille equation
        """
        ax = self.axes_array[0]
        # mpl.rcParams.update({'lines.markersize': 10})
        ax.set_xlabel(labels[9])
        ax.set_ylabel(labels[10])
        ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        pGrad, shear_rate, \
        mflowrate_ff_avg, mflowrate_ff_err, mflowrate_fc_avg, mflowrate_fc_err, \
        mflowrate_hp, mflowrate_hp_slip = [], [], [], [], [], [], [], []

        for i, val in enumerate(self.datasets_x):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)

            # From FF sietalation
            if 'ff' in val:
                mflowrate_hp.append(data.mflowrate_hp()['mflowrate_hp'])
                mflowrate_hp_slip.append(data.mflowrate_hp()['mflowrate_hp_slip'])
                pGrad.append(np.absolute(data.virial()['pGrad']))
                shear_rate.append(data.viscosity_nemd()['shear_rate'])
                mflowrate_ff_avg.append(np.mean(data.mflux()['mflowrate_stable']))
                mflowrate_ff_err.append(sq.get_err(data.mflux()['mflowrate_stable'])['uncertainty'])
            # From FC sietalation
            if 'fc' in val:
                mflowrate_fc_avg.append(np.mean(data.mflux()['mflowrate_stable']))
                mflowrate_fc_err.append(sq.get_err(data.mflux()['mflowrate_stable'])['uncertainty'])

        # Plot with error bars
        if self.config['err_caps'] or self.config['err_fill']:
            if mflowrate_ff_avg:
                self.plot_uncertainty(ax, pGrad, np.array(mflowrate_ff_avg)*1e18, None, np.array(mflowrate_ff_err)*1e18)
            if mflowrate_fc_avg:
                self.plot_uncertainty(ax, pGrad, np.array(mflowrate_fc_avg)*1e18, None, np.array(mflowrate_fc_err)*1e18)

        ax.plot(pGrad, np.array(mflowrate_hp)*1e18)
        ax.plot(pGrad, np.array(mflowrate_hp_slip)*1e18)

        ax2 = ax.twiny()
        ax2.set_xlabel(labels[11])
        ax2.set_xscale('log', nonpositive='clip')
        if mflowrate_ff_avg: ax2.plot(shear_rate, np.array(mflowrate_ff_avg)*1e18, ls= ' ', marker= ' ')
        if mflowrate_fc_avg: ax2.plot(shear_rate, np.array(mflowrate_fc_avg)*1e18, ls= ' ', marker= ' ')

        Modify(pGrad, self.fig, self.axes_array, self.configfile)


    def rate_viscosity(self):
        """
        Plots the shear rate on the x-axis and the viscoisty on the y-axis
        To check for shear behavior (thinning/thickening/Newtonian) and
        compare with theoretical models (Power law or Carreau)
        """
        ax = self.axes_array[0]
        if len(self.axes_array)==1: ax.set_xlabel(labels[11])

        ax.set_ylabel(labels[12])
        ax.set_xscale('log', nonpositive='clip')
        # mpl.rcParams.update({'lines.linewidth': 2})

        shear_rate, viscosity = [], []
        shear_rate_err, viscosity_err = [], []
        shear_rate_ff, viscosity_ff = [], []
        shear_rate_ff_err, viscosity_ff_err = [], []
        shear_rate_fc, viscosity_fc = [], []
        shear_rate_fc_err, viscosity_fc_err = [], []
        shear_rate_vib, viscosity_vib = [], []
        shear_rate_rigid, viscosity_rigid = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'mixture' in val:
                print('Plotting Couette rate-viscosity data')
                # pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                visco = data.viscosity_nemd()
                shear_rate.append(visco['shear_rate'])
                viscosity.append(visco['eta'])
                print(viscosity)
                shear_rate_err.append(visco['shear_rate_hi'] - visco['shear_rate_lo'])
                viscosity_err.append(visco['eta_hi'] - visco['eta_lo'])

            if 'pentane' in val:
                print('Plotting FF rate-viscosity data')
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                visco = data.viscosity_nemd()
                shear_rate_ff.append(visco['shear_rate'])
                viscosity_ff.append(visco['eta'])
                shear_rate_ff_err.append(visco['shear_rate_hi'] - visco['shear_rate_lo'])
                viscosity_ff_err.append(visco['eta_hi'] - visco['eta_lo'])

            if 'fc' in val:
                print('Plotting FC rate-viscosity data')
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_fc.append(data.viscosity_nemd()['shear_rate'])
                viscosity_fc.append(data.viscosity_nemd()['eta'])
                shear_rate_fc_err.append(data.viscosity_nemd()['shear_rate_hi'] - data.viscosity_nemd()['shear_rate_lo'])
                viscosity_fc_err.append(data.viscosity_nemd()['eta_hi'] - data.viscosity_nemd()['eta_lo'])

            if 'vib' in val:
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_vib.append(data.viscosity_nemd()['shear_rate'])
                viscosity_vib.append(data.viscosity_nemd()['eta'])

            if 'rigid' in val:
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_rigid.append(data.viscosity_nemd()['shear_rate'])
                viscosity_rigid.append(data.viscosity_nemd()['eta'])

        # Plot raw data (with fit if specified)
        if not self.config['err_caps'] and not self.config['err_fill']:
            if viscosity:
                ax.plot(shear_rate, viscosity)
            if viscosity_ff:
                ax.plot(shear_rate_ff, viscosity_ff)
            # if viscosity_fc and not viscosity_vib:
            #     ax.plot(shear_rate_fc, viscosity_fc)
            # if viscosity_vib:
            #     ax.plot(shear_rate_rigid, viscosity_rigid)
            #     ax.plot(shear_rate_vib, viscosity_vib)

        # Plot data with error bars
        if self.config['err_caps'] or self.config['err_fill']:
            if viscosity: self.plot_uncertainty(ax, shear_rate, viscosity, shear_rate_err, viscosity_err)
            if viscosity_ff: self.plot_uncertainty(ax, shear_rate_ff, viscosity_ff, shear_rate_ff_err, viscosity_ff_err)
            if viscosity_fc: self.plot_uncertainty(ax, shear_rate_fc, viscosity_fc, shear_rate_fc_err, viscosity_fc_err)

        if self.config['fit']: #plot fit for Couette data
            popt, pcov = curve_fit(funcs.power, shear_rate, viscosity, maxfev=8000)
            ax.plot(shear_rate, funcs.power(shear_rate, *popt))
            popt2, pcov2 = curve_fit(funcs.power, shear_rate_ff, viscosity_ff, maxfev=8000)
            ax.plot(shear_rate_ff, funcs.power(shear_rate_ff, *popt2))

        if viscosity:
            Modify(shear_rate, self.fig, self.axes_array, self.configfile)
        elif viscosity_ff:
            Modify(shear_rate_ff, self.fig, self.axes_array, self.configfile)
        elif viscosity_fc:
            Modify(shear_rate_fc, self.fig, self.axes_array, self.configfile)


    def rate_stress(self):
        """
        Plots the shear rate on the x-axis and the shear stress on the y-axis
        To check for shear behavior (thinning/thickening/Newtonian) and
        compare with theoretical models (Power law or Carreau)
        """
        if len(self.axes_array)==1: # If it is the only plot
            ax=self.axes_array[0]
        else: # If plotted with rate_viscosity figure
            ax=self.axes_array[1]

        ax.set_xlabel(labels[11])
        ax.set_ylabel('$\sigma_{xz}$ (MPa)')
        ax.set_xscale('log', nonpositive='clip')

        shear_rate, stress = [], []
        shear_rate_err, stress_err = [], []
        shear_rate_ff, stress_ff = [], []
        shear_rate_ff_err, stress_ff_err = [], []
        shear_rate_fc, stress_fc = [], []
        shear_rate_fc_err, stress_fc_err = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                print('Plotting Couette rate-stress data')
                pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.viscosity_nemd()['shear_rate'])
                stress.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_err.append(data.viscosity_nemd()['shear_rate_hi'] - data.viscosity_nemd()['shear_rate_lo'])
                stress_err.append(data.sigwall()['sigxz_err_t'])

            if 'ff' in val:
                print('Plotting FF rate-stress data')
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_ff.append(data.viscosity_nemd()['shear_rate'])
                stress_ff.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_ff_err.append(data.viscosity_nemd()['shear_rate_hi'] - data.viscosity_nemd()['shear_rate_lo'])
                stress_ff_err.append(data.sigwall()['sigxz_err_t'])

            if 'fc' in val:
                print('Plotting FC rate-stress data')
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_fc.append(data.viscosity_nemd()['shear_rate'])
                stress_fc.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_fc_err.append(data.viscosity_nemd()['shear_rate_hi'] - data.viscosity_nemd()['shear_rate_lo'])
                stress_fc_err.append(data.sigwall()['sigxz_err_t'])

        if shear_rate:
            ax.plot(shear_rate, stress)
        if shear_rate_ff:
            ax.plot(shear_rate_ff, stress_ff)
        if shear_rate_fc:
            ax.plot(shear_rate_fc, stress_fc)

        if self.config['fit']:
            popt, pcov = curve_fit(funcs.power, shear_rate, stress, maxfev=8000)
            ax.plot(shear_rate, funcs.power(shear_rate, *popt))

        # if len(self.axes_array)==1: # If it is the only plot
        if shear_rate:
            Modify(shear_rate, self.fig, self.axes_array, self.configfile)
        elif shear_rate_ff:
            Modify(shear_rate_ff, self.fig, self.axes_array, self.configfile)
        elif shear_rate_fc:
            Modify(shear_rate_fc, self.fig, self.axes_array, self.configfile)

    def rate_slip(self):
        """
        Plots the shear rate on the x-axis and the slip length on the y-axis
        To check for whether the slip length reaches a plateau or is unbounded
        """
        ax=self.axes_array[0]
        ax.set_xlabel(labels[11])
        ax.set_ylabel('Slip Length $b$ (nm)')
        ax.set_xscale('log', nonpositive='clip')

        shear_rate, slip = [], []
        shear_rate_pd, slip_pd = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.viscosity_nemd()['shear_rate'])
                slip.append(data.slip_length()['b'])

            if 'ff' in val or 'fc' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_pd.append(data.viscosity_nemd()['shear_rate'])
                slip_pd.append(data.slip_length()['b'])

        if shear_rate: ax.plot(shear_rate, slip)
        if shear_rate_pd: ax.plot(shear_rate_pd, slip_pd)

        if shear_rate:
            Modify(shear_rate, self.fig, self.axes_array, self.configfile)
        elif shear_rate_pd:
            Modify(shear_rate_pd, self.fig, self.axes_array, self.configfile)


    def rate_temp(self):
        """
        Plots the shear rate on the x-axis and the temperature on the y-axis
        """
        ax=self.axes_array[0]

        ax.set_xlabel(labels[11])
        ax.set_ylabel(labels[6])
        ax.set_xscale('log', nonpositive='clip')
        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, temp = [], []
        shear_rate_pd, temp_pd = [], []
        shear_rate_vib, shear_rate_rigid, temp_vib, temp_rigid = [], [], [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.viscosity_nemd()['shear_rate'])
                temp.append(np.mean(data.temp()['temp_t']))

            if 'ff' in val or 'fc' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_pd.append(data.viscosity_nemd()['shear_rate'])
                temp_pd.append(np.mean(data.temp()['temp_t']))

            if 'vibrating' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_vib.append(data.viscosity_nemd()['shear_rate'])
                temp_vib.append(np.mean(data.temp()['temp_t']))

            if 'rigid' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_rigid.append(data.viscosity_nemd()['shear_rate'])
                temp_rigid.append(np.mean(data.temp()['temp_t']))

        if shear_rate: ax.plot(shear_rate, temp)
        if shear_rate_pd and not shear_rate_vib: ax.plot(shear_rate_pd, temp_pd)
        if shear_rate_vib:
            ax.plot(shear_rate_rigid, temp_rigid)
            ax.plot(shear_rate_vib, temp_vib)

        if shear_rate:
            Modify(shear_rate, self.fig, self.axes_array, self.configfile)
        elif shear_rate_pd:
            Modify(shear_rate_pd, self.fig, self.axes_array, self.configfile)


    def rate_qdot(self):
        """
        Plots the shear rate on the x-axis and the heat flow rate on the y-axis
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[11])
        ax.ticklabel_format(axis='y', style='sci', useOffset=False)
        ax.set_ylabel(r'${\mathrm{\dot{Q}}} \times 10^{-6}}$ (W)')
        ax.set_xscale('log', nonpositive='clip')
        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, qdot, qdot_continuum = [], [], []
        shear_rate_beskok, qdot_beskok = [], []

        shear_rate_lgv, qdot_lgv = [], []
        shear_rate_be, qdot_be = [], []
        shear_rate_nh, qdot_nh = [], []

        for idx, val in enumerate(self.datasets_x):
            pumpsize = self.pumpsize
            log_file = os.path.dirname(os.path.abspath(self.datasets_x[idx]))+'/log.lammps'
            data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
            cond = data.conductivity_ecouple(log_file)
            if 'beskok' not in val:
                shear_rate.append(data.viscosity_nemd()['shear_rate'])
                qdot.append(np.mean(cond['qdot'])*1e6)
                qdot_continuum.append(np.mean(cond['qdot_continuum'])*1e6)
            else:
                shear_rate_beskok.append(data.viscosity_nemd()['shear_rate'])
                qdot_beskok.append(np.mean(cond['qdot'])*1e6)
                # qdot_continuum.append(np.mean(cond['qdot_continuum'])*1e6)
            # if 'lgv' in val:
            #     shear_rate_lgv.append(data.viscosity_nemd()['shear_rate'])
            #     qdot_lgv.append(np.mean(cond['qdot'])*1e6)
            # if 'berendsen' in val:
            #     shear_rate_be.append(data.viscosity_nemd()['shear_rate'])
            #     qdot_be.append(np.mean(cond['qdot'])*1e6)
            # if 'nh' in val:
            #     shear_rate_nh.append(data.viscosity_nemd()['shear_rate'])
            #     qdot_nh.append(np.mean(cond['qdot'])*1e6)

        ax.plot(shear_rate, qdot)
        if shear_rate_beskok: ax.plot(shear_rate_beskok, qdot_beskok)
        ax.plot(shear_rate, qdot_continuum)

        # ax.plot(shear_rate_lgv, qdot_lgv)
        # ax.plot(shear_rate_lgv, qdot_lgv)
        # ax.plot(shear_rate_be, qdot_be)
        # ax.plot(shear_rate_nh, qdot_nh)

        Modify(shear_rate_lgv, self.fig, self.axes_array, self.configfile)


    def rate_conductivity(self):
        """
        Plots the shear rate on the x-axis and the thermal conductivity (λ) on the y-axis
        The continuum λ comes from Fourier's law of heat transfer
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[11])
        ax.ticklabel_format(axis='y', style='sci', useOffset=False)
        ax.set_ylabel(r'${\mathrm{\lambda} (W/m^2.K)}$')
        ax.set_xscale('log', nonpositive='clip')
        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate_lgv, conductivity_z_lgv, shear_rate_berendsen, \
        conductivity_z_berendsen, shear_rate_nh, conductivity_z_nh = [], [], [], [], [], []

        for idx, val in enumerate(self.datasets_x):
            pumpsize = self.pumpsize
            log_file = os.path.dirname(os.path.abspath(self.datasets_x[idx]))+'/log.lammps'
            data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)

            if 'lgv' in val:
                shear_rate_lgv.append(data.viscosity_nemd()['shear_rate'])
                conductivity_z_lgv.append(np.mean(data.conductivity_IK()['conductivity_z']))
                # conductivity_continuum.append(np.mean(data.conductivity_IK()['conductivity_continuum']))
            if 'berendsen' in val:
                shear_rate_berendsen.append(data.viscosity_nemd()['shear_rate'])
                conductivity_z_berendsen.append(np.mean(data.conductivity_ecouple(log_file)['conductivity_z']))
            if 'nh' in val:
                shear_rate_nh.append(data.viscosity_nemd()['shear_rate'])
                conductivity_z_nh.append(np.mean(data.conductivity_ecouple(log_file)['conductivity_z']))

        if shear_rate_lgv: ax.plot(shear_rate_lgv, conductivity_z_lgv)
        if shear_rate_berendsen: ax.plot(shear_rate_berendsen, conductivity_z_berendsen)
        if shear_rate_nh: ax.plot(shear_rate_nh, conductivity_z_nh)

        # ax.plot(shear_rate, conductivity_continuum)

        Modify(shear_rate_lgv, self.fig, self.axes_array, self.configfile)


    def thermal_conduct(self):
        """
        Plots the thermal conductivity with Pressure. Equilibrium sietalations.
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[7])
        ax.set_ylabel('$\lambda$ (W/mK)')

        # Experimal results are from Wang et al. 2020 (At temp. 335 K)
        exp_conductivity = [0.1009,0.1046,0.1113,0.1170]
        exp_press = [5,10,20,30]
        md_conductivity, md_press = [], []

        for idx, val in enumerate(self.datasets_x):
            data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
            md_conductivity.append(np.mean(data.conductivity_gk()['conductivity_tot']))

        ax.plot(exp_press, md_conductivity)
        ax.plot(exp_press, exp_conductivity)

        Modify(exp_press, self.fig, self.axes_array, self.configfile)


    def eos(self):
        """
        Plots the equation of state of the fluid
        Equilibrium sietaaltions can be:
            * Isotherms (NPT) with density as output, plot ρ vs P
                * Gautschi Integtator
                * Velocity Verlet Integrator
            * Isochores (NVT) with pressure as output, plot T vs P
        Nonequilibrium sietalations:
            * Change of T with P, to see the effect of the pump ΔP on the fluid temperature
        """

        if self.config['log']:
            # self.axes_array[0].xaxis.major.formatter.set_scientific(False)
            self.axes_array[0].set_xscale('log', base=10)
            self.axes_array[0].set_yscale('log', base=10)
            self.axes_array[0].xaxis.set_minor_formatter(ScalarFormatter())
            self.axes_array[1].set_xscale('log', base=10)
            self.axes_array[1].set_yscale('log', base=10)
            self.axes_array[1].xaxis.set_minor_formatter(ScalarFormatter())
            self.fig.supylabel('Pressure $P/P_{o}$')
            self.fig.text(0.3, 0.04, r'Density $\rho/\rho_{o}$', ha='center')
            self.fig.text(0.72, 0.04, 'Temperature $T/T_{o}$', ha='center')
        else:
            if len(self.axes_array)>1:
                self.fig.supylabel('Pressure $P/P_{o}$')
                self.fig.text(0.3, 0.04, r'Density $\rho/\rho_{o}$', ha='center')
                self.fig.text(0.72, 0.04, 'Temperature $T/T_{o}$', ha='center')
            else:
                self.axes_array[0].set_xlabel(labels[3])
                self.axes_array[0].set_ylabel(labels[7])

        if len(self.axes_array)>1: self.axes_array[1].tick_params(labelleft=False)  # don't put tick labels on the left

        ds_nemd, ds_isochores, ds_isotherms = [], [], []
        press_isotherms, den_isotherms = [], []
        press_isochores, temp_list = [], []
        ds_gau_1fs, ds_ver_1fs, ds_gau_4fs, ds_ver_4fs = [], [], [], []
        den_isotherms_gau_1fs, den_isotherms_gau_4fs, den_isotherms_ver_1fs, den_isotherms_ver_4fs = [], [], [], []
        press_isotherms_gau_1fs, press_isotherms_gau_4fs, press_isotherms_ver_1fs, press_isotherms_ver_4fs = [], [], [], []

        for idx, val in enumerate(self.datasets_x):
            # if 'ff' in val or 'fc' in val or 'couette' in val:
            if 'flow' in val:
                ds_nemd.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize))
            if 'isochore' in val:
                ds_isochores.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'isotherm' in val:
                ds_isotherms.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))

        for i in ds_nemd:
            # print(np.mean(i.density()['den_t']))
            out_chunk = np.where(i.virial()['vir_X'] == np.amax(i.virial()['vir_X']))[0][0]
            press = i.virial()['vir_X'][out_chunk:-5] / np.mean(i.virial()['vir_t'])
            temp = i.temp()['temp_X'][out_chunk:-5] / np.mean(i.temp()['temp_t'])
            den = i.density()['den_X'][out_chunk:-5] / np.mean(i.density()['den_t'])
            self.axes_array[0].plot(den, press)
            self.axes_array[1].plot(temp, press)
            coeff,_ = np.polyfit(press, temp, 1)
            print(f'Joule-Thomson coefficient is {coeff} K/MPa')

            if self.config['log']:
                coeffs_den = curve_fit(funcs.power, den, press, maxfev=8000)
                coeffs_temp = curve_fit(funcs.power, temp, press, maxfev=8000)
                print(f'Polytropic index (k) is {-coeffs_den[0][1]}')
                print(f'Polytropic index (k) is {coeffs_temp[0][1]/(1-coeffs_temp[0][1])}')
                self.axes_array[0].plot(den, funcs.power(den, coeffs_den[0][0], coeffs_den[0][1], coeffs_den[0][2]))
                self.axes_array[1].plot(temp, funcs.power(temp, coeffs_temp[0][0], coeffs_temp[0][1], coeffs_temp[0][2]))

        # Isotherms -----------------
        for i in ds_isotherms:
            print(len(i.density()['den_t'][:1000]))
            den_isotherms.append(np.mean(i.density()['den_t'][:1000]))
            press_isotherms.append(np.mean(i.virial()['vir_t']))

        den_isotherms, press_isotherms = np.asarray(den_isotherms), np.asarray(press_isotherms)

        if ds_isotherms:
            self.axes_array[0].plot(den_isotherms, press_isotherms)

            # Fit to cubic EOS
            coeffs_den_cubic = curve_fit(funcs.cubic, den_isotherms, press_isotherms, maxfev=8000)
            coeffs_den_units = curve_fit(funcs.cubic, den_isotherms*1000, press_isotherms*1e6, maxfev=8000) # with the SI units
            print(f'Coefficients of the cubic EOS for n-pentane are {coeffs_den_units[0][0], coeffs_den_units[0][1], coeffs_den_units[0][2], coeffs_den_units[0][3]}')
            self.axes_array[0].plot(den_isotherms, funcs.cubic(den_isotherms, coeffs_den_cubic[0][0],
             coeffs_den_cubic[0][1], coeffs_den_cubic[0][2], coeffs_den_cubic[0][3]))

            # Experimental data (K. Liu et al. / J. of Supercritical Fluids 55 (2010) 701–711)
            exp_density = [0.630, 0.653, 0.672, 0.686, 0.714, 0.739, 0.750]
            exp_press = [28.9, 55.3, 84.1, 110.2, 171.0, 239.5, 275.5]
            self.axes_array[0].plot(exp_density, exp_press)

            if self.config['log']:
                den_isotherms /= np.max(den_isotherms)
                press_isotherms /= np.max(press_isotherms)

                coeffs_den = curve_fit(funcs.power, den_isotherms, press_isotherms, maxfev=8000)
                print(f'Adiabatic exponent (gamma) is {coeffs_den[0][1]}')
                self.axes_array[0].plot(den_isotherms, funcs.power(den_isotherms, coeffs_den[0][0], coeffs_den[0][1], coeffs_den[0][2]))

        # Isochores -----------------
        for i in ds_isochores:
            temp_list.append(np.mean(i.temp()['temp_t']))
            press_isochores.append(np.mean(i.virial()['vir_t']))

        temp_lit, press_isochores = np.asarray(temp_list), np.asarray(press_isochores)

        if ds_isochores:
            self.axes_array[0].plot(temp_list, press_isochores)
            if len(self.axes_array)>1: self.axes_array[1].plot(temp_list, press_isochores)

            if self.config['log']:
                temp_list /= np.max(temp_list)
                press_isochores /= np.max(press_isochores)

                coeffs_temp = curve_fit(funcs.power, temp_list, press_isochores, maxfev=8000)
                print(f'Adiabatic exponent (gamma) is {coeffs_temp[0][1]}')
                if len(self.axes_array)>1: self.axes_array[1].plot(temp_list, funcs.power(temp_list, coeffs_temp[0][0], coeffs_temp[0][1], coeffs_temp[0][2]))

        if ds_nemd: Modify(press, self.fig, self.axes_array, self.configfile)
        elif ds_isotherms: Modify(press_isotherms, self.fig, self.axes_array, self.configfile)
        elif ds_isochores: Modify(press_isochores, self.fig, self.axes_array, self.configfile)


    def coexistence_curve(self):
        """
        Plots the Liquid-Vapor coexistence curve with density ρ on the x-axis and the temperature on the y-axis
        To check with experimental data
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[3])
        ax.set_ylabel('Temperature (K)')

        temp, rho_l, rho_v = [], [], []
        for i in range(len(self.datasets_x)):
            ds = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            temp.append(np.mean(ds.temp()['temp_t']))
            rho_l.append(ds.coexistence_densities()['rho_l'])
            rho_v.append(ds.coexistence_densities()['rho_v'])

        # Plot MD data
        ax.plot(rho_v,temp)
        ax.plot(rho_l,temp)

        if self.mf ==39.948:
            #Exp.: Lemmon et al. (NIST)
            rhoc, Tc = 0.5356,150.65 # g/cm3, K
            temp_exp = [85,90,95,100,105,110,115,120,125,130,135]
            rho_v_exp = [0.0051,0.0085,0.0136,0.0187,0.0255,0.0373,0.0441,0.0611,0.0798,0.1052,0.1392]
            rho_l_exp = [1.4102,1.3813,1.3491,1.3152,1.2812,1.2439,1.2032,1.1624,1.1166,1.0674,1.0080]
            ax.plot(rho_v_exp,temp_exp)
            ax.plot(rho_l_exp,temp_exp)

        if self.mf == 72.15:
            # Exp.: Thol, M., Uhde, T., Lemmon, E.W., and Span, R., 2018. (NIST)
            rhoc, Tc = 0.232, 469.70  # g/cm3, K
            temp_exp = [150,175,200,225,250,275,300,325,350,375,400,425]
            rho_v_exp = [0, 0, 0, 0.00006, 0.0003, 0.0009, 0.0022, 0.0048, 0.0094, 0.0170, 0.0291, 0.0491]
            rho_l_exp = [0.7575,0.7339,0.7115,0.6891,0.6661,0.6431,0.6201,0.5940,0.5662,0.5341,0.5008,0.4554]

            # MC: J. Phys. Chem. B, Vol. 102, No. 14, 1998
            rhoc_mc, Tc_mc = 0.239, 471
            temp_mc = [300, 338, 373, 403, 440]
            rho_l_mc = [0.625, 0.583, 0.540, 0.499, 0.419]
            rho_v_mc = [0.0034, 0.0088, 0.0196, 0.0413, 0.0684]

            # Plot Exp. and MC data
            ax.plot(rho_v_exp,temp_exp)
            ax.plot(rho_l_exp,temp_exp)

            ax.plot(rho_v_mc, temp_mc)
            ax.plot(rho_l_mc, temp_mc)

        # For Density scaling law ---> Get Tc
        rho_diff = np.array(rho_l) - np.array(rho_v)
        popt, pcov = curve_fit(funcs.density_scaling, rho_diff/np.max(rho_diff), temp, maxfev=8000)
        Tc_fit = popt[1]

        # Plot the fits for MD data
        ax.plot(rho_l,funcs.density_scaling(rho_diff/np.max(rho_diff), *popt))
        ax.plot(rho_v,funcs.density_scaling(rho_diff/np.max(rho_diff), *popt))

        # For law of rectilinear diameters ---> Get rhoc
        rho_sum = np.array(rho_l) + np.array(rho_v)
        popt2, pcov2 = curve_fit(funcs.rectilinear_diameters, rho_sum, temp, maxfev=8000)
        rhoc_fit=popt2[1]/(popt2[0]*2)
        print(f'Critical Point is:({rhoc_fit,Tc_fit})')

        # Plot the critical points from MD, Exp. and MC
        ax.plot(rhoc_fit, Tc_fit)
        ax.plot(rhoc, Tc)
        ax.plot(rhoc_mc, Tc_mc)

        Modify(rho_l, self.fig, self.axes_array, self.configfile)


    def rc_gamma(self):
        """
        Plots the cutoff radius on the x-axis and the surface tension (γ) on the y-axis
        To check that the surface tension converges with the cutoff radius
        """
        ax = self.axes_array[0]
        ax.set_xlabel('$r_c/\sigma$')
        ax.set_ylabel('Surface tension $\gamma$ (mN/m)')

        if self.mf == 39.948: rc = [3,4,5,6,7]
        if self.mf == 72.15: rc = [10,12,14,18,21,24]
        gamma =  []
        for i in range(len(self.datasets_x)):
            ds = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            gamma.append(np.mean(ds.surface_tension()['gamma'])*1e3)

        ax.plot(rc,gamma)

        Modify(rc, self.fig, self.axes_array, self.configfile)


    def temp_gamma(self):
        """
        Plots the temperature on the x-axis and the surface tension (γ) on the y-axis
        To check with experimental data
        """
        ax = self.axes_array[0]
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Surface tension $\gamma$ (mN/m)')

        temp, gamma = [], []
        temp_gautschi, gamma_gautschi = [], []
        temp_gautschi_5, gamma_gautschi_5 = [], []

        if self.mf == 39.948: gamma_exp = [11.871,10.661,9.7482,8.3243,7.2016,6.1129,5.0617,4.0527,3.0919,2.2879]
        if self.mf == 72.15: gamma_exp = [32.5,29.562,26.2638,23.736,20.864,18.032,15.250,12.534,9.9014,7.379,5.0029]#,2.8318,0.98]

        for idx, val in enumerate(self.datasets_x):
            if 'gautschi' not in val:
                ds = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                temp.append(np.mean(ds.temp()['temp_t']))
                gamma.append(np.mean(ds.surface_tension()['gamma'])*1e3)
            if 'gautschi-1fs' in val:
                ds = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                temp_gautschi.append(np.mean(ds.temp()['temp_t']))
                gamma_gautschi.append(np.mean(ds.surface_tension()['gamma'])*1e3)
            if 'gautschi-5fs' in val:
                ds = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                temp_gautschi_5.append(np.mean(ds.temp()['temp_t']))
                gamma_gautschi_5.append(np.mean(ds.surface_tension()['gamma'])*1e3)

        if temp:
            ax.plot(temp, gamma)
            popt, pcov = curve_fit(funcs.power_new, temp/np.max(temp), gamma, maxfev=8000)
            print(f'Critical temperature: Tc = {popt[1]*np.max(temp):.2f}')
            ax.plot(temp, gamma_exp)
            ax.plot(temp,funcs.power_new(temp/np.max(temp), *popt))
        if temp_gautschi:
            ax.plot(temp_gautschi, gamma_gautschi)
        if temp_gautschi_5:
            ax.plot(temp_gautschi_5, gamma_gautschi_5)

        Modify(temp, self.fig, self.axes_array, self.configfile)


    def acf(self):
        """
        Plots the auto correlation function
        # TODO: Check that this works properly!
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[2])
        ax.set_ylabel(r'${\mathrm{C_{AA}}}$')
        mpl.rcParams.update({'lines.linewidth':'1'})

        cutoff = 1000

        for i in range(len(self.datasets_x)):
            ds = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            acf_sigxz = sq.acf(ds.virial()['Wxz_t'])['norm']
            acf_sigxz_numpy = sq.acf_conjugate(ds.virial()['Wxz_t'])['C']
            ax.plot(self.time[:cutoff]*1e-6, acf_sigxz[:cutoff])
            ax.plot(self.time[:cutoff]*1e-6, acf_sigxz_numpy[:cutoff])

            # acf_flux = sq.acf(data.mflux()['jx_t'])['norm']
            # ax.plot(self.time*1e-6, acf_flux[:10000])

        ax.axhline(y= 0, color='k', linestyle='dashed', lw=1)

        Modify(self.time[:cutoff]*1e-6, self.fig, self.axes_array, self.configfile)



    def cav_num_length(self, pl, pv):
        """
        Plots the auto correlation function
        # TODO: Check that this works properly!
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[1])
        ax.set_ylabel('Cavitation number $K$')
        # mpl.rcParams.update({'lines.linewidth':'1'})

        for i in range(len(self.datasets_x)):
            ds = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            cav_num = ds.cavitation_num(pl, pv)
            ax.plot(ds.length_array, cav_num)

            # acf_flux = sq.acf(data.mflux()['jx_t'])['norm']
            # ax.plot(self.time*1e-6, acf_flux[:10000])

        # ax.axhline(y= 0, color='k', linestyle='dashed', lw=1)

        Modify(ds.length_array, self.fig, self.axes_array, self.configfile)



    def struc_factor(self, fluid):
        """
        Plots the structure factor on the x-axis and the wave length (kx or ky) on the y-axis in 2D figure
        or both (kx and ky) on the y- and z- axes in a 3D figure
        """
        data = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)

        kx = data.sf()['kx']
        ky = data.sf()['ky']
        # k = data.sf()['k']

        sfx = data.sf()['sf_x']
        sfy = data.sf()['sf_y']
        sf_time = data.sf()['sf_time']
        if fluid == 1:
            sf = data.sf()['sf']
        else:
            sf = data.sf()['sf_solid']
        # sf_solid = data.sf()['sf_solid']

        if self.config['heat']:
            self.ax.set_ylim(bottom=0, top=ky.max())
            self.ax.set_xlim(left=0, right=kx.max())
            self.ax.set_xlabel(r'$k_x \, \mathrm{(\AA^{-1})}$')
            self.ax.set_ylabel(r'$k_y \, \mathrm{(\AA^{-1})}$')
            Kx, Ky = np.meshgrid(kx, ky)

            plt.imshow(sf.T, cmap='viridis', interpolation='lanczos',
                extent=[kx.min(), kx.max(), ky.min(), ky.max()], origin='lower')
            # plt.colorbar()

        elif self.config['3d']:
            self.ax.set_xlabel('$k_x \, (\AA^{-1})$')
            self.ax.set_ylabel('$k_y \, (\AA^{-1})$')
            self.ax.set_zlabel('$S(k)$')
            self.ax.invert_xaxis()
            self.ax.set_ylim(ky[-1]+1,0)
            self.ax.zaxis.set_rotate_label(False)
            self.ax.set_zticks([])

            Kx, Ky = np.meshgrid(kx, ky)

            self.ax.plot_surface(Kx, Ky, sf.T, cmap=mpl.cm.jet,
                rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
            # self.fig.colorbar(surf, shrink=0.5, aspect=5)
            self.ax.view_init(35,60) #(90,0)

        else:
            a = input('x or y or r or t:')
            ax = self.axes_array[0]
            if a=='x':
                ax.set_xlabel('$k_x (\AA^{-1})$')
                ax.set_ylabel('$S(K_x)$')
                ax.plot(kx, sfx)
                Modify(kx, self.fig, self.axes_array, self.configfile)
            elif a=='y':
                ax.set_xlabel('$k_y (\AA^{-1})$')
                ax.set_ylabel('$S(K_y)$')
                ax.plot(ky, sfy)
                Modify(ky, self.fig, self.axes_array, self.configfile)
            elif a=='t':
                ax.set_xlabel('$t (fs)$')
                ax.set_ylabel('$S(K)$')
                ax.plot(self.time[self.skip:], sf_time, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
                Modify(self.time[self.skip:], self.fig, self.axes_array, self.configfile)

    def isf(self):
        """
        Plots the Intermediate structure factor
        # TODO: Check that this works properly!
        """
        ax = self.axes_array[0]

        data = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        isf_lo = data.ISF()['ISF'][:,0,0]
        isf_hi = data.ISF()['ISF'][:,50,10]
        # print(isf.shape)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('I(K)')
        ax.plot(self.time[self.skip:], isf_lo)
        ax.plot(self.time[self.skip:], isf_hi)
        Modify(self.time[self.skip:], self.fig, self.axes_array, self.configfile)


    def transverse_acf(self):
        """
        Plots the transverse auto correlation function
        # TODO: Check that this works properly!
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[2])
        ax.set_ylabel(r'${\mathrm C_{AA}}$')
        mpl.rcParams.update({'lines.linewidth':'1'})

        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            acf_flux = data.trans()['a']
            ax.plot(self.time[:10000]*1e-6, acf[:10000])

        ax.axhline(y= 0, color='k', linestyle='dashed', lw=1)

        Modify(self.time[:10000]*1e-6, self.fig, self.axes_array, self.configfile)
