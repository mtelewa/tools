#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys, os, logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy.constants as sci
from scipy.optimize import curve_fit
import yaml
import funcs
import sample_quality as sq
from compute_thermo import ExtractFromTraj as dataset
from plot_settings import Initialize, Modify


# Uncomment to change Matplotlib backend
# mpl.use('TkAgg')

# Logger Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import golbal plot configuration
plt.style.use('imtek')

#           0             1                   2                    3
labels=('Height (nm)','Length (nm)', 'Time (ns)', r'Density (g/${\mathrm{cm^3}}$)',
#           4                                           5             6                    7
        r'${\mathrm{j_x}}$ (g/${\mathrm{m^2}}$.ns)', 'Vx (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\mathrm{(Force)}}$ (pN)', r'${\mathrm{dP / dx}}$ (MPa/nm)',
#           10                                          11
        r'${\mathrm{\dot{m}}} \times 10^{-18}}$ (g/ns)', r'${\mathrm{\dot{\gamma}} (s^{-1})}$',
#           12
        r'${\mathrm{\eta}}$ (mPa.s)', r'$N_{\mathrm{\pump}}$', r'Energy (Kcal/mol)', '$R(t)$ (${\AA}$)')

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

        # try:
        first_dataset = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        self.Nx = len(first_dataset.length_array)
        self.Nz = len(first_dataset.height_array)
        self.time = first_dataset.time
        self.Ly = first_dataset.Ly


    def plot_uncertainty(self, ax, x, y, arr):
        """
        Plots the uncertainty of the data
        """
        if self.config['err_caps'] is not None:
            err = sq.get_err(arr)['uncertainty']
            markers, caps, bars= ax.errorbar(x, y, xerr=None, yerr=err,
                                        capsize=3.5, markersize=4, lw=2, alpha=0.8)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        if self.config['err_fill'] is not None:
            lo, hi = sq.get_err(arr)['lo'], sq.get_err(arr)['hi']
            ax.fill_between(x, lo, hi, color=color, alpha=0.4)


    def v_distrib(self):
        """
        Plots the thermal velocity distributions for the three velocity components to check
        for local thermodynamic equilibrium (LTE) condition (zero skewness and kurtosis)
        """
        ax = self.axes_array[0]
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('$f(v)$')

        first_dataset = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        T =  np.mean(first_dataset.temp()['temp_t'])
        vx = first_dataset.vel_distrib()['vx_values_lte']
        vx = np.array(vx)
        kb = 8.314462618 # J/mol.K
        mb_distribution = np.sqrt((self.mf / (2*5*np.pi*kb*T))) * np.exp((-self.mf*vx**2)/(2*5*kb*T))

        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            vx_values = data.vel_distrib()['vx_values_lte']
            vx_prob = data.vel_distrib()['vx_prob_lte']

            vy_values = data.vel_distrib()['vy_values_lte']
            vy_prob = data.vel_distrib()['vy_prob_lte']

            vz_values = data.vel_distrib()['vz_values_lte']
            vz_prob = data.vel_distrib()['vz_prob_lte']

            ax.plot(vx_values, vx_prob)
            ax.plot(vy_values, vy_prob)
            ax.plot(vz_values, vz_prob)
            # TODO : Add the Maxwell-Boltzmann distribution
            # ax.plot(vx_values, mb_distribution)

        Modify(vx_values, self.fig, self.axes_array, self.configfile)


    def v_evolution(self):
        """
        Plots the streaming velocity profile in 5 regions along the flow direction
        This is mainly to observe the compressibility of the fluid along the stream
        """
        ax = self.axes_array[0]
        ax.set_xlabel('Height (nm)')
        ax.set_ylabel('$V_{x}$ (m/s)')

        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            h = data.height_array
            vx_R1 =  data.velocity()['vx_R1']
            vx_R2 =  data.velocity()['vx_R2']
            vx_R3 =  data.velocity()['vx_R3']
            vx_R4 =  data.velocity()['vx_R4']
            vx_R5 =  data.velocity()['vx_R5']

            ax.plot(h[vx_R1!=0][1:-1], vx_R1[vx_R1!=0][1:-1])
            ax.plot(h[vx_R2!=0][1:-1], vx_R2[vx_R2!=0][1:-1])
            ax.plot(h[vx_R3!=0][1:-1], vx_R3[vx_R3!=0][1:-1])
            ax.plot(h[vx_R4!=0][1:-1], vx_R4[vx_R4!=0][1:-1])
            ax.plot(h[vx_R5!=0][1:-1], vx_R5[vx_R5!=0][1:-1])

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
        mpl.rcParams.update({'lines.markersize': 7})
        ax.set_xlabel(labels[9])
        ax.set_ylabel(labels[10])
        ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        pGrad, shear_rate, \
        mflowrate_ff_avg, mflowrate_ff_err, mflowrate_fc_avg, mflowrate_fc_err, \
        mflowrate_hp, mflowrate_hp_slip = [], [], [], [], [], [], [], []

        for i, val in enumerate(self.datasets_x):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)

            # Continuum prediction (Hagen Poiseuille)
            if 'cont' in val:
                mflowrate_hp.append(data.mflowrate_hp()['mflowrate_cont'])
                mflowrate_hp_slip.append(data.mflowrate_hp()['mflowrate_hp_slip'])
                # avg_gap_height = np.mean(data.h)*1e-9
                # bulk_den_avg = np.mean(data.density()['den_t'])
                # mu = data.transport()['mu']            # mPa.s

            # From FF simulation
            if 'ff' in val:
                pGrad.append(np.absolute(data.virial()['pGrad']))
                shear_rate.append(data.transport()['shear_rate'])
                mflowrate_ff_avg.append(np.mean(data.mflux()['mflowrate_stable']))
                mflowrate_ff_err.append(sq.get_err(data.mflux()['mflowrate_stable'])['uncertainty'])
            # From FC simulation
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
        mpl.rcParams.update({'lines.linewidth': 2})
        mpl.rcParams.update({'lines.markersize': 8})

        shear_rate, viscosity = [], []
        shear_rate_err, viscosity_err = [], []
        shear_rate_ff, viscosity_ff = [], []
        shear_rate_ff_err, viscosity_ff_err = [], []

        shear_rate_fc, viscosity_fc = [], []
        shear_rate_fc_err, viscosity_fc_err = [], []
        shear_rate_vib, viscosity_vib = [], []
        shear_rate_rigid, viscosity_rigid = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                viscosity.append(data.transport()['mu'])
                shear_rate_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                viscosity_err.append(data.transport()['mu_hi'] - data.transport()['mu_lo'])

            if 'ff' in val:
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_ff.append(data.transport()['shear_rate'])
                viscosity_ff.append(data.transport()['mu'])
                shear_rate_ff_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                viscosity_ff_err.append(data.transport()['mu_hi'] - data.transport()['mu_lo'])

            if 'fc' in val:
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_fc.append(data.transport()['shear_rate'])
                viscosity_fc.append(data.transport()['mu'])
                shear_rate_fc_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                viscosity_fc_err.append(data.transport()['mu_hi'] - data.transport()['mu_lo'])

            if 'vib' in val:
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_vib.append(data.transport()['shear_rate'])
                viscosity_vib.append(data.transport()['mu'])

            if 'rigid' in val:
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_rigid.append(data.transport()['shear_rate'])
                viscosity_rigid.append(data.transport()['mu'])

        # Plot raw data (with fit if specified)
        if not self.config['err_caps'] and not self.config['err_fill']:
            if viscosity:
                ax.plot(shear_rate, viscosity)
            if viscosity_ff and not viscosity_vib:
                ax.plot(shear_rate_ff, viscosity_ff)
            if viscosity_fc and not viscosity_vib:
                ax.plot(shear_rate_fc, viscosity_fc)
            if viscosity_vib:
                ax.plot(shear_rate_rigid, viscosity_rigid)
                ax.plot(shear_rate_vib, viscosity_vib)
            if self.config['fit']: #plot fit for Couette data
                popt, pcov = curve_fit(funcs.power, shear_rate, viscosity, maxfev=8000)
                ax.plot(shear_rate, funcs.power(shear_rate, *popt))

        # Plot data with error bars
        if self.config['err_caps'] or self.config['err_fill']:
            if viscosity: self.plot_uncertainty(ax, shear_rate, viscosity, shear_rate_err, viscosity_err)
            if viscosity_ff: self.plot_uncertainty(ax, shear_rate_ff, viscosity_ff, shear_rate_ff_err, viscosity_ff_err)
            if viscosity_fc: self.plot_uncertainty(ax, shear_rate_fc, viscosity_fc, shear_rate_fc_err, viscosity_fc_err)

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
        mpl.rcParams.update({'lines.markersize': 8})

        shear_rate, stress = [], []
        shear_rate_err, stress_err = [], []
        shear_rate_ff, stress_ff = [], []
        shear_rate_ff_err, stress_ff_err = [], []
        shear_rate_fc, stress_fc = [], []
        shear_rate_fc_err, stress_fc_err = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                stress.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                stress_err.append(data.sigwall()['sigxz_err_t'])

            if 'ff' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_ff.append(data.transport()['shear_rate'])
                stress_ff.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_ff_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                stress_ff_err.append(data.sigwall()['sigxz_err_t'])

            if 'fc' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_fc.append(data.transport()['shear_rate'])
                stress_fc.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_fc_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                stress_fc_err.append(data.sigwall()['sigxz_err_t'])

        # Plot with error bars
        if self.config['err_caps'] or self.config['err_fill']:
            a,b,c = [],[],[]
            if shear_rate:
                for i in stress_err:
                    a.append(np.mean(i))
                pds.plot_err_caps(ax, shear_rate, stress, shear_rate_err, a)
            if shear_rate_ff:
                for j in stress_ff_err:
                    b.append(np.mean(j))
                pds.plot_err_caps(ax, shear_rate_ff, stress_ff, shear_rate_ff_err, b)
            if shear_rate_fc:
                for j in stress_fc_err:
                    c.append(np.mean(j))
                pds.plot_err_caps(ax, shear_rate_fc, stress_fc, shear_rate_fc_err, c)

        # Plot raw data (with fit if specified)
        if not self.config['err_caps']:
            if shear_rate:
                ax.plot(shear_rate, stress)
            if shear_rate_ff:
                ax.plot(shear_rate_ff, stress_ff)
            if shear_rate_fc:
                ax.plot(shear_rate_fc, stress_fc)
            if self.config['fit']:
                popt, pcov = curve_fit(funcs.power, shear_rate, stress)
                ax.plot(shear_rate, funcs.power(shear_rate, *popt))

        if len(self.axes_array)==1: # If it is the only plot
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
        ax.set_ylabel('Slip Length (nm)')
        ax.set_xscale('log', nonpositive='clip')
        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, slip = [], []
        shear_rate_pd, slip_pd = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                slip.append(data.slip_length()['Ls'])

            if 'ff' in val or 'fc' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_pd.append(data.transport()['shear_rate'])
                slip_pd.append(data.slip_length()['Ls'])

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
                shear_rate.append(data.transport()['shear_rate'])
                temp.append(np.mean(data.temp()['temp_t']))

            if 'ff' in val or 'fc' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_pd.append(data.transport()['shear_rate'])
                temp_pd.append(np.mean(data.temp()['temp_t']))

            if 'vibrating' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_vib.append(data.transport()['shear_rate'])
                temp_vib.append(np.mean(data.temp()['temp_t']))

            if 'rigid' in val:
                pumpsize = self.pumpsize
                data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_rigid.append(data.transport()['shear_rate'])
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
        ax.set_ylabel(r'${\mathrm{\dot{Q}}} \times 10^{-24}}$ (W)')
        # ax.set_xscale('log', nonpositive='clip')
        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, qdot, qdot_continuum = [], [], []

        for idx, val in enumerate(self.datasets_x):
            pumpsize = self.pumpsize
            data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
            shear_rate.append(data.transport()['shear_rate'])
            qdot.append(np.mean(data.lambda_IK()['qdot'])*1e24)
            qdot_continuum.append(np.mean(data.lambda_IK()['qdot_continuum'])*1e24)

        ax.plot(shear_rate, qdot)
        ax.plot(shear_rate, qdot_continuum)

        Modify(shear_rate, self.fig, self.axes_array, self.configfile)


    def rate_lambda(self):
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

        shear_rate, lambda_z, lambda_continuum = [], [], []

        for idx, val in enumerate(self.datasets_x):
            pumpsize = self.pumpsize
            data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
            shear_rate.append(data.transport()['shear_rate'])
            lambda_z.append(np.mean(data.lambda_IK()['lambda_z']))
            lambda_continuum.append(np.mean(data.lambda_IK()['lambda_continuum']))

        ax.plot(shear_rate, lambda_z)
        ax.plot(shear_rate, lambda_continuum)

        Modify(shear_rate, self.fig, self.axes_array, self.configfile)


    def thermal_conduct(self):
        """
        Plots the thermal conductivity with Pressure. Equilibrium simulations.
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[7])
        ax.set_ylabel('$\lambda$ (W/mK)')

        # Experimal results are from Wang et al. 2020 (At temp. 335 K)
        exp_lambda = [0.1009,0.1046,0.1113,0.1170]
        exp_press = [5,10,20,30]
        md_lambda, md_press = [], []

        for idx, val in enumerate(self.datasets_x):
            data = dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
            md_lambda.append(np.mean(data.lambda_gk()['lambda_tot']))

        ax.plot(exp_press, md_lambda)
        ax.plot(exp_press, exp_lambda)

        Modify(exp_press, self.fig, self.axes_array, self.configfile)


    def eos(self):
        """
        Plots the equation of state of the fluid
        Equilibrium simualtions can be:
            * Isotherms (NPT) with density as output, plot ρ vs P
                * Gautschi Integtator
                * Velocity Verlet Integrator
            * Isochores (NVT) with pressure as output, plot T vs P
        Nonequilibrium simulations:
            * Change of T with P, to see the effect of the pump ΔP on the fluid temperature
        """
        mpl.rcParams.update({'lines.markersize': 8})

        if self.config['log']:
            # self.axes_array[0].xaxis.major.formatter.set_scientific(False)
            self.axes_array[0].set_xscale('log', base=10)
            self.axes_array[0].set_yscale('log', base=10)
            self.axes_array[0].xaxis.set_minor_formatter(ScalarFormatter())
            self.axes_array[1].set_xscale('log', base=10)
            self.axes_array[1].set_yscale('log', base=10)
            self.axes_array[1].xaxis.set_minor_formatter(ScalarFormatter())
            self.fig.supylabel('$P/P_{o}$')
            self.fig.text(0.3, 0.04, r'$\rho/\rho_{o}$', ha='center', size=14)
            self.fig.text(0.72, 0.04, '$T/T_{o}$', ha='center', size=14)
        else:
            if len(self.axes_array)>1:
                self.fig.supylabel('$P/P_{o}$')
                self.fig.text(0.3, 0.04, r'$\rho/\rho_{o}$', ha='center', size=14)
                self.fig.text(0.72, 0.04, '$T/T_{o}$', ha='center', size=14)
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
            if 'gautschi' in val and '1fs' in val:
                ds_gau_1fs.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'gautschi' in val and '4fs' in val:
                ds_gau_4fs.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'verlet' in val and '1fs' in val:
                ds_ver_1fs.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'verlet' in val and '4fs' in val:
                ds_ver_4fs.append(dataset(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))

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
                print(f'Adiabatic exponent (gamma) is {coeffs_den[0][1]}')
                print(f'Adiabatic exponent (gamma) is {coeffs_temp[0][1]}')
                self.axes_array[0].plot(den, funcs.power(den, coeffs_den[0][0], coeffs_den[0][1], coeffs_den[0][2]))
                self.axes_array[1].plot(temp, funcs.power(temp, coeffs_temp[0][0], coeffs_temp[0][1], coeffs_temp[0][2]))

        # Isotherms -----------------
        for i in ds_isotherms:
            den_isotherms.append(np.mean(i.density()['den_t']))
            press_isotherms.append(np.mean(i.virial()['vir_t']))

        den_isotherms, press_isotherms = np.asarray(den_isotherms), np.asarray(press_isotherms)

        for i in ds_gau_1fs:
            den_isotherms_gau_1fs.append(np.mean(i.density()['den_t']))
            press_isotherms_gau_1fs.append(np.mean(i.virial()['vir_t']))
        for i in ds_gau_4fs:
            den_isotherms_gau_4fs.append(np.mean(i.density()['den_t']))
            press_isotherms_gau_4fs.append(np.mean(i.virial()['vir_t']))
        for i in ds_ver_1fs:
            den_isotherms_ver_1fs.append(np.mean(i.density()['den_t']))
            press_isotherms_ver_1fs.append(np.mean(i.virial()['vir_t']))
        for i in ds_ver_4fs:
            den_isotherms_ver_4fs.append(np.mean(i.density()['den_t']))
            press_isotherms_ver_4fs.append(np.mean(i.virial()['vir_t']))

        if ds_isotherms:
            if self.config['log']:
                # print(np.max(den_isotherms))
                den_isotherms /= 0.72 #np.max(den_isotherms)
                press_isotherms /= 250 #np.max(press_isotherms)

            if ds_gau_1fs: self.axes_array[0].plot(den_isotherms_gau_1fs, press_isotherms_gau_1fs)
            if ds_gau_4fs: self.axes_array[0].plot(den_isotherms_gau_4fs, press_isotherms_gau_4fs)
            if ds_ver_1fs: self.axes_array[0].plot(den_isotherms_ver_1fs, press_isotherms_ver_1fs)
            if ds_ver_4fs: self.axes_array[0].plot(den_isotherms_ver_4fs, press_isotherms_ver_4fs)

            # Experimental data (K. Liu et al. / J. of Supercritical Fluids 55 (2010) 701–711)
            exp_density = [0.630, 0.653, 0.672, 0.686, 0.714, 0.739, 0.750]
            exp_press = [28.9, 55.3, 84.1, 110.2, 171.0, 239.5, 275.5]

            self.axes_array[0].plot(exp_density, exp_press)

            if self.config['log']:
                coeffs_den = curve_fit(funcs.power, den_isotherms, press_isotherms, maxfev=8000)
                print(f'Adiabatic exponent (gamma) is {coeffs_den[0][1]}')
                self.axes_array[0].plot(den_isotherms, funcs.power(den_isotherms, coeffs_den[0][0], coeffs_den[0][1], coeffs_den[0][2]))

        # Isochores -----------------
        for i in ds_isochores:
            temp_list.append(np.mean(i.temp()['temp_t']))
            press_isochores.append(np.mean(i.virial()['vir_t']))
        temp_lst, press_isochores = np.asarray(temp_list), np.asarray(press_isochores)

        if ds_isochores:
            if self.config['log']:
                temp_list /= 300 #np.max(temp_list)
                press_isochores /=  250 #np.max(press_isochores)

            self.axes_array[1].plot(temp_list, press_isochores)

            if self.config['log']:
                coeffs_temp = curve_fit(funcs.power, temp_list, press_isochores, maxfev=8000)
                print(f'Adiabatic exponent (gamma) is {coeffs_temp[0][1]}')
                self.axes_array[1].plot(temp_list, funcs.power(temp_list, coeffs_temp[0][0], coeffs_temp[0][1], coeffs_temp[0][2]))

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

        ax.plot(rho_v,temp)
        ax.plot(rho_l,temp)

        if self.mf ==39.948:
            #Exp.: Lemmon et al.
            rhoc, Tc = 0.5356,150.65 # g/cm3, K
            temp_exp = [85,90,95,100,105,110,115,120,125,130,135]
            rho_v_exp = [0.0051,0.0085,0.0136,0.0187,0.0255,0.0373,0.0441,0.0611,0.0798,0.1052,0.1392]
            rho_l_exp = [1.4102,1.3813,1.3491,1.3152,1.2812,1.2439,1.2032,1.1624,1.1166,1.0674,1.0080]
            ax.plot(rho_v_exp,temp_exp)
            ax.plot(rho_l_exp,temp_exp)

        if self.mf == 72.15:
            # Exp.: B.D. Smith and R. Srivastava, Thermodynamic Data for Pure Compounds: Part A Hydrocarbons and Ketone
            rhoc, Tc = 0.232, 469.70  # g/cm3, K
            temp_exp = [150,175,200,225,250,275,300,325,350,375,400,425]
            rho_v_exp = [0,0,0,0,0,0,0.0006,0.0048,0.0109,0.0182,0.0315,0.0497]
            rho_l_exp = [0.7575,0.7339,0.7115,0.6891,0.6661,0.6431,0.6201,0.5940,0.5662,0.5341,0.5008,0.4554]
            ax.plot(rho_v_exp,temp_exp)
            ax.plot(rho_l_exp,temp_exp)

        # For Density scaling law ---> Get Tc
        rho_diff = np.array(rho_l) - np.array(rho_v)
        popt, pcov = curve_fit(funcs.density_scaling, rho_diff/np.max(rho_diff), temp, maxfev=8000)
        Tc_fit = popt[1]

        ax.plot(rho_l,funcs.density_scaling(rho_diff/np.max(rho_diff), *popt))
        ax.plot(rho_v,funcs.density_scaling(rho_diff/np.max(rho_diff), *popt))

        # For law of rectilinear diameters ---> Get rhoc
        rho_sum = np.array(rho_l) + np.array(rho_v)
        popt2, pcov2 = curve_fit(funcs.rectilinear_diameters, rho_sum, temp, maxfev=8000)
        rhoc_fit=popt2[1]/(popt2[0]*2)
        print(f'Critical Point is:({rhoc_fit,Tc_fit})')

        ax.plot(rhoc_fit, Tc_fit)
        ax.plot(rhoc, Tc)

        Modify(rho_l, self.fig, self.axes_array, self.configfile)


    def rc_gamma(self):
        """
        Plots the cutoff radius on the x-axis and the surface tension (γ) on the y-axis
        To check that the surface tension converges with the cutoff radius
        """
        ax = self.axes_array[0]
        ax.set_xlabel('$r_c/\sigma$')
        ax.set_ylabel('$\gamma$ (mN/m)')

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
        ax.set_ylabel('$\gamma$ (mN/m)')

        temp, gamma = [], []
        for i in range(len(self.datasets_x)):
            ds = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            temp.append(np.mean(ds.temp()['temp_t']))
            gamma.append(np.mean(ds.surface_tension()['gamma'])*1e3)

        if self.mf == 39.948: gamma_exp = [11.871,10.661,9.7482,8.3243,7.2016,6.1129,5.0617,4.0527,3.0919,2.2879]
        if self.mf == 72.15: gamma_exp = [32.5,29.562,26.2638,23.736,20.864,18.032,15.250,12.534,9.9014,7.379,5.0029]#,2.8318,0.98]

        popt, pcov = curve_fit(funcs.power_new, temp/np.max(temp), gamma, maxfev=8000)

        print(f'Critical temperature: Tc = {popt[1]*np.max(temp):.2f}')

        ax.plot(temp,gamma)
        ax.plot(temp,gamma_exp)
        ax.plot(temp,funcs.power_new(temp/np.max(temp), *popt))

        Modify(temp, self.fig, self.axes_array, self.configfile)


    def struc_factor(self):
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
        sf = data.sf()['sf']
        # sf_solid = data.sf()['sf_solid']

        if self.config['3d']:
            self.ax.set_xlabel('$k_x (\AA^{-1})$')
            self.ax.set_ylabel('$k_y (\AA^{-1})$')
            self.ax.set_zlabel('$S(k)$')
            self.ax.invert_xaxis()
            self.ax.set_ylim(ky[-1]+1,0)
            self.ax.zaxis.set_rotate_label(False)
            self.ax.set_zticks([])

            Kx, Ky = np.meshgrid(kx, ky)
            self.ax.plot_surface(Kx, Ky, sf.T, cmap=mpl.cm.jet,
                        rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
            # self.fig.colorbar(surf, shrink=0.5, aspect=5)

            self.ax.view_init(90,0) # (35,60)
            # pickle.dump(self.fig, open('FigureObject.fig.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

        else:
            a = input('x or y or r or t:')
            ax = self.axes_array[0]
            if a=='x':
                ax.set_xlabel('$k_x (\AA^{-1})$')
                ax.set_ylabel('$S(K_x)$')
                ax.plot(kx, sfx, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
                Modify(kx, self.fig, self.axes_array, self.configfile)
            elif a=='y':
                ax.set_xlabel('$k_y (\AA^{-1})$')
                ax.set_ylabel('$S(K_y)$')
                ax.plot(ky, sfy, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
                Modify(ky, self.fig, self.axes_array, self.configfile)
            elif a=='t':
                ax.set_xlabel('$t (fs)$')
                ax.set_ylabel('$S(K)$')
                ax.plot(self.time[self.skip:], sf_time, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
                Modify(self.time[self.skip:], self.fig, self.axes_array, self.configfile)
            # elif a=='r':
            #     ax.set_xlabel('$k (\AA^{-1})$')
            #     ax.set_ylabel('$S(K)$')
            #     ax.plot(k, sf_r, ls= ' ', marker='x', alpha=opacity,
            #                label=input('Label:'))


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

    def acf(self):
        """
        Plots the auto correlation function (till now: only the mass flux is implemented)
        # TODO: Check that this works properly!
        """
        ax = self.axes_array[0]
        ax.set_xlabel(labels[2])
        ax.set_ylabel(r'${\mathrm C_{AA}}$')
        mpl.rcParams.update({'lines.linewidth':'1'})

        for i in range(len(self.datasets_x)):
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            acf_flux = sq.acf(data.mflux()['jx_t'])['norm']
            ax.plot(self.time[:10000]*1e-6, acf_flux[:10000])

        ax.axhline(y= 0, color='k', linestyle='dashed', lw=1)

        Modify(self.time[:10000]*1e-6, self.fig, self.axes_array, self.configfile)


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
