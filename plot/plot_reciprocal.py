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
from compute_reciprocal import ExtractFromTraj as dataset
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

    def __init__(self, skip, datasets, mf, configfile, pumpsize):

        self.skip = skip
        self.mf = mf
        self.datasets = datasets
        self.configfile = configfile
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)
        self.pumpsize = pumpsize
        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())

        # try:
        first_dataset = dataset(self.skip, self.datasets[0], self.mf, self.pumpsize)
        self.time = first_dataset.time


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


    def colorFader(self, c1, c2, mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


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


    def struc_factor(self, fluid):
        """
        Plots the structure factor on the x-axis and the wave length (kx or ky) on the y-axis in 2D figure
        or both (kx and ky) on the y- and z- axes in a 3D figure
        """
        data = dataset(self.skip, self.datasets[0], self.mf, self.pumpsize)

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
            # self.ax.set_ylim(bottom=0, top=ky.max())
            # self.ax.set_xlim(left=0, right=kx.max())
            self.ax.set_xlabel(r'$k_x \, \mathrm{(\AA^{-1})}$')
            self.ax.set_ylabel(r'$k_y \, \mathrm{(\AA^{-1})}$')
            Kx, Ky = np.meshgrid(kx, ky)

            plt.imshow(sf.T, cmap='viridis', interpolation='lanczos',
                extent=[kx.min(), kx.max(), ky.min(), ky.max()], aspect='auto', origin='lower')
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
