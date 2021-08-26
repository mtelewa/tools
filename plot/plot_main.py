#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4
import re
import numpy as np
import sys
import os
from get_variables import derive_data as dd
# import get_variables_210810 as get_variables
import funcs
import sample_quality as sq
import label_lines
from cycler import cycler

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.image as image
import matplotlib.cm as cmx

from scipy.stats import iqr
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline


SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

default_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )

color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']


plt.style.use('imtek')
mpl.rcParams.update({'lines.markersize': 3})

#           0             1                   2                    3
labels=('Height (nm)','Length (nm)', 'Time (ns)', r'Density (g/${\rm cm^3}$)',
#           4                                    5             6                    7
        r'${\rm j_x}$ (g/${\rm m^2}$.ns)', 'Vx (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\rm(F_x)}$ (pN)', r'${\rm \partial p / \partial x}$ (MPa/nm)',
#           10                                          11
        r'${\rm \dot{m}} \times 10^{-18}}$ (g/ns)', r'${\rm \dot{\gamma} (s^{-1})}$',
#           12
        r'${\rm \mu}$ (mPa.s)')


datasets_x, datasets_z, txtfiles = [], [], []
skip = np.int(sys.argv[1])

for i in sys.argv:
    if i.endswith('1.nc'):
        datasets_x.append(i)
    if i.endswith('4.nc'):
        datasets_z.append(i)
    if i.endswith('.txt'):
        txtfiles.append(i)


class plot_from_txt:

    def plot_from_txt(self,  outfile, lt='-', mark='o', opacity=1.0):

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

        # fig.tight_layout()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.set_xlabel(labels[np.int(sys.argv[2])])
        ax.set_ylabel(labels[np.int(sys.argv[3])])

        for i in range(len(txtfiles)):

            data = np.loadtxt(txtfiles[i], skiprows=2, dtype=float)

            xdata = data[:,0]
            ydata = data[:,1]
            # err = data[:,2]

            # ax.set_xscale('log', nonpositive='clip')

            # If the dimension is time, scale by 1e-6
            if np.int(sys.argv[2]) == 2:
                xdata = xdata * 1e-6

            ax.plot(xdata, ydata, ls=' ', marker='o', alpha=1, label=input('Label:'),)

            # popt, pcov = curve_fit(funcs.linear, xdata, ydata)
            # ax.plot(xdata, funcs.linear(xdata, *popt))
            # ax.errorbar(xdata, ydata1 , yerr=err1, ls=lt, fmt=mark, label= 'Expt. (Gehrig et al. 1978)', capsize=3, alpha=opacity)

        ax.legend()
        fig.savefig(outfile)



class plot_from_ds:

    def __init__(self):

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')

        self.Nx = len(dd(datasets_x[0], datasets_z[0], skip).length_array)
        self.Nz = len(dd(datasets_x[0], datasets_z[0], skip).height_array)
        self.Nz_mod = len(dd(datasets_x[0], datasets_z[0], skip).velocity()['height_array_mod'])
        self.time = dd(datasets_x[0], datasets_z[0], skip).time
        self.Ly = dd(datasets_x[0], datasets_z[0], skip).Ly

        self.length_arrays = np.zeros([len(datasets_x), self.Nx])
        self.height_arrays = np.zeros([len(datasets_z), self.Nz])
        self.height_arrays_mod = np.zeros([len(datasets_z), self.Nz_mod])
        for i in range(len(datasets_x)):
            self.length_arrays[i, :] = dd(datasets_x[i], datasets_z[i], skip).length_array
        for i in range(len(datasets_z)):
            self.height_arrays[i, :] = dd(datasets_x[i], datasets_z[i], skip).height_array
            self.height_arrays_mod[i, :] = dd(datasets_x[i], datasets_z[i], skip).velocity()['height_array_mod']

        # self.sigxz_t = np.zeros([len(datasets), len(self.time)])
        # self.avg_sigxz_t = np.zeros([len(datasets)])
        self.pGrad = np.zeros([len(datasets_x)])
        self.viscosities = np.zeros([len(datasets_z)])

        pump_length = 0.2 * np.max(self.length_arrays)
        # smoothed_pump_length = pump_length * 15/8

    def vx_height(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[5])

        mpl.rcParams.update({'lines.markersize': 3})

        vx_chunkZ = np.zeros([len(datasets_z), self.Nz_mod])

        for i in range(len(datasets_z)):
            vx_chunkZ[i, :] = dd(datasets_x[i], datasets_z[i], skip).velocity()['vx_data']
            label=input('Label:')
            self.ax.plot(self.height_arrays_mod[i, :], vx_chunkZ[i, :],
                                 ls=' ', label=label, marker='o', alpha=opacity)

            # Plot the fits
            if 'fit' in sys.argv:
                # popt, pcov = curve_fit(funcs.quadratic, xdata, ydata)
                #     ax.plot(xdata, ydata, *popt))
                x1 = dd(datasets_x[i], datasets_z[i], skip).velocity()['xdata']
                y1 = dd(datasets_x[i], datasets_z[i], skip).velocity()['fit_data']

                self.ax.plot(x1, y1)

            if 'hydro' in sys.argv:
                v_hydro = dd(datasets_x[i], datasets_z[i], skip).hydrodynamic()['v_hydro']

                self.ax.plot(self.height_arrays_mod[i, :], v_hydro,
                                     ls='-', label=label, marker='x', alpha=opacity)

            # Plot the extrapolated lines
            if 'extrapolate' in sys.argv:
                x_left = dd(datasets_x[i], datasets_z[i], skip).slip_length()['xdata_left']
                y_left = dd(datasets_x[i], datasets_z[i], skip).slip_length()['extrapolate_left']

                x_right = dd(datasets_x[i], datasets_z[i], skip).slip_length()['xdata_right']
                y_right = dd(datasets_x[i], datasets_z[i], skip).slip_length()['extrapolate_right']

                self.ax.set_xlim([dd(datasets_x[i], datasets_z[i], skip).slip_length()['root_left'],
                                  dd(datasets_x[i], datasets_z[i], skip).slip_length()['root_right']])

                self.ax.set_ylim(bottom = 0)

                self.ax.plot(x_left, y_left, color='sienna')
                self.ax.plot(x_right, y_right, color='sienna')

            # plot vertical lines for the walls
            if 'walls' in sys.argv:
                if len(datasets_z) == 1:
                    self.ax.axvline(x= self.height_arrays_mod[0][0], color='k',
                                                        linestyle='dashed', lw=1)
                    self.ax.axvline(x= self.height_arrays_mod[0][-1], color= 'k',
                                                        linestyle='dashed', lw=1)
                else:
                    for i in range(len(datasets_z)):
                        ax.axvline(x= self.height_arrays[i][-1], color= line_colors[i],
                                                        linestyle='dashed', lw=1)

            if 'inset' in sys.argv:
                popt2, pcov2 = curve_fit(funcs.quadratic, self.height_arrays_mod[0][1:-1], vx_chunkZ[0, :][1:-1])

                inset_ax = fig.add_axes([0.6, 0.48, 0.2, 0.28]) # X, Y, width, height
                inset_ax.plot(height_arrays_mod[0][-31:-1], vx_chunkZ[0, :][-31:-1],
                                        ls= ' ', marker=mark, alpha=opacity, label=label)
                inset_ax.plot(height_arrays_mod[0][-31:-1], funcs.quadratic(height_arrays_mod[0], *popt2)[-31:-1])


            # print('Velocity at the wall %g m/s at a distance %g nm from the wall' %(vx_wall,z_wall))
            # line_colors.append(ax.lines[j].get_color())

    def vel_distrib(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel('Velocity (m/s)')
        self.ax.set_ylabel('Probability')

        for i in range(len(datasets_x)):
            vx_values = dd(datasets_x[i], datasets_z[i], skip).vel_distrib()['vx_values']
            vx_prob = dd(datasets_x[i], datasets_z[i], skip).vel_distrib()['vx_prob']

            vy_values = dd(datasets_x[i], datasets_z[i], skip).vel_distrib()['vy_values']
            vy_prob = dd(datasets_x[i], datasets_z[i], skip).vel_distrib()['vy_prob']

            self.ax.plot(vx_values, vx_prob, ls=' ', marker='o',label=input('Label:'), alpha=opacity)
            self.ax.plot(vy_values, vy_prob, ls=' ', marker='x',label=input('Label:'), alpha=opacity)


    def mflowrate(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[10])

        if 'mflowrate_length' in sys.argv:
            self.ax.set_xlabel(labels[1])
            mflowrate = np.zeros([len(datasets_x), self.Nx])

            for i in range(len(datasets_x)):
                mflowrate[i, :] = dd(datasets_x[i], datasets_z[i], skip).mflux()['mflowrate_stable']
                self.ax.plot(self.length_arrays[i, :], mflowrate[i, :], ls=lt, marker='o',label=input('Label:'), alpha=opacity)

        if 'mflowrate_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            mflowrate_t = np.zeros([len(datasets_x), len(self.time)])
            mflowrate_avg = np.zeros([len(datasets_x)])

            for i in range(len(datasets_x)):
                mflowrate_t[i, :] = dd(datasets_x[i], datasets_z[i], skip).mflux()['mflowrate_stable']
                mflowrate_avg[i] = np.mean(mflowrate_t[i])
                self.ax.plot(self.time*1e-6,  mflowrate_t[i, :]*1e18, ls='-', marker=' ',label=input('Label:'), alpha=0.5)
                print(self.ax.lines[i].get_color())

            for i in range (len(self.ax.lines)):
                self.ax.axhline(y=mflowrate_avg[i]*1e18, color= self.ax.lines[i].get_color(), linestyle='dashed', lw=1)


    def mflux(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[4])

        if 'jx_length' in sys.argv:
            self.ax.set_xlabel(labels[1])
            jx = np.zeros([len(datasets_x), self.Nx])

            for i in range(len(datasets_x)):
                jx[i, :] = dd(datasets_x[i], datasets_z[i], skip).mflux()[0]
                self.ax.plot(self.length_arrays[i, :], jx[i, :], ls=lt, marker='o',label=input('Label:'), alpha=opacity)

        if 'jx_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            jx_t = np.zeros([len(datasets_x), self.time])
            jx_avg = np.zeros([len(datasets_x)])

            for i in range(len(datasets_x)):
                jx_t[i, :] = dd(datasets_x[i], datasets_z[i], skip).mflux()['']
                jx_avg[i] = np.mean(jx_t[i])
                self.ax.plot(self.time*1e-6,  jx_t[i, :], ls='-', marker=' ',label=input('Label:'), alpha=0.5)

            for i in range (len(self.ax.lines)):
                self.ax.axhline(y=jx_avg[i]*1e18, color= self.ax.lines[i].get_color(), linestyle='dashed', lw=1)

    def density(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[3])

        if 'den_length' in sys.argv:
            denX = np.zeros([len(datasets_x), self.Nx])
            self.ax.set_xlabel(labels[1])
            for i in range(len(datasets_x)):
                denX[i, :] = dd(datasets_x[i], datasets_z[i], skip).density()['den_chunkX']
                self.ax.plot(self.length_arrays[i, :][1:-1], denX[i, :][1:-1], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'den_height' in sys.argv:
            denZ = np.zeros([len(datasets_z), self.Nz])
            self.ax.set_xlabel(labels[0])
            for i in range(len(datasets_z)):
                denZ[i, :] = dd(datasets_x[i], datasets_z[i], skip).density()['den_chunkZ']
                self.ax.plot(self.height_arrays[i, :], denZ[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'den_time' in sys.argv:
            denT = np.zeros([len(datasets_x), self.time])
            bulk_den_avg = np.zeros_like(gap_height_avg)
            self.ax.set_xlabel(labels[2])
            for i in range(len(datasets_x)):
                denT[i, :] = dd(datasets_x[i], datasets_z[i], skip).density()['den_t']
                bulk_den_avg[i] = np.mean(denT[i])
                self.ax.plot(self.time*1e-6, denT[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)


    def press(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[7])

        if 'press_length' in sys.argv:
            self.ax.set_xlabel(labels[1])
            sigzz_chunkX = np.zeros([len(datasets_x), self.Nx])
            vir_chunkX = np.zeros([len(datasets_x), self.Nx])

            for i in range(len(datasets_x)):
                sigzz_chunkX[i, :] = dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigzz_chunkX']
                vir_chunkX[i, :] = dd(datasets_x[i], datasets_z[i], skip).virial()['vir_chunkX']

            if 'virial' in sys.argv:
                for i in range(len(datasets_x)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], vir_chunkX[i, :][1:-1],
                                ls=lt, marker=None, label=input('Label:'), alpha=opacity)

            if 'sigwall' in sys.argv:
                for i in range(len(datasets_x)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], sigzz_chunkX[i, :][1:-1],
                                ls=lt, marker='o', label=input('Label:'), alpha=opacity)
            if 'both' in sys.argv:
                for i in range(len(datasets_x)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], vir_chunkX[i, :][1:-1],
                                ls=lt, marker='o', label=input('Label:'), alpha=opacity)
                for i in range(len(datasets_x)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], sigzz_chunkX[i, :][1:-1],
                                ls='-', marker='x', label=input('Label:'), alpha=opacity)

            if 'inset' in sys.argv:
                inset_ax = fig.add_axes([0.62, 0.57, 0.2, 0.28]) # X, Y, width, height
                inset_ax.axvline(x=0, color='k', linestyle='dashed')
                inset_ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
                inset_ax.set_ylim(220, 280)
                inset_ax.plot(self.length_arrays[0][1:29], vir_chunkX[0, :][1:29] , ls= lt, color=ax.lines[0].get_color(), marker=None, alpha=opacity, label=label)
                inset_ax.plot(self.length_arrays[0][1:29], vir_chunkX[1, :][1:29] , ls= ' ', color=ax.lines[1].get_color(), marker='x', alpha=opacity, label=label)

            if 'sigxz' in sys.argv:
                self.ax.set_xlabel(labels[1])
                self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                sigxz_chunkX = np.zeros([len(datasets_x), self.Nx])

                for i in range(len(datasets_x)):
                    sigxz_chunkX[i, :] =  dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigxz_chunkX']
                    self.ax.plot(self.length_arrays[i, :][1:-1],  sigxz_chunkX[i, :][1:-1],
                                ls='-', marker=' ',label=input('Label:'), alpha=0.5)

        if 'press_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            sigzz_t = np.zeros([len(datasets_x), len(self.time)])
            vir_t = np.zeros([len(datasets_x), len(self.time)])

            for i in range(len(datasets_x)):
                sigzz_t[i, :] = dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigzz_t']
                vir_t[i, :] = dd(datasets_x[i], datasets_z[i], skip).virial()['vir_t']

            if 'virial' in sys.argv:
                for i in range(len(datasets_x)):
                    self.ax.plot(self.time[:]*1e-6, vir_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=1)
            if 'sigwall' in sys.argv:
                for i in range(len(datasets_x)):
                    self.ax.plot(self.time[:]*1e-6, sigzz_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=1)
            if 'both' in sys.argv:
                for i in range(len(datasets_x)):
                    self.ax.plot(self.time[:]*1e-6,  vir_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=0.5)
                for i in range(len(datasets_x)):
                    self.ax.plot(self.time[:]*1e-6,  sigzz_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=0.5)
            if 'sigxz' in sys.argv:
                self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                sigxz_t = np.zeros([len(datasets_x), len(self.time)])
                avg_sigxz = np.zeros(len(datasets_x))

                for i in range(len(datasets_x)):
                    sigxz_t[i, :] =  dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigxz_t']
                    avg_sigxz[i] = np.mean(sigxz_t[i, :])
                    self.ax.plot(self.time*1e-6, sigxz_t[i, :], ls='-', marker=' ',label=input('Label:'), alpha=0.5)
                    self.ax.axhline(y= avg_sigxz[i], color=color[i], linestyle='dashed')


    def temp(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[6])

        if 'temp_length' in sys.argv:
            tempX = np.zeros([len(datasets_x), self.Nx])
            self.ax.set_xlabel(labels[1])
            for i in range(len(datasets_x)):
                tempX[i, :] = dd(datasets_x[i], datasets_z[i], skip).temp()[0]
                self.ax.plot(self.length_arrays[i, :], tempX[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'temp_height' in sys.argv:
            tempZ = np.zeros([len(datasets_z), self.Nz])
            self.ax.set_xlabel(labels[0])
            for i in range(len(datasets_z)):
                tempZ[i, :] = dd(datasets_x[i], datasets_z[i], skip).temp()[1]
                self.ax.plot(self.height_arrays[i, :], tempZ[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'temp_time' in sys.argv:
            tempT = np.zeros([len(datasets_x), len(self.time)])
            self.ax.set_xlabel(labels[2])
            for i in range(len(datasets_x)):
                tempT[i, :] = dd(datasets_x[i], datasets_z[i], skip).temp()[2]
                self.ax.plot(self.time[:]*1e-6, tempT[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

    def pgrad_mflowrate(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[10])

        mpl.rcParams.update({'lines.markersize': 4})
        mpl.rcParams.update({'figure.figsize': (12,12)})

        mflowrate_avg = np.zeros([len(datasets_x)])
        shear_rates = np.zeros([len(datasets_x)])

        for i in range(len(datasets_x)):
            self.pGrad[i] = dd(datasets_x[i], datasets_z[i], skip).virial()[2]
            mflowrate_avg[i] = np.mean(dd(datasets_x[i], datasets_z[i], skip).mflux()[3])
            shear_rates[i] = dd(datasets_x[i], datasets_z[i], skip).shear_rate()[0]
            self.viscosities[i] = dd(datasets_x[i], datasets_z[i], skip).viscosity()     # mPa.s
            bulk_den_avg[i] = np.mean(dd(datasets_x[i], datasets_z[i], skip).density()[2])

        mflowrate_hp =  ((bulk_den_avg*1e3) * (self.Ly*1e-10) * 1e3 * 1e-9 / (12 * (mu*1e6))) \
                                    * pGrad*1e6*1e9 * (avg_gap_height*1e-10)**3

        # ax.set_xscale('log', nonpositive='clip')
        # ax.set_yscale('log', nonpositive='clip')
        self.ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        self.ax.plot(self.pGrad, mflowrate_hp*1e18, ls='--', marker='o', alpha=opacity, label=input('Label:'))
        self.ax.plot(self.pGrad, mflowrate_avg*1e18, ls=lt, marker='o', alpha=opacity, label=input('Label:'))

        ax2 = self.ax.twiny()
        ax2.set_xscale('log', nonpositive='clip')
        ax2.plot(shear_rates, mflowrate_avg, ls= ' ', marker= ' ',
                    alpha=opacity, label=label, color=color[0])
        ax2.set_xlabel(labels[11])

        # err=[]
        # for i in range(len(datasets)):
        #     err.append(sq.get_err(get_variables.get_data(datasets[i], skip)[17])[2])
        #
        # ax.errorbar(pGrad, mflux_avg,yerr=err,ls=lt,fmt=mark,capsize=3,
        #            alpha=opacity)
        # ax.set_ylim(bottom=1e-4)

    def pgrad_viscosity(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        mpl.rcParams.update({'lines.markersize': 4})

        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[12])

        for i in range(len(datasets_x)):
            self.pGrad[i] = dd(datasets_x[i], datasets_z[i], skip).virial()[2]
            self.viscosities[i] = dd(datasets_x[i], datasets_z[i], skip).viscosity()

        self.ax.plot(self.pGrad, self.viscosities, ls=lt, marker='o', alpha=opacity)

    def pgrad_slip(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel('Slip Length b (nm)')

        slip = np.zeros([len(datasets_x)])

        for i in range(len(datasets_x)):
            self.pGrad[i] = dd(datasets_x[i], datasets_z[i], skip).virial()[2]
            slip[i] = dd(datasets_x[i], datasets_z[i], skip).slip_length()[0]

        self.ax.plot(self.pGrad, slip, ls=lt, marker='o', alpha=opacity)


    def height_time(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[2])
        self.ax.set_ylabel(labels[0])

        gap_height = np.zeros([len(datasets_x), len(self.time)])
        gap_height_avg = np.zeros([len(datasets_x)])

        for i in range(len(datasets_x)):
            gap_height[i, :] = dd(datasets_x[i], datasets_z[i], skip).h
            gap_height_avg[i] = dd(datasets_x[i], datasets_z[i], skip).avg_gap_height

            self.ax.plot(self.time*1e-6, gap_height[i, :], ls='-', marker=' ', alpha=opacity, label=input('Label:'))
            self.ax.axhline(y= gap_height_avg, color= 'k', linestyle='dashed', lw=1)

    def weight(self, label=None, err=None, lt='-', mark='o', opacity=1.0):
        self.ax.set_xlabel(labels[1])
        self.ax.set_ylabel('Weight')

        length_padded = np.pad(self.length_arrays[0], (1,0), 'constant')

        self.ax.plot(length_padded, funcs.quartic(length_padded), ls=lt, marker=None, alpha=opacity)
        self.ax.plot(length_padded, funcs.step(length_padded), ls='--', marker=None, alpha=opacity)


if __name__ == "__main__":

    if len(txtfiles) > 0.:
        ptxt = plot_from_txt()
        ptxt.plot_from_txt(sys.argv[-1])

    if len(datasets_x) > 0.:
        pds = plot_from_ds()

        if 'vx_height' in sys.argv:
            pds.vx_height()

        if 'vel_distrib' in sys.argv:
            pds.vel_distrib()

        if 'mflowrate_length' in sys.argv or 'mflowrate_time' in sys.argv:
            pds.mflowrate()

        if 'jx_length' in sys.argv or 'jx_time' in sys.argv:
            pds.mflux()

        if 'den_length' in sys.argv or 'den_height' in sys.argv or 'den_time' in sys.argv:
            pds.density()

        if 'press_length' in sys.argv or 'press_time' in sys.argv:
            pds.press()

        if 'temp_length' in sys.argv or 'temp_height' in sys.argv or 'temp_time' in sys.argv:
            pds.temp()

        if 'pgrad_mflowrate' in sys.argv:
            pds.pgrad_mflowrate()

        if 'pgrad_viscosity' in sys.argv:
            pds.pgrad_viscosity()

        if 'pgrad_slip' in sys.argv:
            pds.pgrad_slip()

        if 'height_time' in sys.argv:
            pds.height_time()

        if 'legend' in sys.argv:
            pds.ax.legend()

        if 'title' in sys.argv:
            pds.ax.set_title('title')

        if 'label-lines' in sys.argv:
            for i in range (len(self.ax.lines)):
                xpos = np.float(input('Label x-pos:'))
                y_offset = np.float(input('Y-offset for label: '))
                rot = np.float(input('Rotation of label: '))
                lab = input('Label:')

                label_lines.label_line(pds.ax.lines[i], xpos, yoffset= y_offset, \
                         label= lab, fontsize= 8, rotation= rot)

        if 'draw-vlines' in sys.argv:
            pds.ax.axvline(x= 0, color='k', linestyle='dashed', lw=1)
            pds.ax.axvline(x= heights[1][-1], color= self.ax.lines[2].get_color(), linestyle='dashed', lw=1)
            pds.ax.axvline(x= heights[2][-1], color= self.ax.lines[4].get_color(), linestyle='dashed', lw=1)
            pds.ax.axvline(x= 0.2*np.max(lengths), color='k', linestyle='dashed')

        # Adjust ticks
        # plt.yticks(np.arange(30, 80, step=10))  # Set label locations.

        pds.ax.set_rasterized(True)
        pds.fig.savefig(sys.argv[-1]+'.png' , format='png')


    # # Name the output files according to the input
    # # base = os.path.basename(infile)
    # # global filename
    # # filename = os.path.splitext(base)[0]
    # # figure(num=None, dpi=80, facecolor='w', edgecolor='k')

    # if 'press-evol' in sys.argv:
    #     ax.set_xlabel(labels[1])
    #     ax.set_ylabel(labels[-1])
    #
    #     # for i in range(0,50):
    #     #     plt.plot((-0.1, 0.6),(i, i), color=(i/50., 0, (50-i)/50.))
    #
    #     nChunks = len(sigzz_chunkXi)
    #     nChunks_range = range(nChunks)
    #
    #     # map = plt.get_cmap('coolwarm', 256)
    #     # cNorm  = colors.Normalize(vmin=0, vmax=nChunks_range[-1])
    #     # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=map)
    #
    #     start = 0.0
    #     stop = 1.0
    #     number_of_lines= nChunks
    #     cm_subsection = np.linspace(start, stop, number_of_lines)
    #
    #     colors = [ cmx.coolwarm(x) for x in cm_subsection ]
    #
    #     for i in range(nChunks):
    #         ax.plot(lengths[i][1:-1], vir_chunkXi[i][1:-1], ls=lt, \
    #             color=colors[i] , marker=None)
    #
    #     im = image.AxesImage(ax, cmap='coolwarm')
    #
