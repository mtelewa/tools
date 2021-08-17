#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4
import re
import numpy as np
import sys
import os
import get_variables
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


datasets, txtfiles = [], []
skip = np.int(sys.argv[1])

for i in sys.argv:
    if i.endswith('.nc'):
        datasets.append(i)
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

        self.Nx = len(get_variables.derive_data(datasets[0], skip).length_array)
        self.Nz = len(get_variables.derive_data(datasets[0], skip).height_array)
        self.time = get_variables.derive_data(datasets[0], skip).time
        self.Ly = get_variables.derive_data(datasets[0], skip).Ly

        self.length_arrays = np.zeros([len(datasets), self.Nx])
        self.height_arrays = np.zeros([len(datasets), self.Nz])

        for i in range(len(datasets)):
            self.length_arrays[i, :] = get_variables.derive_data(datasets[i], skip).length_array
            self.height_arrays[i, :] = get_variables.derive_data(datasets[i], skip).height_array

        self.sigxz_t = np.zeros([len(datasets), len(self.time)])
        self.avg_sigxz_t = np.zeros([len(datasets)])
        self.pGrad = np.zeros([len(datasets)])
        self.viscosities = np.zeros([len(datasets)])

        pump_length = 0.2 * np.max(self.length_arrays)
        smoothed_pump_length = pump_length * 15/8

    def vx_height(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[5])

        mpl.rcParams.update({'lines.markersize': 3})

        Nz_mod = len(get_variables.derive_data(datasets[0], skip).velocity()[0])
        height_arrays_mod = np.zeros([len(datasets), Nz_mod])
        vx_chunkZ = np.zeros([len(datasets), Nz_mod])

        for i in range(len(datasets)):
            height_arrays_mod[i, :] = get_variables.derive_data(datasets[i], skip).velocity()[0]
            vx_chunkZ[i, :] = get_variables.derive_data(datasets[i], skip).velocity()[1]
            label=input('Label:')
            self.ax.plot(height_arrays_mod[i, :], vx_chunkZ[i, :],
                                 ls=' ', label=label, marker='o', alpha=opacity)

            # Plot the fits
            if 'fit' in sys.argv:
                # popt, pcov = curve_fit(funcs.quadratic, xdata, ydata)
                #     ax.plot(xdata, ydata, *popt))
                x1 = get_variables.derive_data(datasets[i], skip).velocity()[2]
                y1 = get_variables.derive_data(datasets[i], skip).velocity()[3]

                self.ax.plot(x1, y1)

            # Plot the extrapolated lines
            if 'extrapolate' in sys.argv:
                x2 = get_variables.derive_data(datasets[i], skip).slip_length()[2]
                y2 = get_variables.derive_data(datasets[i], skip).slip_length()[3]

                self.ax.set_ylim(bottom = 0)
                self.ax.plot(x2, y2, color='sienna')

            if 'inset' in sys.argv:
                popt2, pcov2 = curve_fit(funcs.quadratic, height_arrays_mod[0][1:-1], vx_chunkZ[0, :][1:-1])

                inset_ax = fig.add_axes([0.6, 0.48, 0.2, 0.28]) # X, Y, width, height
                inset_ax.plot(height_arrays_mod[0][-31:-1], vx_chunkZ[0, :][-31:-1] , ls= ' ', marker=mark,
                            alpha=opacity, label=label)
                inset_ax.plot(height_arrays_mod[0][-31:-1], funcs.quadratic(height_arrays_mod[0], *popt2)[-31:-1])

            # plot vertical lines for the walls
            if 'walls' in sys.argv:
                ax.axvline(x=0, color='k', linestyle='dashed', lw=1)
                if len(datasets) == 1:
                    ax.axvline(x= heights[0][-1], color= 'k', linestyle='dashed', lw=1)
                else:
                    for i in range(len(datasets)):
                        ax.axvline(x= heights[i][-1], color= line_colors[i], linestyle='dashed', lw=1)

            # print('Velocity at the wall %g m/s at a distance %g nm from the wall' %(vx_wall,z_wall))
            # line_colors.append(ax.lines[j].get_color())

    def vx_distrib(self, label=None, err=None, lt='-', mark='o', opacity=1.0):
        self.ax.set_xlabel(labels[5])
        self.ax.set_ylabel('Probability')

        for i in range(len(datasets)):
            values = get_variables.derive_data(datasets[i], skip).vx_distrib()[0]
            probabilities = get_variables.derive_data(datasets[i], skip).vx_distrib()[1]

            self.ax.plot(values, probabilities, ls=' ', marker='o',label=input('Label:'), alpha=opacity)

    def mflowrate(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[10])

        if 'mflowrate_length' in sys.argv:
            self.ax.set_xlabel(labels[1])
            mflowrate = np.zeros([len(datasets), self.Nx])

            for i in range(len(datasets)):
                mflowrate[i, :] = get_variables.derive_data(datasets[i], skip).mflux()[3]
                self.ax.plot(self.length_arrays[i, :], mflowrate[i, :], ls=lt, marker='o',label=input('Label:'), alpha=opacity)

        if 'mflowrate_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            mflowrate_t = np.zeros([len(datasets), len(self.time)])
            mflowrate_avg = np.zeros([len(datasets)])

            for i in range(len(datasets)):
                mflowrate_t[i, :] = get_variables.derive_data(datasets[i], skip).mflux()[3]
                mflowrate_avg[i] = np.mean(mflowrate_t[i])
                self.ax.plot(self.time*1e-6,  mflowrate_t[i, :], ls='-', marker=' ',label=input('Label:'), alpha=0.5)


    def mflux(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[4])

        if 'jx_length' in sys.argv:
            self.ax.set_xlabel(labels[1])
            jx = np.zeros([len(datasets), self.Nx])

            for i in range(len(datasets)):
                jx[i, :] = get_variables.derive_data(datasets[i], skip).mflux()[0]
                self.ax.plot(self.length_arrays[i, :], jx[i, :], ls=lt, marker='o',label=input('Label:'), alpha=opacity)

        if 'jx_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            jx_t = np.zeros([len(datasets), self.time])
            jx_avg = np.zeros([len(datasets)])

            for i in range(len(datasets)):
                jx_t[i, :] = get_variables.derive_data(datasets[i], skip).mflux()[2]
                jx_avg[i] = np.mean(jx_t[i])
                self.ax.plot(self.time*1e-6,  jx_t[i, :], ls='-', marker=' ',label=input('Label:'), alpha=0.5)


    def density(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[3])

        if 'den_length' in sys.argv:
            denX = np.zeros([len(datasets), self.Nx])
            self.ax.set_xlabel(labels[1])
            for i in range(len(datasets)):
                denX[i, :] = get_variables.derive_data(datasets[i], skip).density()[0]
                self.ax.plot(self.length_arrays[i, :][1:-1], denX[i, :][1:-1], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'den_height' in sys.argv:
            denZ = np.zeros([len(datasets), self.Nz])
            self.ax.set_xlabel(labels[0])
            for i in range(len(datasets)):
                denZ[i, :] = get_variables.derive_data(datasets[i], skip).density()[1]
                self.ax.plot(self.height_arrays[i, :], denZ[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'den_time' in sys.argv:
            denT = np.zeros([len(datasets), self.time])
            bulk_den_avg = np.zeros_like(gap_height_avg)
            self.ax.set_xlabel(labels[2])
            for i in range(len(datasets)):
                denT[i, :] = get_variables.derive_data(datasets[i], skip).density()[2]
                bulk_den_avg[i] = np.mean(denT[i])
                self.ax.plot(self.time*1e-6, denT[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)


    def press(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[7])

        if 'press_length' in sys.argv:
            self.ax.set_xlabel(labels[1])
            sigzz_chunkX = np.zeros([len(datasets), self.Nx])
            vir_chunkX = np.zeros([len(datasets), self.Nx])

            for i in range(len(datasets)):
                sigzz_chunkX[i, :] = get_variables.derive_data(datasets[i], skip).sigwall()[1]
                vir_chunkX[i, :] = get_variables.derive_data(datasets[i], skip).virial()[0]

            if 'virial' in sys.argv:
                for i in range(len(datasets)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], vir_chunkX[i, :][1:-1],
                                ls=lt, marker=None, label=input('Label:'), alpha=opacity)
                    print(vir_chunkX)

            if 'sigwall' in sys.argv:
                for i in range(len(datasets)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], sigzz_chunkX[i, :][1:-1],
                                ls=lt, marker='o', label=input('Label:'), alpha=opacity)
            if 'both' in sys.argv:
                for i in range(len(datasets)):
                    self.ax.plot(self.length_arrays[i, :][1:-1], vir_chunkX[i, :][1:-1],
                                ls=lt, marker=None, label=input('Label:'), alpha=opacity)
                for i in range(len(datasets)):
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

                for i in range(len(datasets)):
                    self.sigxz_t[i, :] =  get_variables.derive_data(datasets[i], skip).sigwall()[2]
                    self.avg_sigxz_t[i] = np.mean(sigxz_t[i, :])
                    self.ax.plot(self.time*1e-6,  self.sigxz_t[i, :], ls='-', marker=' ',label=input('Label:'), alpha=0.5)
                    self.ax.axhline(y= self.avg_sigxz_t[i], color=color[i], linestyle='dashed')

        if 'press_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            sigzz_t = np.zeros([len(datasets), len(self.time)])
            vir_t = np.zeros([len(datasets), len(self.time)])

            for i in range(len(datasets)):
                sigzz_t[i, :] = get_variables.derive_data(datasets[i], skip).sigwall()[3]
                vir_t[i, :] = get_variables.derive_data(datasets[i], skip).virial()[1]

            if 'virial' in sys.argv:
                for i in range(len(datasets)):
                    self.ax.plot(self.time[:]*1e-6, vir_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=1)
            if 'sigwall' in sys.argv:
                for i in range(len(datasets)):
                    self.ax.plot(self.time[:]*1e-6, sigzz_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=1)
            if 'both' in sys.argv:
                for i in range(len(datasets)):
                    self.ax.plot(self.time[:]*1e-6,  vir_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=0.5)
                for i in range(len(datasets)):
                    self.ax.plot(self.time[:]*1e-6,  sigzz_t[i, :], ls='-',
                                    marker=' ', label=input('Label:'), alpha=0.5)


    def temp(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[6])

        if 'temp_length' in sys.argv:
            tempX = np.zeros([len(datasets), self.Nx])
            self.ax.set_xlabel(labels[1])
            for i in range(len(datasets)):
                tempX[i, :] = get_variables.derive_data(datasets[i], skip).temp()[0]
                self.ax.plot(self.length_arrays[i, :], tempX[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'temp_height' in sys.argv:
            tempZ = np.zeros([len(datasets), self.Nz])
            self.ax.set_xlabel(labels[0])
            for i in range(len(datasets)):
                tempZ[i, :] = get_variables.derive_data(datasets[i], skip).temp()[1]
                self.ax.plot(self.height_arrays[i, :], tempZ[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'temp_time' in sys.argv:
            tempT = np.zeros([len(datasets), len(self.time)-skip])
            self.ax.set_xlabel(labels[2])
            for i in range(len(datasets)):
                tempT[i, :] = get_variables.derive_data(datasets[i], skip).temp()[2]
                self.ax.plot(self.time, tempT[i, :], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

    def pgrad_mflowrate(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[10])

        mpl.rcParams.update({'lines.markersize': 4})
        mpl.rcParams.update({'figure.figsize': (12,12)})

        mflowrate_avg = np.zeros([len(datasets)])
        shear_rates = np.zeros([len(datasets)])

        for i in range(len(datasets)):
            self.pGrad[i] = get_variables.derive_data(datasets[i], skip).virial()[2]
            mflowrate_avg[i] = np.mean(get_variables.derive_data(datasets[i], skip).mflux()[3])
            shear_rates[i] = get_variables.derive_data(datasets[i], skip).shear_rate()[0]
            self.viscosities[i] = get_variables.derive_data(datasets[i], skip).viscosity()     # mPa.s
            bulk_den_avg[i] = np.mean(get_variables.derive_data(datasets[i], skip).density()[2])

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

        for i in range(len(datasets)):
            self.pGrad[i] = get_variables.derive_data(datasets[i], skip).virial()[2]
            self.viscosities[i] = get_variables.derive_data(datasets[i], skip).viscosity()

        self.ax.plot(self.pGrad, self.viscosities, ls=lt, marker='o', alpha=opacity)

    def pgrad_slip(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel('Slip Length b (nm)')

        slip = np.zeros([len(datasets)])

        for i in range(len(datasets)):
            self.pGrad[i] = get_variables.derive_data(datasets[i], skip).virial()[2]
            slip[i] = get_variables.derive_data(datasets[i], skip).slip_length()[0]

        self.ax.plot(self.pGrad, slip, ls=lt, marker='o', alpha=opacity)


    def height_time(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[2])
        self.ax.set_ylabel(labels[0])

        gap_height = np.zeros([len(datasets), len(self.time)])
        gap_height_avg = np.zeros([len(datasets)])

        for i in range(len(datasets)):
            gap_height[i, :] = get_variables.derive_data(datasets[i], skip).h
            gap_height_avg[i] = np.mean(gap_height[i])

            self.ax.plot(self.time*1e-6, gap_height[i, :], ls='-', marker=' ', alpha=opacity)

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

    if len(datasets) > 0.:
        pds = plot_from_ds()

        if 'vx_height' in sys.argv:
            pds.vx_height()

        if 'vx_distrib' in sys.argv:
            pds.vx_distrib()

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
        pds.fig.savefig(sys.argv[-1] , format='png')


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
