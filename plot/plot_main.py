#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4
import re
import numpy as np
import sys
import os
import funcs
import sample_quality as sq
import label_lines
from cycler import cycler
import itertools

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

from get_variables import derive_data as dd

color_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

linestyles= {"line":"-", "dashed":"--", "dashdot":"-."}

opacities= {"transperent":0.3, "intermed":0.6, "opaque":0.9}

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

plt.style.use('imtek')
# mpl.rcParams.update({'axes.prop_cycle': color_cycler})


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

    def __init__(self, skip, datasets_x, datasets_z, mf):

        self.skip=skip
        self.mf=mf
        self.datasets_x=datasets_x
        self.datasets_z=datasets_z

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')

        self.Nx = len(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).length_array)
        self.Nz = len(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).height_array)

        self.time = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).time
        self.Ly = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).Ly

        # self.sigxz_t = np.zeros([len(datasets), len(self.time)])
        # self.avg_sigxz_t = np.zeros([len(datasets)])
        self.pGrad = np.zeros([len(self.datasets_x)])
        self.viscosities = np.zeros([len(self.datasets_z)])


    def label_inline(self, plot):
        rot = input('Rotation of label: ')
        xpos = np.float(input('Label x-pos:'))
        y_offset = 0 #np.float(input('Y-offset for label: '))

        for i in range (len(plot.ax.lines)):
            label_lines.label_line(plot.ax.lines[i], xpos, yoffset= y_offset, \
                     label= plot.ax.lines[i].get_label(), fontsize= 8, rotation= rot)


    def draw_vlines(self, plot):
        total_length = np.max(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).Lx)
        plot.ax.axvline(x= 0, color='k', linestyle='dashed', lw=1)

        for i in range(len(plot.ax.lines)-1):
            pos=np.float(input('vertical line pos:'))
            plot.ax.axvline(x= pos*total_length, color='k', linestyle='dashed', lw=1)


    def draw_inset(self, xpos, ypos, w, h):
        popt2, pcov2 = curve_fit(funcs.quadratic, self.height_arrays_mod[0][1:-1],
                                                    vx_chunkZ[0, :][1:-1])

        inset_ax = fig.add_axes([0.6, 0.48, 0.2, 0.28]) # X, Y, width, height
        inset_ax.plot(height_arrays_mod[0][-31:-1], vx_chunkZ[0, :][-31:-1],
                                ls= ' ', marker=mark, alpha=opacity, label=label)
        inset_ax.plot(height_arrays_mod[0][-31:-1], funcs.quadratic(height_arrays_mod[0], *popt2)[-31:-1])


        if 'inset' in sys.argv:
            inset_ax = fig.add_axes([0.62, 0.57, 0.2, 0.28]) # X, Y, width, height
            inset_ax.axvline(x=0, color='k', linestyle='dashed')
            inset_ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
            inset_ax.set_ylim(220, 280)
            inset_ax.plot(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).length_array[0][1:29],
                         vir_chunkX[0, :][1:29] , ls= lt, color=ax.lines[0].get_color(),
                         marker=None, alpha=opacity, label=label)
            inset_ax.plot(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf).length_array[0][1:29],
                         vir_chunkX[1, :][1:29] , ls= ' ', color=ax.lines[1].get_color(),
                         marker='x', alpha=opacity, label=label)


    def plot_settings(self):
        # Default is dashed with marker 'o'
        mpl.rcParams.update({'lines.linestyle':'-'})
        mpl.rcParams.update({'lines.marker':'o'})

        new_lstyle = input('change lstyle:')
        if  new_lstyle == 'dd': mpl.rcParams.update({'lines.linestyle':'--'})
        if new_lstyle == 'ddot': mpl.rcParams.update({'lines.linestyle':'-.'})
        if new_lstyle == 'none': mpl.rcParams.update({'lines.linestyle':' '})

        new_mstyle = input('change mstyle:')
        if new_mstyle == 'x': mpl.rcParams.update({'lines.marker':'x'})
        if new_mstyle == '+': mpl.rcParams.update({'lines.marker':'+'})
        if new_mstyle == '*': mpl.rcParams.update({'lines.marker':'*'})
        if new_mstyle == 'none': mpl.rcParams.update({'lines.marker':' '})



    def qtty_len(self, *arr_to_plot, opacity=1.0,
            legend=None, lab_lines=None, draw_vlines=None,
            err=None, **kwargs):

        self.ax.set_xlabel(labels[1])
        arrays = np.zeros([len(arr_to_plot), len(self.datasets_x), self.Nx-2])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        for i in range(len(self.datasets_x)):
            if 'vir_chunkX' in arr_to_plot and len(arr_to_plot)==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()['vir_chunkX']

            if 'sigzz_chunkX' in arr_to_plot and len(arr_to_plot)==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall()['sigzz_chunkX']

            if 'sigzz_chunkX' in arr_to_plot and 'vir_chunkX' in arr_to_plot:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()['vir_chunkX']
                arrays[1, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall()['sigzz_chunkX']

            if 'sigxz_chunkX' in sys.argv:
                self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall()['sigxz_chunkX']

            if 'den_length' in arr_to_plot:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).density()['den_chunkX']

            if 'jx_length' in arr_to_plot:
                self.ax.set_ylabel(labels[4])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).mflux()['jx_chunkX']

            if 'temp_length' in arr_to_plot:
                self.ax.set_ylabel(labels[6])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).temp()['tempX']

            if 'mflowrate_length' in arr_to_plot:
                self.ax.set_ylabel(labels[10])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).mflux()['mflowrate_stable']

            for j in range(len(arr_to_plot)):
                pds.plot_settings()
                if err is not None:
                    err_arrays = np.zeros_like(arrays)
                    err_arrays[j, i, :] = dd(self.datasets_x[i], self.datasets_z[i],
                                self.skip).uncertainty()[arr_to_plot[j]+'_err']
                    markers_err, caps, bars= self.ax.errorbar(dd(self.datasets_x[i],
                         self.datasets_z[i], self.skip).length_array[1:-1], arrays[j, i],
                        yerr=err_arrays[j, i], label=input('label:'),
                        capsize=1.5, markersize=1, lw=0.7, alpha=opacity)

                    [bar.set_alpha(0.4) for bar in bars]
                    [cap.set_alpha(0.4) for cap in caps]

                else:
                    self.ax.plot(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).length_array[1:-1],
                            arrays[j,i], label=input('label:'), alpha=0.7)

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)



    def qtty_height(self, *arr_to_plot, opacity=1,
                legend=None, lab_lines=None, draw_vlines=None,
                fit=None, extrapolate=None, **kwargs):

        self.ax.set_xlabel(labels[0])
        arrays = np.zeros([len(arr_to_plot), len(self.datasets_z), self.Nz])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        for i in range(len(self.datasets_z)):
            if 'press_height' in arr_to_plot:       # Here we use the bulk height
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()['vir_chunkZ']

            if 'temp_height' in arr_to_plot:
                self.ax.set_ylabel(labels[6])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).temp()['tempZ']

            if 'vx_height' in arr_to_plot:
                self.ax.set_ylabel(labels[5])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).velocity()['vx_data']

            if 'continuum_vx' in arr_to_plot:
                arrays[1, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).hydrodynamic()['v_hydro']

            if 'den_height' in sys.argv:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).density()['den_chunkZ']

            for j in range(len(arr_to_plot)):
                pds.plot_settings()
                if 'press_height' not in arr_to_plot:
                    x = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).height_array[arrays[j,i] != 0]
                    y = arrays[j, i][arrays[j,i] != 0]
                if 'press_height' in arr_to_plot:
                    x = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).bulk_height_array
                    y = arrays[0, i, :]

                self.ax.plot(x, y, color=colors[i], label=input('Label:'), alpha=opacity)

                if fit is not None:
                    a = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).height_array[arrays[j,i] != 0]
                    b = arrays[j, i][arrays[j,i] != 0]

                    x1 = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).fit(a,b)['xdata']
                    y1 = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).fit(a,b)['fit_data']

                    self.ax.plot(x1, y1, color= colors[i], ls='-', marker=' ', alpha=0.5)

                if extrapolate is not None:
                    x_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()['xdata_left']
                    y_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()['extrapolate_left']
                    x_right = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()['xdata_right']
                    y_right = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()['extrapolate_right']

                    self.ax.set_xlim([dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()['root_left'],
                                      dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()['root_right']])
                    self.ax.set_ylim(bottom = 0)

                    self.ax.plot(x_left, y_left, color='sienna')
                    self.ax.plot(x_right, y_right, color='sienna')

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)



    def qtty_time(self, *arr_to_plot, opacity=1,
                legend=None, lab_lines=None, draw_vlines=None, **kwargs):

        self.ax.set_xlabel(labels[2])
        arrays = np.zeros([len(arr_to_plot), len(self.datasets_x), len(self.time)])
        arrays_avg = np.zeros(len(self.datasets_x))
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        for i in range(len(self.datasets_x)):
            if 'temp_time' in arr_to_plot:
                self.ax.set_ylabel(labels[6])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).temp()['temp_t']

            if 'mflowrate_time' in arr_to_plot:
                self.ax.set_ylabel(labels[10])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).mflux()['mflowrate_stable']*1e18

            if 'jx_time' in arr_to_plot:
                self.ax.set_ylabel(labels[4])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).mflux()['jx_t']

            if 'den_time' in arr_to_plot:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).density()['den_t']

            if 'vir_time' in arr_to_plot and len(arr_to_plot)==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()['vir_t']

            if 'sigzz_time' in arr_to_plot and len(arr_to_plot)==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall()['sigzz_t']

            if 'vir_time' in arr_to_plot and 'sigzz_time' in arr_to_plot:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()['vir_t']
                arrays[1, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall()['sigzz_t']

            if 'sigxz_time' in arr_to_plot:
                self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                arrays[0, i, :] =  dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall()['sigxz_t']

            if 'height_time' in arr_to_plot:
                self.ax.set_ylabel(labels[0])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).h

            arrays_avg[i] = np.mean(arrays[0, i, :])
            for j in range(len(arr_to_plot)):
                self.ax.plot(self.time*1e-6, arrays[0, i, :],
                            label=input('Label:'), alpha=opacity)

                self.ax.axhline(y=arrays_avg[i], color=colors[i],
                    linestyle='dashed', lw=1)

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)


    def v_distrib(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel('Velocity (m/s)')
        self.ax.set_ylabel('Probability')

        pds = plot_from_ds()

        for i in range(len(self.datasets_x)):
            vx_values = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).vel_distrib()['vx_values']
            vx_prob = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).vel_distrib()['vx_prob']

            vy_values = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).vel_distrib()['vy_values']
            vy_prob = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).vel_distrib()['vy_prob']

            self.ax.plot(vx_values, vx_prob, ls='-', marker='o',
                    label=input('Label:'), alpha=opacity)
            self.ax.plot(vy_values, vy_prob, ls='-', marker='x',
                    label=input('Label:'), alpha=opacity)

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)


    def pdiff_pumpsize(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel('Normalized pump length')

        pump_size = []
        vir_pdiff, sigzz_pdiff, vir_err, sigzz_err = [], [], [], []
        for i in range(len(self.datasets_x)):
            pump_size.append(np.float(input('pump size:')))
            vir_pdiff.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial(pump_size[i])['pDiff'])
            vir_err.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial(pump_size[i])['pDiff_err'])
            sigzz_pdiff.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall(pump_size[i])['pDiff'])
            sigzz_err.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall(pump_size[i])['pDiff_err'])

        markers_a, caps, bars= self.ax.errorbar(pump_size, vir_pdiff, yerr=vir_err, ls=lt, fmt=mark,
                label='Virial (Fluid)', capsize=1.5, markersize=1.5, alpha=1)
        markers2, caps2, bars2= self.ax.errorbar(pump_size, sigzz_pdiff, yerr=sigzz_err, ls=lt, fmt='x',
                label='$\sigma_{zz}$ (Solid)', capsize=1.5, markersize=3, alpha=1)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        [bar2.set_alpha(0.5) for bar2 in bars2]
        [cap2.set_alpha(0.5) for cap2 in caps2]


    def pdiff_err(self):

        self.ax.set_xlabel(labels[7])
        self.ax.set_ylabel('Uncertainty $\zeta$ %')

        pressures = np.array([0.5, 5, 50, 250, 500])
        vir_err, sigzz_err = [], []
        for i in range(len(self.datasets_x)):
            vir_err.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial(0.2)['pDiff_err'])
            # sigzz_err.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).sigwall(0.2)['pDiff_err'])

        vir_err_rel = vir_err / pressures
        self.ax.plot(pressures, vir_err_rel, ls='-', marker='o',label='Virial (Fluid)', alpha=1)
        # self.ax.plot(pressures, sigzz_err, ls='-', marker='o',label='$\sigma_{zz}$ (Solid)', alpha=1)


    def pgrad_mflowrate(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[10])

        mpl.rcParams.update({'lines.markersize': 4})
        mpl.rcParams.update({'figure.figsize': (12,12)})

        mflowrate_avg = np.zeros([len(self.datasets_x)])
        shear_rates = np.zeros([len(self.datasets_x)])

        for i in range(len(self.datasets_x)):
            self.pGrad[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial(np.float(input('pump size:')))[2]
            mflowrate_avg[i] = np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).mflux()[3])
            shear_rates[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).shear_rate()[0]
            self.viscosities[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).viscosity()     # mPa.s
            bulk_den_avg[i] = np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).density()[2])

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


    def pgrad_viscosity(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[12])

        mpl.rcParams.update({'lines.markersize': 4})

        self.ax.set_xscale('log')
        self.ax.set_yscale('log')


        for i in range(len(self.datasets_x)):
            self.pGrad[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()[2]
            self.viscosities[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).viscosity()

        self.ax.plot(self.pGrad, self.viscosities, ls=lt, marker='o', alpha=opacity)

    def pgrad_slip(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel('Slip Length b (nm)')

        slip = np.zeros([len(self.datasets_x)])

        for i in range(len(self.datasets_x)):
            self.pGrad[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).virial()[2]
            slip[i] = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf).slip_length()[0]

        self.ax.plot(self.pGrad, slip, ls=lt, marker='o', alpha=opacity)


    def weight(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[1])
        self.ax.set_ylabel('Weight')

        length_padded = np.pad(self.length_arrays[0], (1,0), 'constant')

        self.ax.plot(length_padded, funcs.quartic(length_padded), ls=lt, marker=None, alpha=opacity)
        self.ax.plot(length_padded, funcs.step(length_padded), ls='--', marker=None, alpha=opacity)



#
# Adjust ticks
# plt.yticks(np.arange(30, 80, step=10))  # Set label locations.

# pds.ax.lines[i].get_color()

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
