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
# mpl.use('TkAgg')
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

from get_variables_211018 import derive_data as dd

color_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'seagreen','darkslategrey']

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

        self.Nx = len(dd(self.skip, self.datasets_x[0], self.datasets_z[0]).length_array)
        self.Nz = len(dd(self.skip, self.datasets_x[0], self.datasets_z[0]).height_array)

        self.time = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).time
        self.Ly = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).Ly

        # self.sigxz_t = np.zeros([len(datasets), len(self.time)])
        # self.avg_sigxz_t = np.zeros([len(datasets)])
        # self.pGrad = np.zeros([len(self.datasets_x)])
        # self.viscosities = np.zeros([len(self.datasets_z)])


    def label_inline(self, plot):
        rot = input('Rotation of label: ')
        xpos = np.float(input('Label x-pos:'))
        y_offset = np.float(input('Y-offset for label: '))

        for i in range (len(plot.ax.lines)):
            if plot.ax.lines[i].get_label().startswith('_'):
                pass
            else:
                label_lines.label_line(plot.ax.lines[i], xpos, yoffset= y_offset, \
                         label=plot.ax.lines[i].get_label(), fontsize= 8, rotation= rot)


    def draw_vlines(self, plot):
        total_length = np.max(dd(self.skip, self.datasets_x[0], self.datasets_z[0]).Lx)
        plot.ax.axvline(x= 0, color='k', linestyle='dashed', lw=1)

        for i in range(len(plot.ax.lines)-1):
            pos=np.float(input('vertical line pos:'))
            plot.ax.axvline(x= pos*total_length, color='k', linestyle='dashed', lw=1)


    def draw_inset(self, plot, xpos=0.62, ypos=0.57, w=0.2, h=0.28):

        inset_ax = self.fig.add_axes([xpos, ypos, w, h]) # X, Y, width, height
        # inset_ax.axvline(x=0, color='k', linestyle='dashed')
        # inset_ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
        # inset_ax.set_ylim(220, 280)

        inset_ax.plot(dd(self.skip, self.datasets_x[1], self.datasets_z[1]).length_array[1:29],
                      dd(self.skip, self.datasets_x[1], self.datasets_z[1]).virial()['vir_chunkX'][1:29],
                      ls= plot.ax.lines[1].get_linestyle(), color=plot.ax.lines[1].get_color(),
                      marker=None, alpha=1)


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
            err=None, draw_inset=None, twin_axis=None, **kwargs):

        self.ax.set_xlabel(labels[1])
        arrays = np.zeros([len(arr_to_plot[0]), len(self.datasets_x), self.Nx-2])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        for i in range(len(self.datasets_x)):
            if 'virxy_length' in arr_to_plot[0] and len(arr_to_plot[0])==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['Wxy_chunkX']

            if 'virxz_length' in arr_to_plot[0] and len(arr_to_plot[0])==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['Wxz_chunkX']

            if 'viryz_length' in arr_to_plot[0] and len(arr_to_plot[0])==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['Wyz_chunkX']

            if 'vir_length' in arr_to_plot[0] and len(arr_to_plot[0])==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['vir_chunkX']

            if 'sigzz_length' in arr_to_plot[0] and len(arr_to_plot[0])==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall()['sigzz_chunkX']

            if 'sigzz_length' in arr_to_plot[0] and 'vir_length' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['vir_chunkX']
                arrays[1, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall()['sigzz_chunkX']

            if 'sigxz_length' in arr_to_plot[0]:
                self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall()['sigxz_chunkX']

            if 'den_length' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).density(self.mf)['den_chunkX']

            if 'jx_length' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[4])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).mflux()['jx_chunkX']

            if 'temp_length' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[6])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).temp()['tempX']

            if 'mflowrate_length' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[10])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).mflux()['mflowrate_stable']

            for j in range(len(arr_to_plot[0])):
                pds.plot_settings()
                if err is not None:
                    q = input('fill or caps:')
                    if q == 'caps':
                        err_arrays = np.zeros_like(arrays)
                        err_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty()['err']
                        markers_err, caps, bars= self.ax.errorbar(dd(self.skip, self.datasets_x[i],
                             self.datasets_z[i]).length_array[1:-1], arrays[j, i],
                            yerr=err_arrays[j, i], label=input('label:'),
                            capsize=1.5, markersize=1, lw=2, alpha=opacity)
                        [bar.set_alpha(0.4) for bar in bars]
                        [cap.set_alpha(0.4) for cap in caps]

                    if q == 'fill':
                        lo_arrays, hi_arrays = np.zeros_like(arrays), np.zeros_like(arrays)
                        lo_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty()['lo']
                        hi_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty()['hi']

                        self.ax.fill_between( dd(self.skip, self.datasets_x[i], self.datasets_z[i]).length_array[1:-1],
                                    lo_arrays[j,i], hi_arrays[j,i], color=colors[i], alpha=0.5)
                        self.ax.plot(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).length_array[1:-1],
                                arrays[j,i], color=colors[i], marker=' ', label=input('label:'), alpha=0.7)

                else:
                    self.ax.plot(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).length_array[1:-1],
                            arrays[j,i], color=colors[i], label=input('label:'), alpha=0.7)

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)
        if draw_inset is not None:
            total_length = np.max(dd(self.skip, self.datasets_x[0], self.datasets_z[0]).Lx)

            inset_ax = self.fig.add_axes([0.62, 0.57, 0.2, 0.28]) # X, Y, width, height

            inset_ax.axvline(x=0, color='k', linestyle='dashed')
            inset_ax.axvline(x=0.2*total_length, color='k', linestyle='dashed')
            inset_ax.set_xlim(0, 8)
            inset_ax.set_ylim(245, 255)

            inset_ax.plot(dd(self.skip, self.datasets_x[2], self.datasets_z[2]).length_array[1:29],
                          dd(self.skip, self.datasets_x[2], self.datasets_z[2]).virial()['vir_chunkX'][1:29],
                          ls= self.ax.lines[2].get_linestyle(), color=self.ax.lines[2].get_color(),
                          marker=None, alpha=1)

            inset_ax.plot(dd(self.skip, self.datasets_x[3], self.datasets_z[3]).length_array[1:29],
                          dd(self.skip, self.datasets_x[3], self.datasets_z[3]).virial()['vir_chunkX'][1:29],
                          ls= self.ax.lines[3].get_linestyle(), color=self.ax.lines[3].get_color(),
                          marker=None, alpha=1)

        if twin_axis is not None:

            ax2 = self.ax.twinx()
            ax2.spines['left'].set_color(colors[0])
            ax2.spines['right'].set_color(colors[1])
            self.ax.tick_params(axis='y', colors=colors[0])
            ax2.tick_params(axis='y', colors=colors[1])
            self.ax.yaxis.label.set_color(colors[0])
            ax2.yaxis.label.set_color(colors[1])

            # for i in range(len(self.datasets_x)):
            ax2.plot(dd(self.skip, self.datasets_x[0], self.datasets_z[0]).length_array[1:-1],
                     dd(self.skip, self.datasets_x[0], self.datasets_z[0]).temp()['tempX'],
                     ls= '-', marker= ' ',
                     alpha=opacity, label=input('label:'), color=colors[1])

            ax2.plot(dd(self.skip, self.datasets_x[1], self.datasets_z[1]).length_array[1:-1],
                     dd(self.skip, self.datasets_x[1], self.datasets_z[1]).temp()['tempX'],
                     ls= '--', marker= ' ',
                     alpha=opacity, label=input('label:'), color=colors[1])


            ax2.set_ylabel(labels[6])

        if lab_lines is not None:
            pds.label_inline(self)

#TODO OPacity as input

    def qtty_height(self, *arr_to_plot, opacity=1,
                legend=None, lab_lines=None, draw_vlines=None,
                draw_inset=None, fit=None, extrapolate=None,
                couette=None, pd=None, err=None, **kwargs):

        self.ax.set_xlabel(labels[0])
        arrays = np.zeros([len(arr_to_plot[0]), len(self.datasets_z), self.Nz])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        for i in range(len(self.datasets_z)):
            if 'virxy_height' in arr_to_plot[0]:       # Here we use the bulk height
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['Wxy_chunkZ']

            if 'virxz_height' in arr_to_plot[0]:       # Here we use the bulk height
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['Wxz_chunkZ']

            if 'viryz_height' in arr_to_plot[0]:       # Here we use the bulk height
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['Wyz_chunkZ']

            if 'press_height' in arr_to_plot[0]:       # Here we use the bulk height
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['vir_chunkZ']

            if 'temp_height' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[6])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).temp()['tempZ']

            if 'vx_height' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[5])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).velocity()['vx_data']

            if 'continuum_vx' in arr_to_plot[0]:
                arrays[1, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).hydrodynamic()['v_hydro']

            if 'den_height' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).density(self.mf)['den_chunkZ']


            for j in range(len(arr_to_plot[0])):
                pds.plot_settings()
                x = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).height_array[arrays[j,i] != 0]
                y = arrays[j, i][arrays[j,i] != 0]

                if err is not None:

                    q = input('fill or caps:')
                    if q == 'caps':
                        err_arrays = np.zeros_like(arrays)
                        err_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty()['err']
                        markers_err, caps, bars= self.ax.errorbar(x, y,
                                yerr=err_arrays[j, i][arrays[j,i] != 0],
                                label=input('label:'), capsize=1.5, markersize=1,
                                lw=2, alpha=opacity)
                        [bar.set_alpha(0.4) for bar in bars]
                        [cap.set_alpha(0.4) for cap in caps]

                    if q == 'fill':
                        lo_arrays, hi_arrays = np.zeros_like(x), np.zeros_like(x)
                        lo_arrays = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty()['lo']
                        hi_arrays = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty()['hi']

                        self.ax.fill_between( x, lo_arrays, hi_arrays, color=colors[i], alpha=0.4)
                        self.ax.plot(x, y, color=colors[i], marker=' ', label=input('label:'), alpha=opacity)

                else:
                    if 'press_height' in arr_to_plot[0]:
                        x = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).bulk_height_array
                        y = arrays[0, i, :]

                    self.ax.plot(x, y, color=colors[i], label=input('Label:'), alpha=opacity)

            if fit is not None:
                order = 1#np.float(input('fit order:'))
                a = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).height_array[arrays[0,i] != 0]
                b = arrays[0, i][arrays[0,i] != 0]

                x1 = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).fit(a,b,order)['xdata']
                y1 = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).fit(a,b,order)['fit_data']

                self.ax.plot(x1, y1, color= colors[i], ls='-', marker=' ', alpha=0.9)

            if extrapolate is not None:
                if couette is not None:
                    x_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['xdata']
                    y_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['extrapolate']
                    self.ax.set_xlim([dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['root'], x[-1] + \
                    np.abs(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['root']) ])

                if pd is not None:
                    x_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['xdata']
                    y_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['extrapolate']
                    self.ax.set_xlim([dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['root'], x[-1] + \
                    dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['root'] ])
                # x_right = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length()['xdata_right']
                # y_right = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length()['extrapolate_right']

                self.ax.set_ylim([0,y[-1]])
                self.ax.plot(x_left, y_left, marker=' ', ls='--', color='black')
                # self.ax.plot(x_right, y_right, color='sienna')

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)



    def qtty_time(self, *arr_to_plot, opacity=1, draw_inset=None,
                legend=None, lab_lines=None, draw_vlines=None, **kwargs):

        cut = None
        # remove = len(self.time)-cut

        self.ax.set_xlabel(labels[2])
        arrays = np.zeros([len(arr_to_plot[0]), len(self.datasets_x), len(self.time[:cut])])
        arrays_avg = np.zeros([len(arr_to_plot[0]), len(self.datasets_x)])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)


        for i in range(len(self.datasets_x)):
            if 'temp_time' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[6])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).temp()['temp_t'][:cut]

            if 'mflowrate_time' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[10])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).mflux()['mflowrate_stable'][:cut]*1e18

            if 'jx_time' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[4])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).mflux()['jx_t'][:cut]

            if 'den_time' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).density(self.mf)['den_t'][:cut]

            if 'vir_time' in arr_to_plot[0] and len(arr_to_plot)==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['vir_t'][:cut]

            if 'sigzz_time' in arr_to_plot[0] and len(arr_to_plot)==1:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall()['sigzz_t'][:cut]

            if 'vir_time' in arr_to_plot[0] and 'sigzz_time' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['vir_t'][:cut]
                arrays[1, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall()['sigzz_t'][:cut]

            if 'sigxz_time' in arr_to_plot[0]:
                self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                arrays[0, i, :cut] =  dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall()['sigxz_t'][:cut]

            if 'height_time' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[0])
                arrays[0, i, :cut] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).h[:cut]

            for j in range(len(arr_to_plot[0])):
                arrays_avg[j,i] = np.mean(arrays[j, i, :])
                self.ax.plot(self.time[:cut]*1e-6, arrays[j, i, :],
                            label=input('Label:'), alpha=opacity)

                # TODO: Fix the color getter
                self.ax.axhline(y=arrays_avg[j,i], color=colors[i],
                    linestyle='dashed', lw=1)

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.label_inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)
        if draw_inset is not None:
            pds.draw_inset(self)


    def v_distrib(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel('Velocity (m/s)')
        self.ax.set_ylabel('Probability')

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        for i in range(len(self.datasets_x)):
            vx_values = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).vel_distrib()['vx_values']
            vx_prob = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).vel_distrib()['vx_prob']

            vy_values = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).vel_distrib()['vy_values']
            vy_prob = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).vel_distrib()['vy_prob']

            self.ax.plot(vx_values, vx_prob, ls='-', marker='o',
                    label=input('Label:'), alpha=opacity)
            # self.ax.plot(vy_values, vy_prob, ls='-', marker='x',
            #         label=input('Label:'), alpha=opacity)

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
            vir_pdiff.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial(pump_size[i])['pDiff'])
            vir_err.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial(pump_size[i])['pDiff_err'])
            sigzz_pdiff.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall(pump_size[i])['pDiff'])
            sigzz_err.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall(pump_size[i])['pDiff_err'])

        markers_a, caps, bars= self.ax.errorbar(pump_size, vir_pdiff, yerr=vir_err, ls=lt, fmt=mark,
                label='Virial (Fluid)', capsize=1.5, markersize=1.5, alpha=1)
        markers2, caps2, bars2= self.ax.errorbar(pump_size, sigzz_pdiff, yerr=sigzz_err, ls=lt, fmt='x',
                label='$\sigma_{zz}$ (Solid)', capsize=1.5, markersize=3, alpha=1)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        [bar2.set_alpha(0.5) for bar2 in bars2]
        [cap2.set_alpha(0.5) for cap2 in caps2]


    def pgrad_mflowrate(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[10])

        mpl.rcParams.update({'lines.markersize': 4})
        mpl.rcParams.update({'figure.figsize': (12,12)})

        pGrad, mflowrate_avg, shear_rate, mu, bulk_den_avg, mflowrate_hp, mflowrate_hp_slip = [], [], [], [], [], [], []

        for i in range(len(self.datasets_x)):

            avg_gap_height = np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).h)*1e-9
            slip_vel = np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).velocity()['vx_chunkZ_mod'][0])

            pGrad.append(np.absolute(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['pGrad']))
            mflowrate_avg.append(np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).mflux(self.mf)['mflowrate_stable']))
            shear_rate.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).shear_rate()['shear_rate'])
            mu.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).viscosity()['mu'])    # mPa.s
            bulk_den_avg.append(np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).density(self.mf)['den_t']))


            # Continuum qtts
            flowrate_cont = (bulk_den_avg[i]*1e3 * self.Ly*1e-9 * 1e-9 * 1e3 * pGrad[i]*1e6*1e9 * avg_gap_height**3)  / \
                        (12 * mu[i]*1e-3)

            mflowrate_hp.append(flowrate_cont)
            mflowrate_hp_slip.append(flowrate_cont + (bulk_den_avg[i]*1e6 *  \
                                self.Ly*1e-9 * slip_vel * avg_gap_height*1e-9) )

        self.ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        self.ax.plot(pGrad, np.array(mflowrate_hp)*1e18, ls='--', marker='o', alpha=opacity, label=input('Label:'))
        self.ax.plot(pGrad, np.array(mflowrate_hp_slip)*1e18, ls='--', marker='o', alpha=opacity, label=input('Label:'))
        self.ax.plot(pGrad, np.array(mflowrate_avg)*1e18, ls='-', marker='o', alpha=opacity, label=input('Label:'))

        ax2 = self.ax.twiny()
        ax2.set_xscale('log', nonpositive='clip')
        ax2.plot(shear_rate, mflowrate_avg, ls= ' ', marker= ' ', alpha=opacity, color=colors[0])
        ax2.set_xlabel(labels[11])

        if legend is not None:
            self.ax.legend()


    def pgrad_viscosity(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[12])

        mpl.rcParams.update({'lines.markersize': 4})

        pGrad, mu, shear_rate = [], [], []

        for i in range(len(self.datasets_x)):
            pGrad.append(np.absolute(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['pGrad']))
            if couette is not None:
                shear_rate.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(couette=1)['shear_rate'])
                mu.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(couette=1)['mu'])
            if pd is not None:
                shear_rate.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(pd=1)['shear_rate'])
                mu.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(pd=1)['mu'])

        self.ax.plot(pGrad, mu, ls= '-', marker='o', alpha=opacity,
                    label=input('Label:'))

        ax2 = self.ax.twiny()
        ax2.set_xscale('log', nonpositive='clip')
        ax2.plot(shear_rate, mu, ls= ' ', marker= ' ', alpha=opacity,
                    label=input('Label:'), color=colors[0])
        ax2.set_xlabel(labels[11])

        if legend is not None:
            self.ax.legend()


    def rate_viscosity(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel(labels[11])
        self.ax.set_ylabel(labels[12])
        self.ax.set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 4})

        shear_rate_couette, viscosity_couette = [], []
        shear_rate_pd, viscosity_pd = [], []
        shear_rate_pd_lo, viscosity_pd_lo = [], []
        shear_rate_pd_hi, viscosity_pd_hi = [], []
        couette_ds_x, couette_ds_z, pump_ds_x, pump_ds_z = [], [], [], []

        for i, j in zip(self.datasets_x, self.datasets_z):
            if 'couette' in i:
                couette_ds_x.append(i)
                couette_ds_z.append(j)
            if 'ff' in i or 'fc' in i:
                pump_ds_x.append(i)
                pump_ds_z.append(j)

        if couette_ds_x:
            for i in range(len(couette_ds_x)):
                shear_rate_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['shear_rate'])
                viscosity_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['mu'])
            self.ax.plot(shear_rate_couette, viscosity_couette, ls= '-', marker='o', alpha=opacity, label=input('Label:'))

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate'])
                viscosity_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['mu'])

                # shear_rate_pd_lo.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate_lo'])
                # viscosity_pd_lo.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['mu_lo'])
                #
                # shear_rate_pd_hi.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate_hi'])
                # viscosity_pd_hi.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['mu_hi'])

            self.ax.plot(shear_rate_pd, viscosity_pd, ls= '--', marker='o', alpha=opacity, label=input('Label:'))
            # self.ax.plot(shear_rate_pd_lo, viscosity_pd_lo, ls= '--', marker='o', alpha=opacity, label='Lower limit')
            # self.ax.plot(shear_rate_pd_hi, viscosity_pd_hi, ls= '--', marker='o', alpha=opacity, label='Upper limit')

        if legend is not None:
            self.ax.legend()


    def rate_stress(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel(labels[11])
        self.ax.set_ylabel('$\sigma_{xz}$ (MPa)')
        self.ax.set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 4})

        shear_rate_couette, stress_couette = [], []
        shear_rate_pd, stress_pd = [], []
        couette_ds_x, couette_ds_z, pump_ds_x, pump_ds_z = [], [], [], []

        for i, j in zip(self.datasets_x, self.datasets_z):
            if 'couette' in i:
                couette_ds_x.append(i)
                couette_ds_z.append(j)
            if 'ff' in i or 'fc' in i:
                pump_ds_x.append(i)
                pump_ds_z.append(j)

        if couette_ds_x:
            for i in range(len(couette_ds_x)):
                shear_rate_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['shear_rate'])
                stress_couette.append(np.mean(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).sigwall(couette=1)['sigxz_t']))
            self.ax.plot(shear_rate_couette, stress_couette, ls= '-', marker='o', alpha=opacity, label=input('Label:'))

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate'])
                stress_pd.append(np.mean(dd(self.skip, pump_ds_x[i], pump_ds_z[i]).sigwall(pd=1)['sigxz_t']))
            self.ax.plot(shear_rate_pd, stress_pd, ls= '--', marker='o', alpha=opacity, label=input('Label:'))

        if legend is not None:
            self.ax.legend()


    def rate_slip(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel(labels[11])
        self.ax.set_ylabel('Slip Length (nm)')
        self.ax.set_xscale('log', nonpositive='clip')

        shear_rate_couette, slip_couette = [], []
        shear_rate_pd, slip_pd = [], []
        couette_ds_x, couette_ds_z, pump_ds_x, pump_ds_z = [], [], [], []

        for i, j in zip(self.datasets_x, self.datasets_z):
            if 'couette' in i:
                couette_ds_x.append(i)
                couette_ds_z.append(j)
            if 'ff' in i or 'fc' in i:
                pump_ds_x.append(i)
                pump_ds_z.append(j)

        if couette_ds_x:
            for i in range(len(couette_ds_x)):
                shear_rate_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['shear_rate'])
                slip_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).slip_length(couette=1)['Ls'])
            self.ax.plot(shear_rate_couette, slip_couette, ls= '-', marker='o', alpha=opacity, label=input('Label:'))

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate'])
                slip_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).slip_length(pd=1)['Ls'])
            self.ax.plot(shear_rate_pd, slip_pd, ls= '--', marker='o', alpha=opacity, label=input('Label:'))

        if legend is not None:
            self.ax.legend()

    def struc_factor(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel('$k_{x}$')
        self.ax.set_ylabel('$s(k_x)$')

        freq = 2*np.pi / (self.time*1e-6)

        kx = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['kx']
        sf = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['sf']

        skx = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['ISFx']
        swx = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['DSFx']

        self.ax.plot(kx[1:], sf[1:], ls= '-', marker=' ', alpha=opacity,
                    label=input('Label:'))
        # self.ax.plot(freq, swx, ls= '-', marker='o', alpha=opacity,
        #             label=input('Label:'))


    def acf(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        mpl.rcParams.update({'lines.linewidth':'1'})

        self.ax.set_xlabel(labels[2])
        self.ax.set_ylabel(r'${\rm C_{AA}}$')

        arr = np.zeros([len(self.datasets_x), len(self.time)])

        for i in range(len(self.datasets_x)):
            arr[i, :] = dd(self.skip, self.datasets_x[i],
                                self.datasets_z[i]).mflux(self.mf)['jx_t']
            # Auto-correlation function
            acf = sq.acf(arr[i, :])['norm']

            #TODO: Cutoff
            self.ax.plot(self.time[:10000]*1e-6, acf[:10000],
                        label=input('Label:'), alpha=opacity)

        self.ax.axhline(y= 0, color='k', linestyle='dashed', lw=1)


        if legend is not None:
            self.ax.legend()

    def transverse(self, opacity=1, legend=None, lab_lines=None, **kwargs):
        mpl.rcParams.update({'lines.linewidth':'1'})

        self.ax.set_xlabel(labels[2])
        self.ax.set_ylabel(r'${\rm C_{AA}}$')

        # arr = np.zeros([len(self.datasets_x), len(self.time)])

        for i in range(len(self.datasets_x)):
            # Auto-correlation function
            acf = dd(self.skip, self.datasets_x[i],
                                self.datasets_z[i]).trans(self.mf)['a']
            #TODO: Cutoff
            self.ax.plot(self.time[:10000]*1e-6, acf[:10000],
                        label=input('Label:'), alpha=opacity)

        self.ax.axhline(y= 0, color='k', linestyle='dashed', lw=1)

        if legend is not None:
            self.ax.legend()


#
# Adjust ticks
# plt.yticks(np.arange(30, 80, step=10))  # Set label locations.

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
