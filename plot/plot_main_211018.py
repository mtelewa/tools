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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.image as image
import matplotlib.cm as cmx
from matplotlib.lines import Line2D

import scipy.constants as sci
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

    def plot_from_txt(self, skip, txtfiles, outfile, lt='-', mark='o', opacity=1.0):

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

        # fig.tight_layout()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        ax.set_xlabel(labels[7])
        ax.set_ylabel(labels[3])

        verlet_den_data, verlet_p_data = [], []
        gautschi_den_data, gautschi_p_data = [], []
        verlet_den_data_4, verlet_p_data_4 = [], []
        gautschi_den_data_4, gautschi_p_data_4 = [], []

        exp_density = [0.630, 0.653, 0.672, 0.686, 0.714, 0.739, 0.750]
        exp_press = [28.9, 55.3, 84.1, 110.2, 171.0, 239.5, 275.5]

        for idx, val in enumerate(txtfiles):

            if 'gautschi' in val and '4fs' not in val:
                data = np.loadtxt(txtfiles[idx], skiprows=skip, dtype=float)
                xdata = data[:,10]
                ydata = data[:,12]

                gautschi_den_data.append(np.mean(xdata))
                gautschi_p_data.append(np.mean(ydata) * sci.atm * 1e-6)

            if 'verlet' in val and '4fs' not in val:
                data = np.loadtxt(txtfiles[idx], skiprows=skip, dtype=float)
                xdata = data[:,10]
                ydata = data[:,12]

                verlet_den_data.append(np.mean(xdata))
                verlet_p_data.append(np.mean(ydata) * sci.atm * 1e-6)

            if 'gautschi' in val and '4fs' in val:
                data = np.loadtxt(txtfiles[idx], skiprows=skip, dtype=float)
                xdata = data[:,10]
                ydata = data[:,12]

                gautschi_den_data_4.append(np.mean(xdata))
                gautschi_p_data_4.append(np.mean(ydata) * sci.atm * 1e-6)

            if 'verlet' in val and '4fs' in val:
                data = np.loadtxt(txtfiles[idx], skiprows=skip, dtype=float)
                xdata = data[:,10]
                ydata = data[:,12]

                verlet_den_data_4.append(np.mean(xdata))
                verlet_p_data_4.append(np.mean(ydata) * sci.atm * 1e-6)

        print(gautschi_p_data)

        ax.plot(verlet_p_data, verlet_den_data, ls='-', marker='o', color=colors[0], alpha=1, label='Verlet',)
        ax.plot(verlet_p_data_4, verlet_den_data_4, ls='--', marker=' ', color=colors[0], alpha=1, label=None,)

        if gautschi_den_data:
            ax.plot(gautschi_p_data, gautschi_den_data, ls='-', marker='x', color=colors[1], alpha=1, label='Gautschi',)
            ax.plot(gautschi_p_data_4, gautschi_den_data_4, ls='--', marker=' ', color=colors[1], alpha=1, label=None,)

        ax.plot(exp_press, exp_density, ls=' ', marker='*', alpha=1, color=colors[2], label='Expt. (Liu et al. 2010)',)


            # popt, pcov = curve_fit(funcs.linear, xdata, ydata)
            # ax.plot(xdata, funcs.linear(xdata, *popt))
            # ax.errorbar(xdata, ydata1 , yerr=err1, ls=lt, fmt=mark, label= 'Expt. (Gehrig et al. 1978)', capsize=3, alpha=opacity)

        ax.legend()
        fig.savefig(outfile)



class plot_from_ds:

    def __init__(self, skip, datasets_x, datasets_z, mf, plot_type='2d'):

        self.skip=skip
        self.mf=mf
        self.datasets_x=datasets_x
        self.datasets_z=datasets_z
        self.plot_type = plot_type

        if self.plot_type=='2d':
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)
            self.ax.xaxis.set_ticks_position('both')
            self.ax.yaxis.set_ticks_position('both')
        else:
            self.fig, self.ax = plt.figure(dpi=1200), plt.axes(projection='3d')
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

        rot = input('Rotation of label:') or None
        xpos = np.float(input('Label x-pos:'))

        for i in range (len(plot.ax.lines)):
            # print(plot.ax.lines[i])
            if plot.ax.lines[i].get_label().startswith('_'):
                pass
            else:
                y_offset = np.float(input('Y-offset for label: '))
                label_lines.label_line(plot.ax.lines[i], xpos, yoffset= y_offset, \
                         label=plot.ax.lines[i].get_label(), fontsize= 12, rotation= rot)


    def draw_vlines(self, plot):
        total_length = np.max(dd(self.skip, self.datasets_x[0], self.datasets_z[0]).Lx)
        plot.ax.axvline(x= 0*total_length, color='k', linestyle='dotted', lw=1.5)

        #for i in range(len(plot.ax.lines)-1):
        pos=np.float(input('vertical line pos:'))
        plot.ax.axvline(x= pos*total_length, color='k', linestyle='dotted', lw=1.5)


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
        if  new_lstyle == 'dd' or new_lstyle == '--':
            mpl.rcParams.update({'lines.linestyle':'--'})
            mpl.rcParams.update({'lines.linewidth':2})
        if new_lstyle == 'ddot' or new_lstyle == '-.': mpl.rcParams.update({'lines.linestyle':'-.'})
        if new_lstyle == 'none': mpl.rcParams.update({'lines.linestyle':' '})

        new_mstyle = input('change mstyle:')
        if new_mstyle == 'x': mpl.rcParams.update({'lines.marker':'x'})
        if new_mstyle == '+': mpl.rcParams.update({'lines.marker':'+'})
        if new_mstyle == '*': mpl.rcParams.update({'lines.marker':'*'})
        if new_mstyle == 'none': mpl.rcParams.update({'lines.marker':' '})



    def qtty_len(self, *arr_to_plot, opacity,
            legend=None, lab_lines=None, draw_vlines=None,
            err=None, draw_inset=None, twin_axis=None, **kwargs):

        self.ax.set_xlabel(labels[1])
        arrays = np.zeros([len(arr_to_plot[0]), len(self.datasets_x), self.Nx-2])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)
        n = 0

        mpl.rcParams.update({'lines.linewidth': 2})
        mpl.rcParams.update({'lines.markersize': 6})

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
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall(pd=1)['sigzz_chunkX']

            if 'sigzz_length' in arr_to_plot[0] and 'vir_length' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[7])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['vir_chunkX']
                arrays[1, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).sigwall(pd=1)['sigzz_chunkX']

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
                x = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).length_array[1:-1]
                y = arrays[j, i, :]
                # np.savetxt(f'vir_length_{i}.txt', np.c_[x,y], delimiter= "  ", header= "Length(nm)       Sigmazz(MPa)")

                if err is not None:
                    q = input('fill or caps:')
                    qtty=input('qtty:')

                    if q == 'caps':
                        err_arrays = np.zeros_like(arrays)
                        err_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty(qtty)['err']
                        markers_err, caps, bars= self.ax.errorbar(dd(self.skip, self.datasets_x[i],
                             self.datasets_z[i]).length_array[1:-1], arrays[j, i],
                            yerr=err_arrays[j, i], label=input('label:'),
                            capsize=1.5, markersize=1, lw=2, alpha=opacity)
                        [bar.set_alpha(0.4) for bar in bars]
                        [cap.set_alpha(0.4) for cap in caps]

                    if q == 'fill':
                        lo_arrays, hi_arrays = np.zeros_like(arrays), np.zeros_like(arrays)
                        lo_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty(qtty)['lo']
                        hi_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty(qtty)['hi']

                        a = input('plot?:')
                        if a =='y':
                            self.ax.fill_between(x, lo_arrays[j,i], hi_arrays[j,i], color=colors[n], alpha=0.4)
                            self.ax.plot(x, y, color=colors[n], marker=' ', label=input('label:'), alpha=1)
                        else: pass
                        n+=1

                else:
                    if twin_axis is None:
                        # a = input('plot?:')
                        # if a =='y':
                        self.ax.plot(x, y, color=input('color:'), label=input('label:'), alpha=opacity)
                        # else: pass
                    else:
                        self.ax.plot(x, y, color=colors[0], label=input('label:'), alpha=opacity)

        if legend is not None:
            # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Fixed Force'),
            #                    Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Fixed Current')]
            # self.ax.legend(handles=legend_elements, frameon=False)
            self.ax.legend(frameon=False)
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

#TODO Opacity as input

    def qtty_height(self, *arr_to_plot, opacity,
                legend=None, lab_lines=None, draw_vlines=None,
                draw_inset=None, fit=None, extrapolate=None,
                couette=None, pd=None, err=None, **kwargs):

        self.ax.set_xlabel(labels[0])
        arrays = np.zeros([len(arr_to_plot[0]), len(self.datasets_z), self.Nz])
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf)

        # mpl.rcParams.update({'lines.linewidth': 1.5})
        # mpl.rcParams.update({'lines.markersize': 2})

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

            if 'den_height' in arr_to_plot[0]:
                self.ax.set_ylabel(labels[3])
                arrays[0, i, :] = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).density(self.mf)['den_chunkZ']


            for j in range(len(arr_to_plot[0])):
                pds.plot_settings()
                x = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).height_array[arrays[j,i] != 0]
                y = arrays[j, i][arrays[j,i] != 0]
                # np.savetxt(f'{i}.txt', np.c_[x,y], delimiter= "  ", header= "Height(nm)       Velocity(m/s)")

                if err is not None:

                    q = input('fill or caps:')
                    qtty=input('qtty:')

                    if q == 'caps':
                        err_arrays = np.zeros_like(arrays)
                        err_arrays[j, i, :] = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty(qtty)['err']
                        markers_err, caps, bars= self.ax.errorbar(x, y,
                                yerr=err_arrays[j, i][arrays[j,i] != 0],
                                label=input('label:'), capsize=1.5, markersize=1,
                                lw=2, alpha=opacity)
                        [bar.set_alpha(0.4) for bar in bars]
                        [cap.set_alpha(0.4) for cap in caps]

                    if q == 'fill':
                        lo_arrays, hi_arrays = np.zeros_like(x), np.zeros_like(x)
                        lo_arrays = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty(qtty)['lo']
                        hi_arrays = dd(self.skip, self.datasets_x[i],
                                    self.datasets_z[i]).uncertainty(qtty)['hi']

                        self.ax.fill_between(x, lo_arrays, hi_arrays, color=colors[i], alpha=0.4)
                        self.ax.plot(x, y, color=colors[i], marker=input('marker:'), label=input('label:'), alpha=opacity)

                else:
                    if 'press_height' in arr_to_plot[0]:
                        x = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).bulk_height_array
                        y = arrays[j, i, :]

                    self.ax.plot(x, y, color=colors[i], label=input('Label:'), markersize=6, alpha=opacity)

            if fit is not None:
                a = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).height_array[arrays[0,i] != 0][1:-1]
                b = arrays[0, i][arrays[0,i] != 0][1:-1]

                try:
                    order = np.int(input('fit order:'))
                    ans = input('plot fit?:')
                    if ans == 'y':
                        y1 = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).fit(a,b,order)['fit_data']
                        self.ax.plot(a, y1, color= 'k', ls='--', lw=1.5, marker=' ', label='_', alpha=1)

                        y2 = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).fit_with_cos(a,b,order)['fit_data']
                        self.ax.plot(a, y2, color= 'k', ls='-', lw=1.5, marker=' ', label='_', alpha=1)
                    else:
                        pass

                    # y3 = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).fit_with_fourier(a,b)['fit_data']
                    # print(y3)
                    # self.ax.plot(a, y3, color= 'k', ls='dotted', lw=1.5, marker=' ', label=' New Quadratic + Cosine Fit', alpha=1)

                except ValueError:
                    y1 = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).fit_with_cos(a,b)['fit_data']
                    self.ax.plot(a, y1, color= 'k', ls='--', lw=1.5, marker=' ', label='Fit', alpha=1)



            if extrapolate is not None:

                if couette is not None:
                    x_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['xdata']
                    y_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['extrapolate']
                    self.ax.set_xlim([dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['root'],
                        np.max(x) + np.abs(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(couette=1)['root'])])

                if pd is not None:
                    x_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['xdata']
                    y_left = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['extrapolate']
                    self.ax.set_xlim([dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['root'],
                        np.max(x) + np.abs(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length(pd=1)['root'])])
                # x_right = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length()['xdata_right']
                # y_right = dd(self.skip, self.datasets_x[i], self.datasets_z[i]).slip_length()['extrapolate_right']

                self.ax.set_ylim([0, 1.1*np.max(y)])
                self.ax.plot(x_left, y_left, marker=' ', ls='dotted', color='k')

        if legend is not None:
            # #where some data has already been plotted to ax
            # handles, labs = self.ax.get_legend_handles_labels()
            # #Additional elements
            # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Quartic fit'),
            #                    Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Quartic + Cosine fit')]
                               # Line2D([0], [0], color='k', lw=2.5, ls='dotted', marker=' ', label='Lin. extrapolation')]
            # handles.extend(legend_elements)
            # self.ax.legend(handles=handles, frameon=False)

            self.ax.legend(frameon=False)

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
            self.ax.legend(frameon=False)
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
            self.ax.legend(frameon=False)
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

        mpl.rcParams.update({'lines.markersize': 6})
        mpl.rcParams.update({'figure.figsize': (12,12)})

        pGrad, mflowrate_avg, shear_rate, mu, bulk_den_avg, mflowrate_hp, mflowrate_hp_slip = [], [], [], [], [], [], []

        for i in range(len(self.datasets_x)):

            avg_gap_height = np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).h)*1e-9
            slip_vel = np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).velocity()['vx_chunkZ_mod'][0])

            pGrad.append(np.absolute(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['pGrad']))
            mflowrate_avg.append(np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).mflux(self.mf)['mflowrate_stable']))
            shear_rate.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(pd=1)['shear_rate'])
            mu.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(pd=1)['mu'])    # mPa.s
            bulk_den_avg.append(np.mean(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).density(self.mf)['den_t']))


            # Continuum qtts
            flowrate_cont = (bulk_den_avg[i]*1e3 * self.Ly*1e-9 * 1e-9 * 1e3 * pGrad[i]*1e6*1e9 * avg_gap_height**3)  / \
                        (12 * mu[i]*1e-3)

            mflowrate_hp.append(flowrate_cont)
            mflowrate_hp_slip.append(flowrate_cont + (bulk_den_avg[i]*1e6 *  \
                                self.Ly*1e-9 * slip_vel * avg_gap_height*1e-9) )

        self.ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        self.ax.plot(pGrad, np.array(mflowrate_avg)*1e18, ls=' ', marker='x', alpha=opacity, label=input('Label:'))
        self.ax.plot(pGrad, np.array(mflowrate_hp)*1e18, ls='--', marker=' ', alpha=opacity, label='HP (no-slip)')
        self.ax.plot(pGrad, np.array(mflowrate_hp_slip)*1e18, ls='-', marker=' ', alpha=opacity, label='HP (slip)')

        ax2 = self.ax.twiny()
        ax2.set_xscale('log', nonpositive='clip')
        ax2.plot(shear_rate, mflowrate_avg, ls= ' ', marker= ' ', alpha=opacity, color=colors[0])
        ax2.set_xlabel(labels[11])

        if legend is not None:
            self.ax.legend(frameon=False)


    def pgrad_viscosity(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[12])

        mpl.rcParams.update({'lines.markersize': 4})

        pGrad, mu, shear_rate = [], [], []

        for i in range(len(self.datasets_x)):
            pGrad.append(np.absolute(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['pGrad']))
            # if couette is not None:
            #     shear_rate.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(couette=1)['shear_rate'])
            #     mu.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).transport(couette=1)['mu'])
            # if pd is not None:
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
            self.ax.legend(frameon=False)


    def rate_viscosity(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel(labels[11])
        self.ax.set_ylabel(labels[12])
        self.ax.set_xscale('log', nonpositive='clip')
        # self.ax.set_yscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.linewidth': 1})
        # mpl.rcParams.update({'lines.markersize': 4})


        shear_rate_couette, viscosity_couette = [], []
        shear_rate_couette_lo, viscosity_couette_lo = [], []
        shear_rate_couette_hi, viscosity_couette_hi = [], []
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
                shear_rate_couette_lo.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['shear_rate_lo'])
                shear_rate_couette_hi.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['shear_rate_hi'])

                viscosity_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['mu'])
                viscosity_couette_lo.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['mu_lo'])
                viscosity_couette_hi.append(dd(self.skip,couette_ds_x[i], couette_ds_z[i]).transport(couette=1)['mu_hi'])

            shear_rate_couette_err = np.array(shear_rate_couette_hi) - np.array(shear_rate_couette_lo)
            viscosity_couette_err = np.array(viscosity_couette_hi)- np.array(viscosity_couette_lo)

            # self.ax.plot(shear_rate_couette, viscosity_couette, ls= '-', marker='o', alpha=opacity, label=input('Label:'))

            markers_err, caps, bars= self.ax.errorbar(shear_rate_couette, viscosity_couette,
                xerr=shear_rate_couette_err,
                yerr=viscosity_couette_err,
                label='Couette',
                ls = ' ',
                capsize=1.5, marker='x', markersize=3, alpha=opacity)
            [bar.set_alpha(0.4) for bar in bars]
            [cap.set_alpha(0.4) for cap in caps]

            popt, pcov = curve_fit(funcs.power, shear_rate_couette, viscosity_couette)
            print(popt)
            self.ax.plot(shear_rate_couette, funcs.power(shear_rate_couette, *popt), 'k--')

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate'])
                shear_rate_pd_lo.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate_lo'])
                shear_rate_pd_hi.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate_hi'])

                viscosity_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['mu'])
                viscosity_pd_lo.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['mu_lo'])
                viscosity_pd_hi.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['mu_hi'])

            shear_rate_pd_err = np.array(shear_rate_pd_hi) - np.array(shear_rate_pd_lo)
            viscosity_pd_err = np.array(viscosity_pd_hi)- np.array(viscosity_pd_lo)

            # self.ax.plot(shear_rate_pd, viscosity_pd, ls= '--', marker='o', alpha=opacity, label=input('Label:'))

            markers_err, caps, bars= self.ax.errorbar(shear_rate_pd, viscosity_pd,
                xerr=shear_rate_pd_err,
                yerr=viscosity_pd_err,
                label='Poiseuille',
                ls = ' ',
                capsize=1.5, marker='o' , markersize=3, alpha=opacity)
            [bar.set_alpha(0.4) for bar in bars]
            [cap.set_alpha(0.4) for cap in caps]

            popt, pcov = curve_fit(funcs.power, shear_rate_pd, viscosity_pd)
            print(popt)
            self.ax.plot(shear_rate_pd, funcs.power(shear_rate_pd, *popt), 'k--')

        # coeffs = np.polyfit(np.log(shear_rate_pd), np.log(viscosity_pd), 1)
        # polynom = np.poly1d(coeffs)
        # y = polynom(shear_rate_pd)
        # self.ax.plot(shear_rate_pd, y, ls='--', color='k', label='Carreau')

        if legend is not None:
            handles, labs = self.ax.get_legend_handles_labels()
            legend_elements = [Line2D([0], [0], color='k', ls='--', marker=' ', label='Carreau')]
            handles.extend(legend_elements)
            self.ax.legend(handles=handles, frameon=False)
            # self.ax.legend(frameon=False)


    def rate_stress(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel(labels[11])
        self.ax.set_ylabel('$\sigma_{xz}$ (MPa)')
        self.ax.set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 6})

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
            self.ax.plot(shear_rate_couette, stress_couette, ls= ' ', marker='o', alpha=opacity, label='Couette')

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate'])
                stress_pd.append(np.mean(dd(self.skip, pump_ds_x[i], pump_ds_z[i]).sigwall(pd=1)['sigxz_t']))
            self.ax.plot(shear_rate_pd, stress_pd, ls= ' ', marker='x', alpha=opacity, label='Poiseuille')

        popt, pcov = curve_fit(funcs.power, shear_rate_pd, stress_pd)
        print(popt)
        self.ax.plot(shear_rate_pd, funcs.power(shear_rate_pd, *popt), 'k--', lw=1)

        popt, pcov = curve_fit(funcs.power, shear_rate_couette, stress_couette)
        print(popt)
        self.ax.plot(shear_rate_couette, funcs.power(shear_rate_couette, *popt), 'k--', lw=1, label="Carreau")

        if legend is not None:
            self.ax.legend(frameon=False)


    def rate_slip(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel(labels[11])
        self.ax.set_ylabel('Slip Length (nm)')
        self.ax.set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 6})

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
            self.ax.plot(shear_rate_couette, slip_couette, ls= '-', marker='o', alpha=opacity, label='Couette')

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).transport(pd=1)['shear_rate'])
                slip_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i]).slip_length(pd=1)['Ls'])
            self.ax.plot(shear_rate_pd, slip_pd, ls= '--', marker='x', alpha=opacity, label='Poiseuille')

        if legend is not None:
            self.ax.legend(frameon=False)


    def pt_ratio(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel('log$(P_{2}/P_{1})$')
        self.ax.set_ylabel('log$(T_{2}/T_{1})$')
        # self.ax.set_xscale('log', nonpositive='clip')
        # self.ax.set_yscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 4})

        press_ratio, temp_ratio = [], []

        for i in range(len(self.datasets_x)):
            press_ratio.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).virial()['p_ratio'])
            temp_ratio.append(dd(self.skip, self.datasets_x[i], self.datasets_z[i]).temp()['temp_ratio'])

        self.ax.plot(np.log(press_ratio), np.log(temp_ratio), ls= '-', marker='o', alpha=opacity, label=input('Label:'))

        coeffs = np.polyfit(np.log(press_ratio), np.log(temp_ratio), 1)
        print(f'Slope is {coeffs[0]}')
        # self.ax.plot(press_ratio, funcs.linear(press_ratio, coeffs[0], coeffs[1]),  ls= '--')

        if legend is not None:
            self.ax.legend(frameon=False)


    def struc_factor(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        kx = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['kx']
        ky = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['ky']

        sfx = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['sf_x']
        sfy = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['sf_y']

        sf = dd(self.skip, self.datasets_x[0], self.datasets_z[0]).dsf()['sf']

        if self.plot_type=='2d':
            a = input('x or y:')
            if a=='x':
                self.ax.set_xlabel('$k_x (\AA^{-1})$')
                self.ax.set_ylabel('$S(K_x)$')
                self.ax.plot(kx, sfx/np.max(sfx), ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
            elif a=='y':
                self.ax.set_xlabel('$k_y (\AA^{-1})$')
                self.ax.set_ylabel('$S(K_y)$')
                self.ax.plot(ky, sfy/np.max(sfy), ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
        else:
            self.ax.set_xlabel('$k_x (\AA^{-1})$')
            self.ax.set_ylabel('$k_y (\AA^{-1})$')
            self.ax.set_zlabel('$S(k)$')
            self.ax.invert_xaxis()
            # self.ax.invert_zaxis()
            self.ax.set_ylim(ky[-1]+1,0)
            self.ax.zaxis.set_rotate_label(False)
            self.ax.set_zticks([])

            Kx, Ky = np.meshgrid(kx, ky)
            self.ax.plot_surface(Kx, Ky, sf.T/np.max(sf.T), cmap=cmx.jet,
                        rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
            # self.fig.colorbar(surf, shrink=0.5, aspect=5)
            self.ax.view_init(35,60)
            # self.ax.view_init(0,90)

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
            self.ax.legend(frameon=False)

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
            self.ax.legend(frameon=False)


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
