#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cProfile
import re
import netCDF4
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
from matplotlib.gridspec import GridSpec

import scipy.constants as sci
from scipy.stats import iqr
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from get_variables_211018 import derive_data as dd

import yaml

color_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'seagreen','darkslategrey']

linestyles= {"line":"-", "dashed":"--", "dashdot":"-."}

opacities= {"transperent":0.3, "intermed":0.6, "opaque":0.9}

#           0             1                   2                    3
labels=('Height (nm)','Length (nm)', 'Time (ns)', r'Density (g/${\mathrm{cm^3}}$)',
#           4                                    5             6                    7
        r'${\mathrm{j_x}}$ (g/${\mathrm{m^2}}$.ns)', 'Vx (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\mathrm{(F_x)}}$ (pN)', r'${\mathrm{\partial p / \partial x}}$ (MPa/nm)',
#           10                                          11
        r'${\mathrm{\dot{m}}} \times 10^{-18}}$ (g/ns)', r'${\mathrm{\dot{\gamma}} (s^{-1})}$',
#           12
        r'${\mathrm{\mu}}$ (mPa.s)')

sublabels=('(a)', '(b)', '(c)', '(d)')

plt.style.use('imtek')
# mpl.rcParams.update({'axes.prop_cycle': color_cycler})

class plot_from_txt:

    def plot_from_txt(self, skip, txtfiles, outfile, lt='-', mark='o', opacity=1.0):

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

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

    def __init__(self, skip, datasets_x, datasets_z, mf, configfile, pumpsize):

        self.skip=skip
        self.mf=mf
        self.datasets_x=datasets_x
        self.datasets_z=datasets_z
        self.configfile = configfile
        self.pumpsize = pumpsize

        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)

        self.nsubplots = self.config['nsubplots']
        # self.ax_format = None#self.config['ax']

        if self.nsubplots==1:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1, sharex=True)
            self.ax = np.array(self.ax)
            self.axes_array = self.ax.reshape(-1)
            self.axes_array[0].xaxis.set_ticks_position('both')
            self.axes_array[0].yaxis.set_ticks_position('both')

        elif self.nsubplots > 1:
            self.fig, self.ax = plt.subplots(nrows=self.nsubplots, ncols=1, sharex=True, figsize=(7,8))
            # adjust space between axes
            self.fig.subplots_adjust(hspace=0.05)
            # Make axes array
            self.axes_array = self.ax.reshape(-1)
            # Plot ticks on both sides
            for ax in self.axes_array:
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')

            if self.config['broken_axis'] is not None:
                # Hide the bottom spines and ticks of all the axes except the last (bottom) one
                for ax in self.axes_array[:-2]:
                    ax.spines.bottom.set_visible(False)
                    ax.tick_params(labeltop=False, bottom=False)  # don't put tick labels at the bottom
                # Hide the top spines and ticks of all the axes except the first (top) one
                for ax in self.axes_array[1:-1]:
                    ax.spines.top.set_visible(False)
                    ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top

        elif self.config['3d']:
            self.fig, self.ax = plt.figure(dpi=1200), plt.axes(projection='3d')
            self.ax.xaxis.set_ticks_position('both')
            self.ax.yaxis.set_ticks_position('both')

        try:
            self.Nx = len(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).length_array)
            self.Nz = len(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).height_array)
            self.time = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).time
            self.Ly = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).Ly

        except IndexError:
            print("Dataset directory is not correctly set!")
            exit()


    def label_inline(self, lines):

        xpos = self.config['Label_x-pos']

        for i, line in enumerate(lines):
            if line.get_linestyle() == 'pop' or line.get_label() == ' ' or line.get_label() is None: #.startswith('_'):
                pass
            else:
                rot = self.config[f'Rotation_of_label_{i}']
                y_offset = self.config[f'Y-offset_for_label_{i}']
                label_lines.label_line(line, xpos, yoffset= y_offset, \
                         label=line.get_label(), fontsize= 14, rotation= rot)

    def label_subplot(self, plot):
        if self.config['broken_axis'] is None:
            for i, ax in enumerate(plot.axes_array):
                plot.axes_array[i].text(-0.1, 1.1, sublabels[i], transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='top', ha='right')

        else: # TODO: Generalize later
            plot.axes_array[0].text(-0.1, 1.1, sublabels[0], transform=plot.axes_array[0].transAxes,
                        fontsize=16, fontweight='bold', va='top', ha='right')
            plot.axes_array[3].text(-0.1, 1.1, sublabels[1], transform=plot.axes_array[3].transAxes,
                        fontsize=16, fontweight='bold', va='top', ha='right')

    def plot_vlines(self, plot):

        total_length = np.max(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).Lx)
        # Draw vlines only if the position was given in the  yaml file
        pos1 = self.config['vertical_line_pos_1']
        pos2 = self.config['vertical_line_pos_2']

        for ax in range(len(plot.axes_array)):
            plot.axes_array[ax].axvline(x= pos1*total_length, color='k', marker=' ', linestyle='dotted', lw=1.5)
            plot.axes_array[ax].axvline(x= pos2*total_length, color='k', marker=' ', linestyle='dotted', lw=1.5)

    def add_legend(self, ax):

        #where some data has already been plotted to ax
        handles, labs = ax.get_legend_handles_labels()
        #Additional elements
        # TODO: Generalize
        legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls=' ', marker='^', label='Fixed Force'),
                           Line2D([0], [0], color='k', lw=2.5, ls=' ', marker='v', label='Fixed Current')]
                           #Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Quadratic fit')]
        handles.extend(legend_elements)

        if self.config['legend_elements']=='h+e':
            # Extend legend elements to already existing ones (from labels)
            ax.legend(handles=handles, frameon=False)
        elif self.config['legend_elements']=='e':
            # Only legend elements specified
            if self.config['legend_loc'] is not None:
                 ax.legend(handles=legend_elements, frameon=False, loc=(0.,0.35))
            else:
                ax.legend(handles=legend_elements, frameon=False)
        elif self.config['legend_elements']=='h':
            # Only the existing ones (from labels)
            ax.legend(frameon=False)

    def plot_err_caps(self, ax, x, y, err):

        markers, caps, bars= ax.errorbar(x, y, yerr=err,
                                    capsize=1.5, markersize=1, lw=2, alpha=opacity)
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]

    def plot_err_fill(self, ax, x, lo, hi):
        ## TODOL color of the line
        ax.fill_between(x, lo, hi, alpha=0.4)

    def plot_inset(self, plot, xpos=0.62, ypos=0.57, w=0.2, h=0.28):

        inset_ax = self.fig.add_axes([xpos, ypos, w, h]) # X, Y, width, height
        # TODO : Generalize
        # inset_ax.axvline(x=0, color='k', linestyle='dashed')
        # inset_ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
        # inset_ax.set_ylim(220, 280)
        inset_ax.plot(dd(self.skip, self.datasets_x[1], self.datasets_z[1], self.mf).length_array[1:29],
                      dd(self.skip, self.datasets_x[1], self.datasets_z[1], self.mf).virial()['vir_chunkX'][1:29],
                      ls= plot.ax.lines[1].get_linestyle(), color=plot.ax.lines[1].get_color(),
                      marker=None, alpha=1)

    def set_ax_height(self, pt):
        gs = GridSpec(len(pt.axes_array), 1, height_ratios=[1,1,1,2])
        for i, ax in enumerate(pt.axes_array):
            ax.set_position(gs[i].get_position(pt.fig))


    def plot_broken(self, pt):

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        # Draw the dashes
        pt.axes_array[0].plot([0, 1], [0, 0], transform=pt.axes_array[0].transAxes, **kwargs)
        pt.axes_array[1].plot([0, 1], [1, 1], transform=pt.axes_array[1].transAxes, **kwargs)

        pt.axes_array[1].plot([0, 1], [0, 0], transform=pt.axes_array[1].transAxes, **kwargs)
        pt.axes_array[2].plot([0, 1], [1, 1], transform=pt.axes_array[2].transAxes, **kwargs)

        # Remove all axes label
        for ax in pt.axes_array:
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)

        # Set the common labels # TODO : generalize the y-axis labels
        pt.fig.text(0.5, 0.04, pt.axes_array[-1].get_xlabel(), ha='center', size=14)
        pt.fig.text(0.04, 0.67, pt.axes_array[0].get_ylabel(), va='center', rotation='vertical', size=14)
        pt.fig.text(0.04, 0.25, pt.axes_array[-1].get_ylabel(), va='center', rotation='vertical', size=14)


    def plot_settings(self, ax, x, y, arr, arr2=None):
        n = 100 # samples/block
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        if self.config['err_caps'] is None and self.config['dim']=='l':
            ax.plot(x, y[1:-1])

        if self.config['err_caps'] is None and self.config['dim']=='h':
            ax.plot(x[y!=0], y[y!=0])

        if self.config['err_caps'] is None and self.config['dim']=='t':
            ax.plot(x, y)
            ax.axhline(y=np.mean(y))

        if self.config['err_caps'] is not None and self.config['dim'] == 'l' and arr2 is None:
            blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nx, n)
            err = sq.get_err(blocks)['uncertainty'][1:-1]
            pds.plot_err_caps(ax, x ,y ,err)

        if self.config['err_caps'] is not None and self.config['dim'] == 'h' and arr2 is None:
            blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nz, n)
            err = sq.get_err(blocks)['uncertainty'][y!=0]
            pds.plot_err_caps(ax, x ,y ,err)

        if self.config['err_fill'] is not None and self.config['dim'] == 'l' and arr2 is None:
            blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nx, n)
            lo, hi = sq.get_err(blocks)['lo'][1:-1], sq.get_err(blocks)['hi'][1:-1]
            pds.plot_err_fill(ax, x ,lo ,hi)

        if self.config['err_fill'] is not None and self.config['dim'] == 'h' and arr2 is None:
            blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nz, n)
            lo, hi = sq.get_err(blocks)['lo'][y!=0], sq.get_err(blocks)['hi'][y!=0]
            pds.plot_err_fill(ax, x ,lo ,hi)

        # Wall stresses
        if self.config['err_caps'] is not None and arr2 is not None:
            blocks1 = sq.block_ND(len(self.time[self.skip:]), arr, self.Nx, n)
            blocks2 = sq.block_ND(len(self.time[self.skip:]), arr2, self.Nx, n)
            err = sq.prop_uncertainty(blocks1,blocks2)['uncertainty'][1:-1]
            pds.plot_err_caps(ax, x ,y ,err)

        if self.config['err_fill'] is not None and arr2 is not None:
            blocks1 = sq.block_ND(len(self.time[self.skip:]), arr, self.Nx, n)
            blocks2 = sq.block_ND(len(self.time[self.skip:]), arr2, self.Nx, n)
            lo = sq.prop_uncertainty(blocks1,blocks2)['lo'][1:-1]
            hi = sq.prop_uncertainty(blocks1,blocks2)['hi'][1:-1]
            pds.plot_err_caps(ax, x ,y ,err)


    def qtty_dim(self, *arr_to_plot, **kwargs):

        qtts = arr_to_plot[0]
        lines = []  # list to append all the lines to
        datasets = self.datasets_x
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        # Label the bottom axis
        if self.config['dim'] == 'l': self.axes_array[-1].set_xlabel(labels[1])
        if self.config['dim'] == 'h': self.axes_array[-1].set_xlabel(labels[0])

        mpl.rcParams.update({'lines.linewidth': 2})
        mpl.rcParams.update({'lines.markersize': 6})

        # The for loop plots all the quantities needed for each dataset then moves
        # on to the next dataset. The quantites can be plotted on the same subplot
        # if nsubplots variable is set to 1. OR they can be plotted separately.
        for i in range(len(datasets)):
            n=0    # subplot
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            if self.config['dim'] == 'l': x = data.length_array[1:-1]   # nm
            if self.config['dim'] == 'h': x = data.height_array      # nm
            if self.config['dim'] == 't': x = self.time * 1e-6      # ns

            if 'vx_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[5])
                if self.config['dim'] == 'l':
                    arr, y = data.velocity()['vx_full_x'], data.velocity()['vx_X']
                if self.config['dim'] == 'h':
                    arr, y = data.velocity()['vx_full_z'], data.velocity()['vx_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.velocity()['vx_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                # Fitting the data
                if self.config[f'fit'] is not None and self.config['dim'] == 'h':
                    fit_data = funcs.fit(x[y!=0][1:-1] ,y[y!=0][1:-1], self.config[f'fit'])['fit_data']
                    self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                if self.config['extrapolate'] is not None:
                    x_left = data.slip_length()['xdata_left']
                    y_left = data.slip_length()['extrapolate_left']
                    ax.plot(x_left, y_left, marker=' ', ls='--', color='k')
                    # ax.set_xlim([data.slip_length()['root_left'], 0.5*np.max(x)])
                    # ax.set_ylim([0, 1.1*np.max(y)])
                if self.nsubplots>1: n+=1

                if self.config['broken_axis'] is not None:
                    while n<self.nsubplots-1: # plot the same data on all the axes except the last one
                        self.axes_array[n].plot(x[y!=0], y[y!=0])
                        if self.config[f'fit'] is not None: self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                        n+=1

            if 'jx_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[4])
                if self.config['dim'] == 'l':
                    arr, y = data.mflux()['jx_full_x'], data.mflux()['jx_X']
                if self.config['dim'] == 'h':
                    arr, y = data.mflux()['jx_full_z'], data.mflux()['jx_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.mflux()['jx_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'den_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[3])
                if self.config['dim'] == 'l':
                    arr, y = data.density()['den_full_x'], data.density()['den_X']
                if self.config['dim'] == 'h':
                    arr, y = data.density()['den_full_z'], data.density()['den_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.density()['den_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'virxy_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wxy_full_x'], data.virial()['Wxy_X']
                if self.config['dim'] == 'h':
                    x = data.bulk_height_array
                    arr, y = data.virial()['Wxy_full_z'], data.virial()['Wxy_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wxy_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'virxz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wxz_full_x'], data.virial()['Wxz_X']
                if self.config['dim'] == 'h':
                    x = data.bulk_height_array
                    arr, y = data.virial()['Wxz_full_z'], data.virial()['Wxz_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wxz_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'viryz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wyz_full_x'], data.virial()['Wyz_X']
                if self.config['dim'] == 'h':
                    x = data.bulk_height_array
                    arr, y = data.virial()['Wyz_full_z'], data.virial()['Wyz_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wyz_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'vir_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['vir_full_x'], data.virial()['vir_X']
                if self.config['dim'] == 'h':
                    x = data.bulk_height_array
                    arr, y = data.virial()['vir_full_z'], data.virial()['vir_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['vir_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'sigzz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, arr2, y = data.sigwall()['sigzzU_full'],\
                                   data.sigwall()['sigzzL_full'], data.sigwall()['sigzz_X']
                if self.config['dim'] == 't':
                    arr, y = None, data.sigwall()['sigzz_t']
                pds.plot_settings(self.axes_array[n], x, y, arr, arr2)
                if self.nsubplots>1: n+=1

            if 'sigxz_dim' in qtts:
                self.axes_array[n].set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                if self.config['dim'] == 'l':
                    arr, arr2, y = data.sigwall()['sigxzU_full'],\
                                   data.sigwall()['sigxzL_full'], data.sigwall()['sigxz_X']
                if self.config['dim'] == 't':
                    arr, y = None, data.sigwall()['sigxz_t']
                pds.plot_settings(self.axes_array[n], x, y, arr, arr2)
                if self.nsubplots>1: n+=1

            if 'temp_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[6])
                if self.config['dim'] == 'l':
                    arr, y = data.temp()['temp_full_x'], data.temp()['temp_X']
                if self.config['dim'] == 'h':
                    arr, y = data.temp()['temp_full_z'], data.temp()['temp_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.temp()['temp_t']
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

            if 'height_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[0])
                if self.config['dim'] == 't': arr, y = None, data.h
                pds.plot_settings(self.axes_array[n], x, y, arr)
                if self.nsubplots>1: n+=1

        # Make a list that contains all the lines in all the axes starting from the last axes.
        for ax in self.axes_array:
            lines.append(list(ax.get_lines()))

        lines = [item for sublist in lines for item in sublist]
        print(f'Lines on the figure: {len(lines)}')

        for i, line in enumerate(lines):
            if self.config[f'lstyle_{i}']=='pop':
                line.set_linestyle(' ')
                line.set_marker(' ')
                line.set_label(' ')
            else:
                line.set_linestyle(self.config[f'lstyle_{i}'])
                line.set_marker(self.config[f'mstyle_{i}'])
                line.set_color(self.config[f'color_{i}'])
                line.set_label(self.config[f'label_{i}'])
                line.set_alpha(self.config[f'alpha_{i}'])
            print(line.get_label())

        if self.config['legend_elements'] is not None:
            pds.add_legend(self.axes_array[self.config['legend_on_ax']])
        if self.config['Label_x-pos'] is not None:
            pds.label_inline(lines)
        if self.config['vertical_line_pos_1'] is not None:
            pds.plot_vlines(self)
        if self.config['plot_inset'] is not None:
            pds.plot_inset(self)
        if self.config['broken_axis'] is not None:
            pds.plot_broken(self)
        if self.config['label_subplot'] is not None:
            self.label_subplot(self)

        for i, ax in enumerate(self.axes_array):
            if self.config[f'xlo_{i}'] is not None:
                ax.set_xlim(left=self.config[f'xlo_{i}']*np.max(x))
            if self.config[f'xhi_{i}'] is not None:
                ax.set_xlim(right=self.config[f'xhi_{i}']*np.max(x))
            if self.config[f'ylo_{i}'] is not None:
                ax.set_ylim(bottom=self.config[f'ylo_{i}'])
            if self.config[f'yhi_{i}'] is not None:
                ax.set_ylim(top=self.config[f'yhi_{i}'])

        if self.config['set_ax_height'] is not None:
            pds.set_ax_height(self)

    def v_distrib(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel('Velocity (m/s)')
        self.ax.set_ylabel('Probability')

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, pumpsize)

        for i in range(len(self.datasets_x)):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            vx_values = data.vel_distrib()['vx_values']
            vx_prob = data.vel_distrib()['vx_prob']

            vy_values = data.vel_distrib()['vy_values']
            vy_prob = data.vel_distrib()['vy_prob']

            self.ax.plot(vx_values, vx_prob, ls='-', marker='o',
                    label=input('Label:'), alpha=opacity)
            # self.ax.plot(vy_values, vy_prob, ls='-', marker='x',
            #         label=input('Label:'), alpha=opacity)

        if legend is not None:
            self.ax.legend(frameon=False)
        if lab_lines is not None:
            pds.label_inline(self)


    def pdiff_pumpsize(self, opacity=1, legend=None, lab_lines=None, **kwargs):
        """
        Plot the virial and wall sigmazz with versus the pump size
        """

        self.ax.set_xlabel('Normalized pump length')

        pump_size = []
        vir_pdiff, sigzz_pdiff, vir_err, sigzz_err = [], [], [], []
        for i in range(len(self.datasets_x)):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            pump_size.append(self.pumpsize)
            vir_pdiff.append(data.virial(pump_size[i])['pDiff'])
            vir_err.append(data.virial(pump_size[i])['pDiff_err'])
            sigzz_pdiff.append(data.sigwall(pump_size[i])['pDiff'])
            sigzz_err.append(data.sigwall(pump_size[i])['pDiff_err'])

        markers_a, caps, bars= self.ax.errorbar(pump_size, vir_pdiff, yerr=vir_err, ls=lt, fmt=mark,
                label='Virial (Fluid)', capsize=1.5, markersize=1.5, alpha=1)
        markers2, caps2, bars2= self.ax.errorbar(pump_size, sigzz_pdiff, yerr=sigzz_err, ls=lt, fmt='x',
                label='$\sigma_{zz}$ (Solid)', capsize=1.5, markersize=3, alpha=1)

        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        [bar2.set_alpha(0.5) for bar2 in bars2]
        [cap2.set_alpha(0.5) for cap2 in caps2]


    def pgrad_mflowrate(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        # Get the error in mflowrate in MD
        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[10])

        mpl.rcParams.update({'lines.markersize': 6})
        mpl.rcParams.update({'figure.figsize': (12,12)})

        pGrad, mflowrate_avg, shear_rate, mu, bulk_den_avg, mflowrate_hp, mflowrate_hp_slip = [], [], [], [], [], [], []

        for i in range(len(self.datasets_x)):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)

            avg_gap_height = np.mean(data.h)*1e-9
            slip_vel = np.mean(data.velocity()['vx_chunkZ_mod'][0])

            pGrad.append(np.absolute(data.virial()['pGrad']))
            mflowrate_avg.append(np.mean(data.mflux(self.mf)['mflowrate_stable']))
            # mflowrate_err.append(np.(data.mflux(self.mf)['mflowrate_stable']))
            shear_rate.append(data.transport(pd=1)['shear_rate'])
            mu.append(data.transport(pd=1)['mu'])    # mPa.s
            bulk_den_avg.append(np.mean(data.density(self.mf)['den_t']))


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


    def pgrad_viscosity(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.ax.set_xlabel(labels[9])
        self.ax.set_ylabel(labels[12])

        mpl.rcParams.update({'lines.markersize': 4})

        pGrad, mu, shear_rate = [], [], []

        for i in range(len(self.datasets_x)):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)

            pGrad.append(np.absolute(data.virial()['pGrad']))
            shear_rate.append(data.transport(pd=1)['shear_rate'])
            mu.append(data.transport(pd=1)['mu'])

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
                shear_rate_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['shear_rate'])
                shear_rate_couette_lo.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['shear_rate_lo'])
                shear_rate_couette_hi.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['shear_rate_hi'])

                viscosity_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['mu'])
                viscosity_couette_lo.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['mu_lo'])
                viscosity_couette_hi.append(dd(self.skip,couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['mu_hi'])

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
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['shear_rate'])
                shear_rate_pd_lo.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['shear_rate_lo'])
                shear_rate_pd_hi.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['shear_rate_hi'])

                viscosity_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['mu'])
                viscosity_pd_lo.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['mu_lo'])
                viscosity_pd_hi.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['mu_hi'])

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
                shear_rate_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['shear_rate'])
                stress_couette.append(np.mean(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).sigwall(couette=1)['sigxz_t']))
            self.ax.plot(shear_rate_couette, stress_couette, ls= ' ', marker='o', alpha=opacity, label='Couette')

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['shear_rate'])
                stress_pd.append(np.mean(dd(self.skip, pump_ds_x[i], pump_ds_z[i], self.mf).sigwall(pd=1)['sigxz_t']))
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
                shear_rate_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).transport(couette=1)['shear_rate'])
                slip_couette.append(dd(self.skip, couette_ds_x[i], couette_ds_z[i], self.mf).slip_length(couette=1)['Ls'])
            self.ax.plot(shear_rate_couette, slip_couette, ls= '-', marker='o', alpha=opacity, label='Couette')

        if pump_ds_x:
            for i in range(len(pump_ds_x)):
                shear_rate_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).transport(pd=1)['shear_rate'])
                slip_pd.append(dd(self.skip, pump_ds_x[i],  pump_ds_z[i], self.mf).slip_length(pd=1)['Ls'])
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
            press_ratio.append(data.virial()['p_ratio'])
            temp_ratio.append(data.temp()['temp_ratio'])

        self.ax.plot(np.log(press_ratio), np.log(temp_ratio), ls= '-', marker='o', alpha=opacity, label=input('Label:'))

        coeffs = np.polyfit(np.log(press_ratio), np.log(temp_ratio), 1)
        print(f'Slope is {coeffs[0]}')
        # self.ax.plot(press_ratio, funcs.linear(press_ratio, coeffs[0], coeffs[1]),  ls= '--')

        if legend is not None:
            self.ax.legend(frameon=False)


    def press_temp(self, opacity=1, legend=None, lab_lines=None, couette=None, pd=None, **kwargs):

        self.ax.set_xlabel('log$(P_{2}/P_{1})$')
        self.ax.set_ylabel('log$(T_{2}/T_{1})$')
        # self.ax.set_xscale('log', nonpositive='clip')
        # self.ax.set_yscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 4})

        data = pds.

        press_ratio, temp_ratio = [], []

        p = data.virial()['vir_X']

        for i in range(len(self.datasets_x)):
            press_ratio.append(data.virial()['p_ratio'])
            temp_ratio.append(data.temp()['temp_ratio'])

        self.ax.plot(np.log(press_ratio), np.log(temp_ratio), ls= '-', marker='o', alpha=opacity, label=input('Label:'))

        coeffs = np.polyfit(np.log(press_ratio), np.log(temp_ratio), 1)
        print(f'Slope is {coeffs[0]}')
        # self.ax.plot(press_ratio, funcs.linear(press_ratio, coeffs[0], coeffs[1]),  ls= '--')

        if legend is not None:
            self.ax.legend(frameon=False)





    def struc_factor(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        kx = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['kx']
        ky = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['ky']
        # k = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['k']

        sfx = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['sf_x']
        sfy = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['sf_y']

        sf = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['sf']
        # sf_r = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['sf_r']
        sf_time = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).dsf()['sf_time']

        if self.ax_format=='single':
            a = input('x or y or r or t:')
            if a=='x':
                self.ax.set_xlabel('$k_x (\AA^{-1})$')
                self.ax.set_ylabel('$S(K_x)$')
                self.ax.plot(kx, sfx, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
            elif a=='y':
                self.ax.set_xlabel('$k_y (\AA^{-1})$')
                self.ax.set_ylabel('$S(K_y)$')
                self.ax.plot(ky, sfy, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
            elif a=='r':
                self.ax.set_xlabel('$k (\AA^{-1})$')
                self.ax.set_ylabel('$S(K)$')
                self.ax.plot(k, sf_r, ls= ' ', marker='x', alpha=opacity,
                           label=input('Label:'))
            elif a=='t':
                self.ax.set_xlabel('$t (fs)$')
                self.ax.set_ylabel('$S(K)$')
                self.ax.plot(self.time[:self.skip], sf_time, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
        else:
            self.ax.set_xlabel('$k_x (\AA^{-1})$')
            self.ax.set_ylabel('$k_y (\AA^{-1})$')
            self.ax.set_zlabel('$S(k)$')
            self.ax.invert_xaxis()
            self.ax.set_ylim(ky[-1]+1,0)
            self.ax.zaxis.set_rotate_label(False)
            self.ax.set_zticks([])

            Kx, Ky = np.meshgrid(kx, ky)
            self.ax.plot_surface(Kx, Ky, sf.T, cmap=cmx.jet,
                        rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
            # self.fig.colorbar(surf, shrink=0.5, aspect=5)
            self.ax.view_init(35,60)
            # self.ax.view_init(0,90)

        # self.ax.plot(freq, swx, ls= '-', marker='o', alpha=opacity,
        #             label=input('Label:'))


    def acf(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        mpl.rcParams.update({'lines.linewidth':'1'})

        self.ax.set_xlabel(labels[2])
        self.ax.set_ylabel(r'${\mathrm C_{AA}}$')

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
        self.ax.set_ylabel(r'${\mathrm C_{AA}}$')

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
