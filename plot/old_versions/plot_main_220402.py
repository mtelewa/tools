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
from matplotlib.ticker import ScalarFormatter

import scipy.constants as sci
from scipy.stats import iqr
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from get_variables import derive_data as dd

import yaml

import pickle

color_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf', 'seagreen','darkslategrey']

linestyles= {"line":"-", "dashed":"--", "dashdot":"-."}

opacities= {"transperent":0.3, "intermed":0.6, "opaque":0.9}

#           0             1                   2                    3
labels=('Height (nm)','Length (nm)', 'Time (ns)', r'Density (g/${\mathrm{cm^3}}$)',
#           4                                           5             6                    7
        r'${\mathrm{j_x}}$ (g/${\mathrm{m^2}}$.ns)', 'Vx (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\mathrm{(F_x)}}$ (pN)', r'${\mathrm{dP / dx}}$ (MPa/nm)',
#           10                                          11
        r'${\mathrm{\dot{m}}} \times 10^{-18}}$ (g/ns)', r'${\mathrm{\dot{\gamma}} (s^{-1})}$',
#           12
        r'${\mathrm{\eta}}$ (mPa.s)')

sublabels=('(a)', '(b)', '(c)', '(d)')

plt.style.use('imtek')
# mpl.rcParams.update({'axes.prop_cycle': color_cycler})

def plot_settings(pt, x, configfile):

    # Make a list that contains all the lines in all the axes starting from the last axes.
    # lines = []  # list to append all the lines to
    # for ax in pt.axes_array:
    lines = pt.ax.get_lines()
    # Read the yaml file
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)

    # lines = [item for sublist in lines for item in sublist]
    print(f'Lines on the figure: {len(lines)}')

    for i, line in enumerate(lines):
        if config[f'lstyle_{i}']=='pop':
            line.set_linestyle(' ')
            line.set_marker(' ')
            line.set_label(None)
        else:
            line.set_linestyle(config[f'lstyle_{i}'])
            line.set_marker(config[f'mstyle_{i}'])
            line.set_color(config[f'color_{i}'])
            line.set_label(config[f'label_{i}'])
            line.set_alpha(config[f'alpha_{i}'])
        print(line.get_label())

    if config['legend_elements'] is not None:
        pt.ax.legend(frameon=False) #pds.add_legend(pt.axes_array[config['legend_on_ax']])
    if config['xlo_0'] is not None:
        pt.ax.set_xlim(left=config[f'xlo_0'])
    if config['xhi_0'] is not None:
        pt.ax.set_xlim(right=config[f'xhi_0']*np.max(x))
    if config['ylo_0'] is not None:
        pt.ax.set_ylim(bottom=config[f'ylo_0'])
    if config['yhi_0'] is not None:
        pt.ax.set_ylim(top=config[f'yhi_0'])
    # if self.config['set_ax_height'] is not None: ptxt.set_ax_height(pt)


class plot_from_txt:

    def __init__(self, skip, txtfiles, configfile):

        self.skip = skip
        self.txts = txtfiles
        self.configfile = configfile
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, sharex=True)

        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)

    def plot_txt(self, *arr_to_plot):

        qtts = arr_to_plot[0]

        self.ax.xaxis.set_ticks_position('both')
        self.ax.yaxis.set_ticks_position('both')

        if 'eos' in qtts:
            x_data, y_data = [], []
            self.ax.set_xlabel(labels[7])
            self.ax.set_ylabel(labels[3])
            exp_density = [0.630, 0.653, 0.672, 0.686, 0.714, 0.739, 0.750]
            exp_press = [28.9, 55.3, 84.1, 110.2, 171.0, 239.5, 275.5]
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,10]
                ydata = data[:,12]
                print(np.mean(ydata) * sci.atm * 1e-6)
                x_data.append(np.mean(xdata))
                y_data.append(np.mean(ydata) * sci.atm * 1e-6)
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        if 'npump' in qtts:
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel(r'$N_{pump}$')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,13]
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        if 'fw' in qtts:
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel('$F$ (nN)')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,14] * 1e9     # nanoN 9
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        if 'fp' in qtts:
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel('$F$ (nN)')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,15] * 1e9     # nanoN   10
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        if 'fpump' in qtts:
            kcalpermolA_to_N = 6.9477e-11
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel(r'$F_{pump}$ (pN)')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,16] * kcalpermolA_to_N * 1e12     # picoN
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")


        if 'energy' in qtts:
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel(r'$E$ (Kcal/mol)')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,9] # Kcal/mol
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        if 'Fin' in qtts:
            kcalpermolA_to_N = 6.9477e-11
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel(r'Force (N)')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,17] * kcalpermolA_to_N # N
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        if 'Fout' in qtts:
            kcalpermolA_to_N = 6.9477e-11
            self.ax.set_xlabel(labels[2])
            self.ax.set_ylabel(r'Force (N)')
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,0] * 1e-6      # ns
                ydata = data[:,18] * kcalpermolA_to_N # N
                self.ax.plot(xdata, ydata)
                self.ax.axhline(y = np.mean(ydata))
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")


        if 'press_md-cont' in qtts:
            self.ax.set_xlabel(labels[1])
            self.ax.set_ylabel(labels[7])
            for idx, val in enumerate(self.txts):
                data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)
                xdata = data[:,0]      # ns
                ydata = data[:,1]
                self.ax.plot(xdata, ydata)

        if 'radius' in qtts:
            self.ax.set_xlabel('Time (ns)')
            self.ax.set_ylabel(r'$R(t)$ (${\AA}$)')
            for idx, val in enumerate(self.txts):
                data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)
                xdata = data[:,0]/1e6   #ns
                ydata = data[:,1]
                self.ax.plot(xdata, ydata)


        ptxt = plot_from_txt(self.skip, self.txts, self.configfile)
        try:
            plot_settings(self, xdata, self.configfile)
        except UnboundLocalError:
            print('No data on the x-axis, check the quanity to plot!')
            exit()



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

        self.nrows = self.config['nrows']
        self.ncols = self.config['ncols']

        if self.nrows==1 and self.ncols==1:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1, sharex=True) #,figsize=(4.6, 4.1))
            self.ax = np.array(self.ax)
            self.axes_array = self.ax.reshape(-1)
            self.axes_array[0].xaxis.set_ticks_position('both')
            self.axes_array[0].yaxis.set_ticks_position('both')

        if self.nrows > 1 or self.ncols > 1:
            if self.nrows>1:
                self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=True, figsize=(7,8))
                # self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=True)
                self.fig.subplots_adjust(hspace=0.05)
            if self.ncols>1:
                self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharey=True, figsize=(8,7))
                self.fig.subplots_adjust(wspace=0.05)
            # adjust space between axes
            # Make axes array
            self.axes_array = self.ax.reshape(-1)
            # Plot ticks on both sides
            for ax in self.axes_array:
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')

            if self.config['broken_axis'] is not None:
                # Hide the bottom spines and ticks of all the axes except the last (bottom) one
                # for ax in self.axes_array[:-2]:
                for ax in self.axes_array[:-1]:
                    ax.spines.bottom.set_visible(False)
                    ax.tick_params(labeltop=False, bottom=False)  # don't put tick labels at the bottom
                # Hide the top spines and ticks of all the axes except the first (top) one
                # for ax in self.axes_array[1:-1]:
                for ax in self.axes_array[1:]:
                    ax.spines.top.set_visible(False)
                    ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top

        if self.config['3d']:
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
                # xpos = self.config[f'Label_x-pos_{i}']
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
            last_axis = len(plot.axes_array)-1
            plot.axes_array[last_axis].text(-0.1, 1.1, sublabels[1], transform=plot.axes_array[last_axis].transAxes,
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
        # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls=' ', marker='^', label='Fixed Force'),
        #                    Line2D([0], [0], color='k', lw=2.5, ls=' ', marker='v', label='Fixed Current'),
        #                    Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Quadratic fit'),
        #                    Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Lin. extrapolation')]
        # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='$C\dot{\gamma}^{n}$')]
        legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Quartic fit')]
        # legend_elements = [Line2D([0], [0], color='tab:gray', lw=2.5, ls=' ', marker='s', label='Wall $\sigma_{xz}$')]
        # legend_elements = [Line2D([0], [0], color='k', lw=2, ls='--', marker=' ', label='Wall temp.')]

        handles.extend(legend_elements)

        if self.config['legend_elements']=='h+e':
            # Extend legend elements to already existing ones (from labels)
            ax.legend(handles=handles, frameon=False)
        elif self.config['legend_elements']=='e':
            # Only legend elements specified
            if self.config['legend_loc'] == 1:
                ax.legend(handles=legend_elements, frameon=False, loc=(0.,0.45))
            if isinstance(self.config['legend_loc'], str):
                ax.legend(handles=legend_elements, frameon=False, loc=self.config['legend_loc'])
            if self.config['legend_loc'] is None:
                ax.legend(handles=legend_elements, frameon=False)
        elif self.config['legend_elements']=='h':
            # Only the existing ones (from labels)
            ax.legend(frameon=False)

    def plot_err_caps(self, ax, x, y, xerr, yerr):
        markers, caps, bars= ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                                    capsize=3.5, markersize=4, lw=2, alpha=0.8)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]
        # [cap.set_markeredgewidth(10)]

    def plot_err_fill(self, ax, x, lo, hi, color=None):
        if color==None:
            ax.fill_between(x, lo, hi, alpha=0.4)
        else:
            ax.fill_between(x, lo, hi, color=color, alpha=0.4)

    def plot_inset(self, pt, xpos=0.64, ypos=0.28, w=0.23, h=0.17):
        inset_ax = pt.fig.add_axes([xpos, ypos, w, h]) # X, Y, width, height
        inset_ax.set_xlim(right=0.2*np.max(dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize).length_array))
        inset_ax.set_xlabel('Length (nm)')
        inset_ax.set_ylabel('Pressure (MPa)')
        # for i in range(len(self.datasets_x)):
        # TODO : Generalize, data0 is wetting and data4 is non-wetting
        data0 = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        # data2 = dd(self.skip, self.datasets_x[2], self.datasets_z[2], self.mf, self.pumpsize)
        data4 = dd(self.skip, self.datasets_x[4], self.datasets_z[4], self.mf, self.pumpsize)
        # print(self.datasets_x[4])
        inset_ax.plot(data0.length_array[1:-1], data0.virial()['vir_X'][1:-1], color='tab:blue')
        # inset_ax.plot(data2.length_array[1:-1], data2.virial()['vir_X'][1:-1])
        inset_ax.plot(data4.length_array[1:-1], data4.virial()['vir_X'][1:-1], color='tab:green')
        inset_ax.plot(data4.length_array[1:-1], data4.sigwall()['sigzz_X'][1:-1], ls='--', color='k')

        lo0, hi0 = sq.get_err(data0.virial()['vir_full_x'])['lo'][1:-1], sq.get_err(data0.virial()['vir_full_x'])['hi'][1:-1]
        lo5, hi5 = sq.get_err(data4.virial()['vir_full_x'])['lo'][1:-1], sq.get_err(data4.virial()['vir_full_x'])['hi'][1:-1]

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)
        pds.plot_err_fill(inset_ax, data0.length_array[1:-1], lo0, hi0, color='tab:blue')
        pds.plot_err_fill(inset_ax, data4.length_array[1:-1], lo5, hi5, color='tab:green')

    def set_ax_height(self, pt):
        #gs = GridSpec(len(pt.axes_array), 1, height_ratios=[1,1,1,2])
        list = np.ones(len(pt.axes_array))
        list[-1] = 2
        gs = GridSpec(len(pt.axes_array), 1, height_ratios=list)
        for i, ax in enumerate(pt.axes_array):
            ax.set_position(gs[i].get_position(pt.fig))


    def plot_broken(self, pt, shared_label=None):

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        # Draw the dashes
        pt.axes_array[0].plot([0, 1], [0, 0], transform=pt.axes_array[0].transAxes, **kwargs)
        pt.axes_array[1].plot([0, 1], [1, 1], transform=pt.axes_array[1].transAxes, **kwargs)

        # pt.axes_array[1].plot([0, 1], [0, 0], transform=pt.axes_array[1].transAxes, **kwargs)
        # pt.axes_array[2].plot([0, 1], [1, 1], transform=pt.axes_array[2].transAxes, **kwargs)

        # Remove all axes label
        for ax in pt.axes_array:
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)

        if shared_label:
            pt.fig.supylabel(pt.axes_array[0].get_ylabel())
        else:
            #Set the common labels # TODO : generalize the y-axis labels and their positions
            pt.fig.text(0.5, 0.04, pt.axes_array[-1].get_xlabel(), ha='center', size=14)
            pt.fig.text(0.04, 0.70, pt.axes_array[0].get_ylabel(), va='center', rotation='vertical', size=14)
            pt.fig.text(0.04, 0.30, pt.axes_array[-1].get_ylabel(), va='center', rotation='vertical', size=14)

    def ax_settings(self, ax, x, y, arr):
        n = 100 # samples/block
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        if self.config['err_caps'] is None and self.config['dim']=='l':
            ax.plot(x, y[1:-1])

        if self.config['err_caps'] is None and self.config['dim']=='h':
            ax.plot(x[y!=0][2:-2], y[y!=0][2:-2])

        if self.config['err_caps'] is None and self.config['dim']=='t':
            ax.plot(x, y)
            ax.axhline(y=np.mean(y))

        if self.config['err_caps'] is not None and self.config['dim'] == 'l' and arr.ndim==3:
            # blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nx, n)
            err = sq.get_err(arr)['uncertainty'][1:-1]
            pds.plot_err_caps(ax, x[1:-1] , y[1:-1], None, err)

        if self.config['err_caps'] is not None and self.config['dim'] == 'h' and arr.ndim==3:
            # blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nz, n)
            err = sq.get_err(arr)['uncertainty'][y!=0]
            pds.plot_err_caps(ax, x[y!=0] ,y[y!=0], None, err)

        if self.config['err_fill'] is not None and self.config['dim'] == 'l' and arr.ndim==3:
            # blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nx, n)
            lo, hi = sq.get_err(arr)['lo'][1:-1], sq.get_err(arr)['hi'][1:-1]
            pds.plot_err_fill(ax, x ,lo ,hi)

        if self.config['err_fill'] is not None and self.config['dim'] == 'h' and arr.ndim==3:
            # blocks = sq.block_ND(len(self.time[self.skip:]), arr, self.Nz, n)
            lo, hi = sq.get_err(arr)['lo'][y!=0], sq.get_err(arr)['hi'][y!=0]
            pds.plot_err_fill(ax, x[y!=0] ,lo ,hi)

    def plot_settings(self, pt, x):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        # Make a list that contains all the lines in all the axes starting from the last axes.
        lines = []  # list to append all the lines to
        for ax in pt.axes_array:
            lines.append(list(ax.get_lines()))

        lines = [item for sublist in lines for item in sublist]
        print(f'Lines on the figure: {len(lines)}')

        for i, line in enumerate(lines):
            if self.config[f'lstyle_{i}']=='pop':
                line.set_linestyle(' ')
                line.set_marker(' ')
                line.set_label(None)
            else:
                line.set_linestyle(self.config[f'lstyle_{i}'])
                line.set_marker(self.config[f'mstyle_{i}'])
                line.set_color(self.config[f'color_{i}'])
                line.set_label(self.config[f'label_{i}'])
                line.set_alpha(self.config[f'alpha_{i}'])
            print(line.get_label())

        if self.config['legend_elements'] is not None:
            pds.add_legend(pt.axes_array[self.config['legend_on_ax']])
        if self.config['Label_x-pos'] is not None: pds.label_inline(lines)
        # if self.config['Label_x-pos_0'] is not None: pds.label_inline(lines)
        if self.config['vertical_line_pos_1'] is not None: pds.plot_vlines(pt)
        if self.config['plot_inset'] is not None: pds.plot_inset(pt)
        if self.config['broken_axis'] is not None: pds.plot_broken(pt)
        if self.config['label_subplot'] is not None: self.label_subplot(pt)
        for i, ax in enumerate(pt.axes_array): # TODO Generalize if the range is percentage or exact value
            if self.config[f'xlo_{i}'] is not None:
                ax.set_xlim(left=self.config[f'xlo_{i}'])
            if self.config[f'xhi_{i}'] is not None:
                ax.set_xlim(right=self.config[f'xhi_{i}']*np.max(x))
            if self.config[f'ylo_{i}'] is not None:
                ax.set_ylim(bottom=self.config[f'ylo_{i}'])
            if self.config[f'yhi_{i}'] is not None:
                ax.set_ylim(top=self.config[f'yhi_{i}'])
        if self.config['set_ax_height'] is not None: pds.set_ax_height(pt)


    def qtty_dim(self, *arr_to_plot, **kwargs):

        qtts = arr_to_plot[0]
        datasets = self.datasets_x
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        # Label the bottom axis
        if self.config['dim'] == 'l': self.axes_array[-1].set_xlabel(labels[1])
        if self.config['dim'] == 'h': self.axes_array[-1].set_xlabel(labels[0])
        if self.config['dim'] == 't': self.axes_array[-1].set_xlabel(labels[2])

        mpl.rcParams.update({'lines.linewidth': 2})
        mpl.rcParams.update({'lines.markersize': 5})
        # mpl.rcParams.update({'axes.labelsize': 10})
        # mpl.rcParams.update({'xtick.labelsize': 10})
        # mpl.rcParams.update({'ytick.labelsize': 10})


        # The for loop plots all the quantities needed for each dataset then moves
        # on to the next dataset. The quantites can be plotted on the same subplot
        # if nsubplots variable is set to 1. OR they can be plotted separately.
        for i in range(len(datasets)):
            n=0    # subplot
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            if self.config['dim'] == 'l': x = data.length_array[1:-1]   # nm
            if self.config['dim'] == 'h': x = data.height_array      # nm
            if self.config['dim'] == 't': x = self.time[self.skip:] * 1e-6      # ns

            if 'vx_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[5])
                if self.config['dim'] == 'l':
                    arr, y = data.velocity()['vx_full_x'], data.velocity()['vx_X']
                if self.config['dim'] == 'h':
                    arr, y = data.velocity()['vx_full_z'], data.velocity()['vx_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.velocity()['vx_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                # Fitting the data
                if self.config[f'fit'] is not None and self.config['dim'] == 'h':
                    fit_data = funcs.fit(x[y!=0][1:-1] ,y[y!=0][1:-1], self.config[f'fit'])['fit_data']
                    self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                if self.config['extrapolate'] is not None:
                    x_left = data.slip_length()['xdata_left']
                    y_left = data.slip_length()['extrapolate_left']
                    self.axes_array[n].plot(x_left, y_left, marker=' ', ls='--', color='k')
                    # ax.set_xlim([data.slip_length()['root_left'], 0.5*np.max(x)])
                    # ax.set_ylim([0, 1.1*np.max(y)])
                if self.nrows>1: n+=1

                if self.config['broken_axis'] is not None:
                    try:
                        if self.config['plot_on_all'] is not None:
                            for n in len(self.nrows): # plot the same data on all the axes except the last one
                                self.axes_array[n].plot(x[y!=0], y[y!=0])
                                if self.config[f'fit'] is not None: self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                                n+=1
                    except KeyError:
                        while n<self.nrows-1: # plot the same data on all the axes except the last one
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
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1


            if 'mflowrate_dim' in qtts:
                self.axes_array[n].set_ylabel(r'${\mathrm{\dot{m}}}$ (g/ns)')
                if self.config['dim'] == 'l':
                    arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_X']#*1e20
                if self.config['dim'] == 't':
                    arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_t']#*1e18
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1


            if 'den_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[3])
                if self.config['dim'] == 'l':
                    arr, y = data.density()['den_full_x'], data.density()['den_X']
                if self.config['dim'] == 'h':
                    arr, y = data.density()['den_full_z'], data.density()['den_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.density()['den_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'virxy_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wxy_full_x'], data.virial()['Wxy_X']
                if self.config['dim'] == 'h':
                    arr, y = data.virial()['Wxy_full_z'], data.virial()['Wxy_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wxy_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'virxx_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wxx_full_x'], data.virial()['Wxx_X']
                if self.config['dim'] == 'h':
                    arr, y = data.virial()['Wxx_full_z'], data.virial()['Wxx_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wxx_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'viryy_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wyy_full_x'], data.virial()['Wyy_X']
                if self.config['dim'] == 'h':
                    arr, y = data.virial()['Wyy_full_z'], data.virial()['Wyy_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wyy_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'virzz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wzz_full_x'], data.virial()['Wzz_X']
                if self.config['dim'] == 'h':
                    arr, y = data.virial()['Wzz_full_z'], data.virial()['Wzz_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wzz_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'virxz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wxz_full_x'], data.virial()['Wxz_X']
                if self.config['dim'] == 'h':
                    arr, y = data.virial()['Wxz_full_z'], data.virial()['Wxz_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wxz_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'viryz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['Wyz_full_x'], data.virial()['Wyz_X']
                if self.config['dim'] == 'h':
                    arr, y = data.virial()['Wyz_full_z'], data.virial()['Wyz_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['Wyz_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'vir_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = data.virial()['vir_full_x'], data.virial()['vir_X']
                if self.config['dim'] == 'h':
                    try:
                        x = data.bulk_height_array  # simulation with walls
                    except AttributeError:
                        x = data.height_array       # bulk simulations
                    arr, y = data.virial()['vir_full_z'], data.virial()['vir_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.virial()['vir_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                # Save the pressure data to a txt file (to compare with the continuum)
                ymax = np.argmax(y)-1
                ymin = np.argmin(y)-1

                # Include the part before the pump

                xval = x[ymax:] - x[ymax]

                if self.config['dim'] == 'l':
                    np.savetxt('press-profile-MD.txt', np.c_[xval, y[1:-1][ymax:]],  delimiter=' ',\
                                                    header='Length (nm)              Pressure (MPa)')
                # np.savetxt('press-profile-MD-full.txt', np.c_[x, y[1:-1]],  delimiter=' ',\
                #                 header='Length (nm)              Pressure (MPa)')
                if self.nrows>1: n+=1

            if 'sigzz_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[7])
                if self.config['dim'] == 'l':
                    arr, y = None, data.sigwall()['sigzz_X']
                    pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.config['dim'] == 't':
                    arr, y = None, data.sigwall()['sigzz_t']
                    pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.config['err_caps'] is not None:
                    err =  data.sigwall()['sigzz_err'][1:-1]
                    pds.plot_err_caps(self.axes_array[n], x, y, None, err)
                if self.config['err_fill'] is not None:
                    lo =  data.sigwall()['sigzz_lo'][1:-1]
                    hi =  data.sigwall()['sigzz_hi'][1:-1]
                    pds.plot_err_fill(self.axes_array[n], x, lo, hi)
                if self.nrows>1: n+=1

            if 'sigxz_dim' in qtts:
                self.axes_array[n].set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                if self.config['dim'] == 'l':
                    arr, y = None, data.sigwall()['sigxz_X']
                    pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.config['dim'] == 't':
                    arr, y = None, data.sigwall()['sigxz_t']
                    pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.config['err_caps'] is not None:
                    if self.config['dim']=='l': # Error in eachchunk
                        err =  data.sigwall()['sigxz_err'][1:-1]
                    else:
                        err =  data.sigwall()['sigxz_err_t'][1:-1]
                    pds.plot_err_caps(self.axes_array[n], x, y, None, err)
                if self.config['err_fill'] is not None:
                    if self.config['dim']=='l': # Error in eachchunk
                        lo =  data.sigwall()['sigxz_lo'][1:-1]
                        hi =  data.sigwall()['sigxz_hi'][1:-1]
                    else:   # Error in timeseries
                        lo =  data.sigwall()['sigxz_lo_t']
                        hi =  data.sigwall()['sigxz_hi_t']
                    pds.plot_err_fill(self.axes_array[n], x, lo, hi)
                if self.nrows>1: n+=1

            if 'temp_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[6])
                if self.config['dim'] == 'l':
                    arr, y = data.temp()['temp_full_x'], data.temp()['temp_X']
                if self.config['dim'] == 'h':
                    arr, y = data.temp()['temp_full_z'], data.temp()['temp_Z']
                if self.config['dim'] == 't':
                    arr, y = None, data.temp()['temp_t']
                if self.config['broken_axis'] is None: pds.ax_settings(self.axes_array[n], x, y, arr)
                # Fitting the data
                if self.config[f'fit'] is not None and self.config['dim'] == 'h':
                    fit_data = funcs.fit(x[y!=0][1:-1] ,y[y!=0][1:-1], self.config[f'fit'])['fit_data']
                    self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                if self.nrows>1: n+=1
                if self.config['broken_axis'] is not None:
                    try:
                        if self.config['plot_on_all'] is not None:
                            for n in range(self.nrows): # plot the same data on all the axes except the last one
                                self.axes_array[n].plot(x, y[1:-1])
                                n+=1
                    except KeyError:
                        while n<self.nrows-1: # plot the same data on all the axes except the last one
                            self.axes_array[n].plot(x, y[1:-1])
                            n+=1

            if 'tempX_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[6])
                if self.config['dim'] == 'l':
                    arr, y = data.temp()['tempX_full_x'], data.temp()['tempX_len']
                if self.config['dim'] == 'h':
                    arr, y = data.temp()['tempX_full_z'], data.temp()['tempX_height']
                if self.config['dim'] == 't':
                    arr, y = None, data.temp()['tempX_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)

            if 'tempY_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[6])
                if self.config['dim'] == 'l':
                    arr, y = data.temp()['tempY_full_x'], data.temp()['tempY_len']
                if self.config['dim'] == 'h':
                    arr, y = data.temp()['tempY_full_z'], data.temp()['tempY_height']
                if self.config['dim'] == 't':
                    arr, y = None, data.temp()['tempY_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)

            if 'tempZ_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[6])
                if self.config['dim'] == 'l':
                    arr, y = data.temp()['tempZ_full_x'], data.temp()['tempZ_len']
                if self.config['dim'] == 'h':
                    arr, y = data.temp()['tempZ_full_z'], data.temp()['tempZ_height']
                if self.config['dim'] == 't':
                    arr, y = None, data.temp()['tempZ_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)

            if 'tempS_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[6])
                try:
                    if self.config['dim'] == 'l': arr, y = data.temp()['temp_full_x_solid'], data.temp()['tempS_len']
                    if self.config['dim'] == 'h': arr, y = data.temp()['temp_full_z_solid'], data.temp()['tempS_height']
                    if self.config['dim'] == 't': arr, y = None, data.temp()['tempS_t']
                except KeyError:
                    pass
                if np.mean(data.temp()['tempS_len'])>1:     # Don't plot data where the solid temp. is zero (for example TF system)
                    pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'height_dim' in qtts:
                self.axes_array[n].set_ylabel(labels[0])
                if self.config['dim'] == 't': arr, y = None, data.h
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

            if 'je_dim' in qtts:
                self.axes_array[n].set_ylabel('J_e')
                if self.config['dim'] == 't': arr, y = None, data.heat_flux()['jez_t']
                pds.ax_settings(self.axes_array[n], x, y, arr)
                if self.nrows>1: n+=1

        pds.plot_settings(self, x)


    def v_distrib(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.axes_array[0].set_xlabel('Velocity (m/s)')
        self.axes_array[0].set_ylabel('$f(v)$')

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        kb = 8.314462618 # J/mol.K
        da = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        T =  np.mean(da.temp()['temp_t'])
        vx = da.vel_distrib()['vx_values_lte']
        vx = np.array(vx)
        mb_distribution = np.sqrt((self.mf / (2*5*np.pi*kb*T))) * np.exp((-self.mf*vx**2)/(2*5*kb*T))

        for i in range(len(self.datasets_x)):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            vx_values = data.vel_distrib()['vx_values_lte']
            vx_prob = data.vel_distrib()['vx_prob_lte']

            vy_values = data.vel_distrib()['vy_values_lte']
            vy_prob = data.vel_distrib()['vy_prob_lte']

            vz_values = data.vel_distrib()['vz_values_lte']
            vz_prob = data.vel_distrib()['vz_prob_lte']

            self.axes_array[0].plot(vx_values, vx_prob)
            self.axes_array[0].plot(vy_values, vy_prob)
            self.axes_array[0].plot(vz_values, vz_prob)

            # self.axes_array[0].plot(vx_values, mb_distribution)

        pds.plot_settings(self, vx_values)


    def v_evolution(self, opacity=1, legend=None, lab_lines=None, **kwargs):

        self.axes_array[0].set_xlabel('Height (nm)')
        self.axes_array[0].set_ylabel('$V_{x}$ (m/s)')

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)


        for i in range(len(self.datasets_x)):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            h = data.height_array
            vx_R1 =  data.velocity()['vx_R1']
            vx_R2 =  data.velocity()['vx_R2']
            vx_R3 =  data.velocity()['vx_R3']
            vx_R4 =  data.velocity()['vx_R4']
            vx_R5 =  data.velocity()['vx_R5']

            self.axes_array[0].plot(h[vx_R1!=0][1:-1], vx_R1[vx_R1!=0][1:-1])
            self.axes_array[0].plot(h[vx_R2!=0][1:-1], vx_R2[vx_R2!=0][1:-1])
            self.axes_array[0].plot(h[vx_R3!=0][1:-1], vx_R3[vx_R3!=0][1:-1])
            self.axes_array[0].plot(h[vx_R4!=0][1:-1], vx_R4[vx_R4!=0][1:-1])
            self.axes_array[0].plot(h[vx_R5!=0][1:-1], vx_R5[vx_R5!=0][1:-1])

        pds.plot_settings(self, h)



    def pgrad_mflowrate(self):

        mpl.rcParams.update({'lines.markersize': 7})

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        # Get the error in mflowrate in MD
        self.axes_array[0].set_xlabel(labels[9])
        self.axes_array[0].set_ylabel(labels[10])

        pGrad, shear_rate, \
        mflowrate_ff_avg, mflowrate_ff_err, mflowrate_fc_avg, mflowrate_fc_err, \
        mflowrate_hp, mflowrate_hp_slip = [], [], [], [], [], [], [], []

        for i, val in enumerate(self.datasets_x):
            data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)

            avg_gap_height = np.mean(data.h)*1e-9

            if 'ff' in val:
                mflowrate_ff_avg.append(np.mean(data.mflux()['mflowrate_stable']))
                mflowrate_ff_err.append(sq.get_err(data.mflux()['mflowrate_stable'])['uncertainty'])

                # Continuum prediction
                pGrad.append(np.absolute(data.virial()['pGrad']))
                shear_rate.append(data.transport()['shear_rate'])
                mu = data.transport()['mu']            # mPa.s
                slip_vel = np.mean(data.slip_length()['Vs'])
                bulk_den_avg = np.mean(data.density()['den_t'])

                flowrate_cont = (bulk_den_avg*1e3 * self.Ly*1e-9 * 1e-9 * 1e3 * pGrad[i]*1e6*1e9 * avg_gap_height**3)  / \
                            (12 * mu*1e-3)

                mflowrate_hp.append(flowrate_cont)
                mflowrate_hp_slip.append(flowrate_cont + (bulk_den_avg*1e6 *  \
                                    self.Ly*1e-9 * slip_vel * avg_gap_height*1e-9) )

            if 'fc' in val:
                mflowrate_fc_avg.append(np.mean(data.mflux()['mflowrate_stable']))
                mflowrate_fc_err.append(sq.get_err(data.mflux()['mflowrate_stable'])['uncertainty'])

        # print(pGrad)
        # print(mflowrate_ff_avg, mflowrate_ff_err)
        # print(mflowrate_fc_avg, mflowrate_fc_err)

        self.axes_array[0].ticklabel_format(axis='y', style='sci', useOffset=False)
        if mflowrate_ff_avg:
            pds.plot_err_caps(self.axes_array[0], pGrad, np.array(mflowrate_ff_avg)*1e18, None, np.array(mflowrate_ff_err)*1e18)
        if mflowrate_fc_avg:
            pds.plot_err_caps(self.axes_array[0], pGrad, np.array(mflowrate_fc_avg)*1e18, None, np.array(mflowrate_fc_err)*1e18)
        self.axes_array[0].plot(pGrad, np.array(mflowrate_hp)*1e18)
        self.axes_array[0].plot(pGrad, np.array(mflowrate_hp_slip)*1e18)

        ax2 = self.axes_array[0].twiny()
        ax2.set_xlabel(labels[11])
        ax2.set_xscale('log', nonpositive='clip')
        if mflowrate_ff_avg: ax2.plot(shear_rate, np.array(mflowrate_ff_avg)*1e18, ls= ' ', marker= ' ')
        if mflowrate_fc_avg: ax2.plot(shear_rate, np.array(mflowrate_fc_avg)*1e18, ls= ' ', marker= ' ')

        pds.plot_settings(self, pGrad)


    def rate_viscosity(self):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        if len(self.axes_array)==1: self.axes_array[0].set_xlabel(labels[11])

        self.axes_array[0].set_ylabel(labels[12])
        self.axes_array[0].set_xscale('log', nonpositive='clip')
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
            print(val)
            if 'couette' in val:
                pumpsize = 0
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                # print(data.transport()['shear_rate'])
                viscosity.append(data.transport()['mu'])
                shear_rate_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                viscosity_err.append(data.transport()['mu_hi'] - data.transport()['mu_lo'])

            if 'ff' in val:
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_ff.append(data.transport()['shear_rate'])
                # print(data.transport()['shear_rate'])
                viscosity_ff.append(data.transport()['mu'])
                shear_rate_ff_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                viscosity_ff_err.append(data.transport()['mu_hi'] - data.transport()['mu_lo'])

            if 'fc' in val:
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_fc.append(data.transport()['shear_rate'])
                viscosity_fc.append(data.transport()['mu'])
                shear_rate_fc_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                viscosity_fc_err.append(data.transport()['mu_hi'] - data.transport()['mu_lo'])

            if 'vib' in val:
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_vib.append(data.transport()['shear_rate'])
                viscosity_vib.append(data.transport()['mu'])

            if 'rigid' in val:
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
                shear_rate_rigid.append(data.transport()['shear_rate'])
                viscosity_rigid.append(data.transport()['mu'])

        # Plot with error bars
        if self.config['err_caps']:
            if viscosity: pds.plot_err_caps(self.axes_array[0], shear_rate, viscosity, shear_rate_err, viscosity_err)
            if viscosity_ff: pds.plot_err_caps(self.axes_array[0], shear_rate_ff, viscosity_ff, shear_rate_ff_err, viscosity_ff_err)
            if viscosity_fc: pds.plot_err_caps(self.axes_array[0], shear_rate_fc, viscosity_fc, shear_rate_fc_err, viscosity_fc_err)

        # Plot with fit
        if self.config['fit']:
            if viscosity:
                popt, pcov = curve_fit(funcs.power, shear_rate, viscosity, maxfev=8000)
                if not self.config['err_caps']: self.axes_array[0].plot(shear_rate, viscosity)
                self.axes_array[0].plot(shear_rate, funcs.power(shear_rate, *popt))
            if viscosity_ff:
                popt, pcov = curve_fit(funcs.power, shear_rate_ff, viscosity_ff, maxfev=8000)
                if not self.config['err_caps']: self.axes_array[0].plot(shear_rate_ff, viscosity_ff)
                self.axes_array[0].plot(shear_rate_ff, funcs.power(shear_rate_ff, *popt))
            if viscosity_fc:
                popt, pcov = curve_fit(funcs.power, shear_rate_fc, viscosity_fc, maxfev=8000)
                if not self.config['err_caps']: self.axes_array[0].plot(shear_rate_fc, viscosity_fc)
                self.axes_array[0].plot(shear_rate_fc, funcs.power(shear_rate_fc, *popt))

        # Plot raw data
        if not self.config['fit'] and not self.config['err_caps']:
            if viscosity: self.axes_array[0].plot(shear_rate, viscosity)
            if viscosity_ff and not viscosity_vib: self.axes_array[0].plot(shear_rate_ff, viscosity_ff)
            if viscosity_fc and not viscosity_vib: self.axes_array[0].plot(shear_rate_fc, viscosity_fc)
            if viscosity_vib:
                self.axes_array[0].plot(shear_rate_rigid, viscosity_rigid)
                self.axes_array[0].plot(shear_rate_vib, viscosity_vib)

        if viscosity:
            pds.plot_settings(self, shear_rate)
        elif viscosity_ff:
            pds.plot_settings(self, shear_rate_ff)
        elif viscosity_fc:
            pds.plot_settings(self, shear_rate_fc)


    def rate_stress(self):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        if len(self.axes_array)==1:
            axis=self.axes_array[0]
        else:
            axis=self.axes_array[1]

        axis.set_xlabel(labels[11])
        axis.set_ylabel('$\sigma_{xz}$ (MPa)')
        axis.set_xscale('log', nonpositive='clip')
        mpl.rcParams.update({'lines.markersize': 8})

        shear_rate, stress = [], []
        shear_rate_err, stress_err = [], []
        shear_rate_ff, stress_ff = [], []
        shear_rate_ff_err, stress_ff_err = [], []
        shear_rate_fc, stress_fc = [], []
        shear_rate_fc_err, stress_fc_err = [], []

        for idx, val in enumerate(self.datasets_x):
            print(val)
            if 'couette' in val:
                pumpsize = 0
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                stress.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                stress_err.append(data.sigwall()['sigxz_err_t'])

            if 'ff' in val:
                pumpsize = self.pumpsize
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_ff.append(data.transport()['shear_rate'])
                stress_ff.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_ff_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                stress_ff_err.append(data.sigwall()['sigxz_err_t'])

            if 'fc' in val:
                pumpsize = self.pumpsize
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_fc.append(data.transport()['shear_rate'])
                stress_fc.append(np.mean(data.sigwall()['sigxz_t']))
                shear_rate_fc_err.append(data.transport()['shear_rate_hi'] - data.transport()['shear_rate_lo'])
                stress_fc_err.append(data.sigwall()['sigxz_err_t'])


        # Plot with error bars
        if self.config['err_caps']:
            a,b,c = [],[],[]
            if shear_rate:
                for i in stress_err:
                    a.append(np.mean(i))
                pds.plot_err_caps(axis, shear_rate, stress, shear_rate_err, a)
            if shear_rate_ff:
                for j in stress_ff_err:
                    b.append(np.mean(j))
                pds.plot_err_caps(axis, shear_rate_ff, stress_ff, shear_rate_ff_err, b)
            if shear_rate_fc:
                for j in stress_fc_err:
                    c.append(np.mean(j))
                pds.plot_err_caps(axis, shear_rate_fc, stress_fc, shear_rate_fc_err, c)

        # Plot with fit
        if self.config['fit']:
            if shear_rate:
                popt, pcov = curve_fit(funcs.power, shear_rate, stress)
                if not self.config['err_caps']: axis.plot(shear_rate, stress)
                axis.plot(shear_rate, funcs.power(shear_rate, *popt))
            if shear_rate_ff:
                popt, pcov = curve_fit(funcs.power, shear_rate_ff, stress_ff)
                if not self.config['err_caps']: axis.plot(shear_rate_ff, stress_ff)
                axis.plot(shear_rate_ff, funcs.power(shear_rate_ff, *popt))
            if shear_rate_fc:
                popt, pcov = curve_fit(funcs.power, shear_rate_fc, stress_fc)
                if not self.config['err_caps']: axis.plot(shear_rate_fc, stress_fc)
                axis.plot(shear_rate_fc, funcs.power(shear_rate_fc, *popt))

        # Plot raw data
        if not self.config['fit'] and not self.config['err_caps']:
            if shear_rate: axis.plot(shear_rate, stress)
            if shear_rate_ff: axis.plot(shear_rate_ff, stress_ff)
            if shear_rate_fc: axis.plot(shear_rate_fc, stress_fc)

        if len(self.axes_array)==1: # If it is the only plot
            if shear_rate: pds.plot_settings(self, shear_rate)
            elif shear_rate_ff: pds.plot_settings(self, shear_rate_ff)
            elif shear_rate_fc: pds.plot_settings(self, shear_rate_fc)

    def rate_slip(self):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        self.axes_array[0].set_xlabel(labels[11])
        self.axes_array[0].set_ylabel('Slip Length (nm)')
        self.axes_array[0].set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, slip = [], []
        shear_rate_pd, slip_pd = [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                slip.append(data.slip_length()['Ls'])

            if 'ff' in val or 'fc' in val:
                pumpsize = self.pumpsize
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_pd.append(data.transport()['shear_rate'])
                slip_pd.append(data.slip_length()['Ls'])

        if shear_rate: self.axes_array[0].plot(shear_rate, slip)
        if shear_rate_pd: self.axes_array[0].plot(shear_rate_pd, slip_pd)

        if shear_rate:
            pds.plot_settings(self, shear_rate)
        elif shear_rate_pd:
            pds.plot_settings(self, shear_rate_pd)


    def rate_temp(self):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        self.axes_array[0].set_xlabel(labels[11])
        self.axes_array[0].set_ylabel(labels[6])
        self.axes_array[0].set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, temp = [], []
        shear_rate_pd, temp_pd = [], []
        shear_rate_vib, shear_rate_rigid, temp_vib, temp_rigid = [], [], [], []

        for idx, val in enumerate(self.datasets_x):
            if 'couette' in val:
                pumpsize = 0
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate.append(data.transport()['shear_rate'])
                temp.append(np.mean(data.temp()['temp_t']))

            if 'ff' in val or 'fc' in val:
                pumpsize = self.pumpsize
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_pd.append(data.transport()['shear_rate'])
                temp_pd.append(np.mean(data.temp()['temp_t']))

            if 'vibrating' in val:
                pumpsize = self.pumpsize
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_vib.append(data.transport()['shear_rate'])
                temp_vib.append(np.mean(data.temp()['temp_t']))

            if 'rigid' in val:
                pumpsize = self.pumpsize
                data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
                shear_rate_rigid.append(data.transport()['shear_rate'])
                temp_rigid.append(np.mean(data.temp()['temp_t']))

        if shear_rate: self.axes_array[0].plot(shear_rate, temp)
        if shear_rate_pd and not shear_rate_vib: self.axes_array[0].plot(shear_rate_pd, temp_pd)
        if shear_rate_vib:
            self.axes_array[0].plot(shear_rate_rigid, temp_rigid)
            self.axes_array[0].plot(shear_rate_vib, temp_vib)

        if shear_rate:
            pds.plot_settings(self, shear_rate)
        elif shear_rate_pd:
            pds.plot_settings(self, shear_rate_pd)


    def rate_qdot(self):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        self.axes_array[0].set_xlabel(labels[11])
        self.axes_array[0].ticklabel_format(axis='y', style='sci', useOffset=False)
        self.axes_array[0].set_ylabel(r'${\mathrm{\dot{Q}}} \times 10^{-24}}$ (W)')
        # self.axes_array[0].set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, qdot, qdot_continuum = [], [], []

        for idx, val in enumerate(self.datasets_x):
            # if 'rigid' in val:
            pumpsize = self.pumpsize
            data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
            shear_rate.append(data.transport()['shear_rate'])
            qdot.append(np.mean(data.lambda_IK()['qdot'])*1e24)
            qdot_continuum.append(np.mean(data.lambda_IK()['qdot_continuum'])*1e24)

        self.axes_array[0].plot(shear_rate, qdot)
        self.axes_array[0].plot(shear_rate, qdot_continuum)

        pds.plot_settings(self, shear_rate)


    def rate_lambda(self):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        self.axes_array[0].set_xlabel(labels[11])
        self.axes_array[0].ticklabel_format(axis='y', style='sci', useOffset=False)
        self.axes_array[0].set_ylabel(r'${\mathrm{\lambda} (W/m^2.K)}$')

        self.axes_array[0].set_xscale('log', nonpositive='clip')

        mpl.rcParams.update({'lines.markersize': 6})

        shear_rate, lambda_z, lambda_continuum = [], [], []

        for idx, val in enumerate(self.datasets_x):
            # if 'rigid' in val:
            pumpsize = self.pumpsize
            data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize)
            shear_rate.append(data.transport()['shear_rate'])
            lambda_z.append(np.mean(data.lambda_IK()['lambda_z']))
            lambda_continuum.append(np.mean(data.lambda_IK()['lambda_continuum']))

        self.axes_array[0].plot(shear_rate, lambda_z)
        self.axes_array[0].plot(shear_rate, lambda_continuum)

        pds.plot_settings(self, shear_rate)



    def thermal_conduct(self):
        """
        Plot thermal conductivity with Pressure. Equilibrium simulations.
        """

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)
        self.axes_array[0].set_xlabel(labels[7])
        self.axes_array[0].set_ylabel('$\lambda$ (W/mK)')

        # Experimal results are from Wang et al. 2020 (At temp. 335 K)
        exp_lambda = [0.1009,0.1046,0.1113,0.1170]
        exp_press = [5,10,20,30]

        md_lambda, md_press = [], []

        for idx, val in enumerate(self.datasets_x):
            data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
            md_lambda.append(np.mean(data.lambda_gk()['lambda_tot']))

        self.axes_array[0].plot(exp_press, md_lambda)
        self.axes_array[0].plot(exp_press, exp_lambda)

        pds.plot_settings(self, exp_press)


    def eos(self):

        mpl.rcParams.update({'lines.markersize': 8})
        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)

        if self.config['log']:
            # self.axes_array[0].xaxis.major.formatter.set_scientific(False)
            self.axes_array[0].set_yscale('log', base=10)
            self.axes_array[1].set_yscale('log', base=10)
            self.axes_array[0].set_xscale('log', base=10)
            self.axes_array[1].set_xscale('log', base=10)
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
        press_list, den_list = [], []
        press2_list, temp_list = [], []
        ds_gau_1fs, ds_ver_1fs, ds_gau_4fs, ds_ver_4fs = [], [], [], []
        den_list_gau_1fs, den_list_gau_4fs, den_list_ver_1fs, den_list_ver_4fs = [], [], [], []
        press_list_gau_1fs, press_list_gau_4fs, press_list_ver_1fs, press_list_ver_4fs = [], [], [], []

        for idx, val in enumerate(self.datasets_x):
            # if 'ff' in val or 'fc' in val or 'couette' in val:
            if 'flow' in val:
                ds_nemd.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize))
            if 'isochore' in val:
                ds_isochores.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'isotherm' in val:
                ds_isotherms.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'gautschi' in val and '1fs' in val:
                ds_gau_1fs.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'gautschi' in val and '4fs' in val:
                ds_gau_4fs.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'verlet' in val and '1fs' in val:
                ds_ver_1fs.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))
            if 'verlet' in val and '4fs' in val:
                ds_ver_4fs.append(dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, pumpsize=0))

                # data = dd(self.skip, self.datasets_x[idx], self.datasets_z[idx], self.mf, self.pumpsize)
        for i in ds_nemd:
            # print(np.mean(i.density()['den_t']))
            # print(np.mean(i.virial()['vir_t']))
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
            den_list.append(np.mean(i.density()['den_t']))
            press_list.append(np.mean(i.virial()['vir_t']))

        den_list, press_list = np.asarray(den_list), np.asarray(press_list)

        for i in ds_gau_1fs:
            den_list_gau_1fs.append(np.mean(i.density()['den_t']))
            press_list_gau_1fs.append(np.mean(i.virial()['vir_t']))
        for i in ds_gau_4fs:
            den_list_gau_4fs.append(np.mean(i.density()['den_t']))
            press_list_gau_4fs.append(np.mean(i.virial()['vir_t']))
        for i in ds_ver_1fs:
            den_list_ver_1fs.append(np.mean(i.density()['den_t']))
            press_list_ver_1fs.append(np.mean(i.virial()['vir_t']))
        for i in ds_ver_4fs:
            den_list_ver_4fs.append(np.mean(i.density()['den_t']))
            press_list_ver_4fs.append(np.mean(i.virial()['vir_t']))

        if ds_isotherms:
            if self.config['log']:
                # print(np.max(den_list))
                # print(np.max(press_list))
                den_list /= 0.72 #np.max(den_list)
                press_list /= 250 #np.max(press_list)

            if ds_gau_1fs:
                self.axes_array[0].plot(den_list_gau_1fs, press_list_gau_1fs)
            if ds_gau_4fs:
                self.axes_array[0].plot(den_list_gau_4fs, press_list_gau_4fs)
            if ds_ver_1fs:
                self.axes_array[0].plot(den_list_ver_1fs, press_list_ver_1fs)
            if ds_ver_4fs:
                self.axes_array[0].plot(den_list_ver_4fs, press_list_ver_4fs)
            # else:
            #     self.axes_array[0].plot(den_list, press_list)

            # Experimental data (K. Liu et al. / J. of Supercritical Fluids 55 (2010) 701–711)
            exp_density = [0.630, 0.653, 0.672, 0.686, 0.714, 0.739, 0.750]
            exp_press = [28.9, 55.3, 84.1, 110.2, 171.0, 239.5, 275.5]

            self.axes_array[0].plot(exp_density, exp_press)

            if self.config['log']:
                coeffs_den = curve_fit(funcs.power, den_list, press_list, maxfev=8000)
                print(f'Adiabatic exponent (gamma) is {coeffs_den[0][1]}')
                self.axes_array[0].plot(den_list, funcs.power(den_list, coeffs_den[0][0], coeffs_den[0][1], coeffs_den[0][2]))

        # Isochores -----------------
        for i in ds_isochores:
            temp_list.append(np.mean(i.temp()['temp_t']))
            press2_list.append(np.mean(i.virial()['vir_t']))
        temp_lst, press2_list = np.asarray(temp_list), np.asarray(press2_list)

        if ds_isochores:
            if self.config['log']:
                temp_list /= 300 #np.max(temp_list)
                press2_list /=  250 #np.max(press2_list)

            self.axes_array[1].plot(temp_list, press2_list)

            if self.config['log']:
                coeffs_temp = curve_fit(funcs.power, temp_list, press2_list, maxfev=8000)
                print(f'Adiabatic exponent (gamma) is {coeffs_temp[0][1]}')
                self.axes_array[1].plot(temp_list, funcs.power(temp_list, coeffs_temp[0][0], coeffs_temp[0][1], coeffs_temp[0][2]))

        if ds_nemd: pds.plot_settings(self, press)
        elif ds_isotherms: pds.plot_settings(self, press_list)
        elif ds_isochores: pds.plot_settings(self, press2_list)

        # if 'ff' in val or 'fc' in val or 'couette' in val: ds_nemd.append(y[1].set_yscale('log', base=10)

        if self.config['log']:
            self.axes_array[0].xaxis.set_minor_formatter(ScalarFormatter())
            self.axes_array[1].xaxis.set_minor_formatter(ScalarFormatter())


    def coexistence_curve(self, **kwargs):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)
        self.axes_array[0].set_xlabel(labels[3])
        self.axes_array[0].set_ylabel('Temperature (K)')

        temp, rho_l, rho_v = [], [], []
        for i in range(len(self.datasets_x)):
            ds = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            temp.append(np.mean(ds.temp()['temp_t']))
            rho_l.append(ds.coexistence_densities()['rho_l'])
            rho_v.append(ds.coexistence_densities()['rho_v'])

        self.axes_array[0].plot(rho_v,temp)
        self.axes_array[0].plot(rho_l,temp)

        if self.mf ==39.948:
            #Exp. data Lemmon et al.
            rhoc, Tc = 0.5356,150.65 # g/cm3, K
            temp_exp = [85,90,95,100,105,110,115,120,125,130,135]
            rho_v_exp = [0.0051,0.0085,0.0136,0.0187,0.0255,0.0373,0.0441,0.0611,0.0798,0.1052,0.1392]
            rho_l_exp = [1.4102,1.3813,1.3491,1.3152,1.2812,1.2439,1.2032,1.1624,1.1166,1.0674,1.0080]
            self.axes_array[0].plot(rho_v_exp,temp_exp)
            self.axes_array[0].plot(rho_l_exp,temp_exp)

        if self.mf == 72.15:
            # Exp.: B.D. Smith and R. Srivastava, Thermodynamic Data for Pure Compounds: Part A Hydrocarbons and Ketone
            # TrAPPE-EH: from (Journal of Physics: Conf. Series 946 (2018) 012099)
            rhoc, Tc = 0.232, 469.70  # g/cm3, K
            temp_exp = [150,175,200,225,250,275,300,325,350,375,400,425]
            rho_v_exp = [0,0,0,0,0,0,0.0006,0.0048,0.0109,0.0182,0.0315,0.0497]
            rho_l_exp = [0.7575,0.7339,0.7115,0.6891,0.6661,0.6431,0.6201,0.5940,0.5662,0.5341,0.5008,0.4554]
            #rho_v_trappe_eh = [0.0011,0.0033,0.0022,0.0056,0.0089,0.0145,0.0134,0.0189,0.0178,0.0334,0.0434]
            #rho_l_trappe_eh = [0.6122,0.5967,0.5866,0.5744,0.5622,0.5466,0.5499,0.5054,0.5054,0.4876,0.4776]
            self.axes_array[0].plot(rho_v_exp,temp_exp)
            self.axes_array[0].plot(rho_l_exp,temp_exp)

        # For Density scaling law ---> Get Tc
        rho_diff = np.array(rho_l) - np.array(rho_v)
        popt, pcov = curve_fit(funcs.density_scaling, rho_diff/np.max(rho_diff), temp, maxfev=8000)
        Tc_fit = popt[1]
        # print(Tc_fit)

        self.axes_array[0].plot(rho_l,funcs.density_scaling(rho_diff/np.max(rho_diff), *popt))
        self.axes_array[0].plot(rho_v,funcs.density_scaling(rho_diff/np.max(rho_diff), *popt))

        # For law of rectilinear diameters ---> Get rhoc
        rho_sum = np.array(rho_l) + np.array(rho_v)
        popt2, pcov2 = curve_fit(funcs.rectilinear_diameters, rho_sum, temp, maxfev=8000)
        rhoc_fit=popt2[1]/(popt2[0]*2)
        # print(rhoc_fit)

        self.axes_array[0].plot(rhoc_fit, Tc_fit)
        self.axes_array[0].plot(rhoc, Tc)

        pds.plot_settings(self, rho_l)


    def rc_gamma(self, **kwargs):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)
        self.axes_array[0].set_xlabel('$r_c/\sigma$')
        self.axes_array[0].set_ylabel('$\gamma$ (mN/m)')

        if self.mf == 39.948: rc = [3,4,5,6,7]
        if self.mf == 72.15: rc = [10,12,14,18,21,24]
        gamma =  []
        for i in range(len(self.datasets_x)):
            ds = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            gamma.append(np.mean(ds.surface_tension()['gamma'])*1e3)

        self.axes_array[0].plot(rc,gamma)
        pds.plot_settings(self, rc)

    def temp_gamma(self, **kwargs):

        pds = plot_from_ds(self.skip, self.datasets_x, self.datasets_z, self.mf, self.configfile, self.pumpsize)
        self.axes_array[0].set_xlabel('Temperature (K)')
        self.axes_array[0].set_ylabel('$\gamma$ (mN/m)')

        temp, gamma = [], []
        for i in range(len(self.datasets_x)):
            ds = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            temp.append(np.mean(ds.temp()['temp_t']))
            gamma.append(np.mean(ds.surface_tension()['gamma'])*1e3)

        if self.mf == 39.948: gamma_exp = [11.871,10.661,9.7482,8.3243,7.2016,6.1129,5.0617,4.0527,3.0919,2.2879]
        if self.mf == 72.15: gamma_exp = [32.5,29.562,26.2638,23.736,20.864,18.032,15.250,12.534,9.9014,7.379,5.0029]#,2.8318,0.98]

        popt, pcov = curve_fit(funcs.power_new, temp/np.max(temp), gamma, maxfev=8000)

        print(f'Critical temperature: Tc = {popt[1]*np.max(temp):.2f}')

        self.axes_array[0].plot(temp,gamma)
        self.axes_array[0].plot(temp,gamma_exp)
        self.axes_array[0].plot(temp,funcs.power_new(temp/np.max(temp), *popt))

        pds.plot_settings(self, temp)



    def struc_factor(self, solid=0, **kwargs):

        data = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)

        kx = data.sf()['kx']
        ky = data.sf()['ky']
        # k = data.sf()['k']

        sfx = data.sf()['sf_x']
        sfy = data.sf()['sf_y']
        sf = data.sf()['sf']
        # sf_solid = data.sf()['sf_solid']
        sf_time = data.sf()['sf_time']

        if self.config['3d']:
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
            # if solid==1:
            #     self.ax.plot_surface(Kx, Ky, sf.T, cmap=cmx.jet,
            #                 rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
            # self.fig.colorbar(surf, shrink=0.5, aspect=5)
            # self.ax.view_init(35,60)
            self.ax.view_init(90,0)
            # pickle.dump(self.fig, open('FigureObject.fig.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

        else:
            a = input('x or y or r or t:')
            if a=='x':
                self.axes_array[0].set_xlabel('$k_x (\AA^{-1})$')
                self.axes_array[0].set_ylabel('$S(K_x)$')
                self.axes_array[0].plot(kx, sfx, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
            elif a=='y':
                self.axes_array[0].set_xlabel('$k_y (\AA^{-1})$')
                self.axes_array[0].set_ylabel('$S(K_y)$')
                self.axes_array[0].plot(ky, sfy, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))
            # elif a=='r':
            #     self.axes_array[0].set_xlabel('$k (\AA^{-1})$')
            #     self.axes_array[0].set_ylabel('$S(K)$')
            #     self.axes_array[0].plot(k, sf_r, ls= ' ', marker='x', alpha=opacity,
            #                label=input('Label:'))
            elif a=='t':
                self.axes_array[0].set_xlabel('$t (fs)$')
                self.axes_array[0].set_ylabel('$S(K)$')
                self.axes_array[0].plot(self.time[self.skip:], sf_time, ls= '-', marker=' ', alpha=opacity,
                           label=input('Label:'))

    def isf(self, **kwargs):

        data = dd(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        isf_lo = data.ISF()['ISF'][:,0,0]
        isf_hi = data.ISF()['ISF'][:,50,10]
        # print(isf.shape)
        self.axes_array[0].set_xlabel('Time (fs)')
        self.axes_array[0].set_ylabel('I(K)')
        self.axes_array[0].plot(self.time[self.skip:], isf_lo)
        self.axes_array[0].plot(self.time[self.skip:], isf_hi)

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






    # def pdiff_pumpsize(self):
    #     """
    #     Plot the virial and wall sigmazz with versus the pump size
    #     """
    #
    #     self.ax.set_xlabel('Normalized pump length')
    #
    #     pump_size = []
    #     vir_pdiff, sigzz_pdiff, vir_err, sigzz_err = [], [], [], []
    #     for i in range(len(self.datasets_x)):
    #         data = dd(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
    #         pump_size.append(self.pumpsize)
    #         vir_pdiff.append(data.virial(pump_size[i])['pDiff'])
    #         vir_err.append(data.virial(pump_size[i])['pDiff_err'])
    #         sigzz_pdiff.append(data.sigwall(pump_size[i])['pDiff'])
    #         sigzz_err.append(data.sigwall(pump_size[i])['pDiff_err'])
    #
    #     markers_a, caps, bars= self.ax.errorbar(pump_size, vir_pdiff, yerr=vir_err, ls=lt, fmt=mark,
    #             label='Virial (Fluid)', capsize=1.5, markersize=1.5, alpha=1)
    #     markers2, caps2, bars2= self.ax.errorbar(pump_size, sigzz_pdiff, yerr=sigzz_err, ls=lt, fmt='x',
    #             label='$\sigma_{zz}$ (Solid)', capsize=1.5, markersize=3, alpha=1)
    #
    #     [bar.set_alpha(0.5) for bar in bars]
    #     [cap.set_alpha(0.5) for cap in caps]
    #     [bar2.set_alpha(0.5) for bar2 in bars2]
    #     [cap2.set_alpha(0.5) for cap2 in caps2]