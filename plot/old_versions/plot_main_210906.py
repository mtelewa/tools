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
nchunks = np.int(sys.argv[2])

for k in sys.argv[3:]:
    ds_dir = os.path.join(k, "data/out")
    for root, dirs, files in os.walk(ds_dir):
        for i in files:
            if i.endswith(f'{nchunks}x1.nc'):
                datasets_x.append(os.path.join(ds_dir, i))
            if i.endswith(f'1x{nchunks}.nc'):
                datasets_z.append(os.path.join(ds_dir, i))
            elif i.endswith('.txt'):
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

        self.time = dd(datasets_x[0], datasets_z[0], skip).time
        self.Ly = dd(datasets_x[0], datasets_z[0], skip).Ly

        # self.sigxz_t = np.zeros([len(datasets), len(self.time)])
        # self.avg_sigxz_t = np.zeros([len(datasets)])
        self.pGrad = np.zeros([len(datasets_x)])
        self.viscosities = np.zeros([len(datasets_z)])

    def inline(self, plot):
        rot = input('Rotation of label: ')
        xpos = np.float(input('Label x-pos:'))
        y_offset = 0 #np.float(input('Y-offset for label: '))

        for i in range (len(plot.ax.lines)):
            label_lines.label_line(plot.ax.lines[i], xpos, yoffset= y_offset, \
                     label= plot.ax.lines[i].get_label(), fontsize= 8, rotation= rot)


    def draw_vlines(self, plot):
        total_length = np.max(dd(datasets_x[0], datasets_z[0], skip).Lx)
        plot.ax.axvline(x= 0, color='k', linestyle='dashed', lw=1)

        for i in range(len(plot.ax.lines)-1):
            pos=np.float(input('vertical line pos:'))
            plot.ax.axvline(x= pos*total_length, color='k', linestyle='dashed', lw=1)




    def vx_height(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[5])

        mpl.rcParams.update({'lines.markersize': 3})

        # vx_chunkZ = np.zeros([len(datasets_z), self.Nz])

        for i in range(len(datasets_z)):
            vx_chunkZ = dd(datasets_x[i], datasets_z[i], skip).velocity()['vx_data']

            # Remove chunks with no atoms
            height_array_mod = dd(datasets_x[i], datasets_z[i], skip).height_array[vx_chunkZ !=0]
            vx_chunkZ_mod = vx_chunkZ[vx_chunkZ !=0]

            label=input('Label:')

            if label == 'FF':
                self.ax.plot(height_array_mod, vx_chunkZ_mod,
                                     ls=' ', label=label, marker='o', alpha=opacity)
            elif label == 'FC':
                self.ax.plot(height_array_mod, vx_chunkZ_mod,
                                     ls=' ', label=label, marker='x', alpha=opacity)

        # Plot the fits
        if 'fit' in sys.argv:
            for i in range(len(datasets_z)):
                # popt, pcov = curve_fit(funcs.quadratic, xdata, ydata)
                #     ax.plot(xdata, ydata, *popt))
                x1 = dd(datasets_x[i], datasets_z[i], skip).velocity()['xdata']
                y1 = dd(datasets_x[i], datasets_z[i], skip).velocity()['fit_data']

                self.ax.plot(x1, y1, color= self.ax.lines[i].get_color(), alpha=0.7)

        if 'hydro' in sys.argv:
            for i in range(len(datasets_z)):
                v_hydro = dd(datasets_x[i], datasets_z[i], skip).hydrodynamic()['v_hydro']

                self.ax.plot(height_array_mod, v_hydro,
                              ls='--', label=None, color= self.ax.lines[i].get_color(),
                              marker=' ', alpha=0.7)

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
                height_array_mod = dd(datasets_x[0], datasets_z[0], skip).height_array[vx_chunkZ !=0]
                self.ax.axvline(x= height_array_mod[0], color='k',
                                                    linestyle='dashed', lw=1)
                self.ax.axvline(x= height_array_mod[-1], color= 'k',
                                                    linestyle='dashed', lw=1)
            else:
                for i in range(len(datasets_z)):
                    height_array_mod = dd(datasets_x[i], datasets_z[i], skip).height_array[vx_chunkZ !=0]
                    self.ax.axvline(x= height_array_mod[-1], color= 'k',
                                                    linestyle='dashed', lw=1)

        if 'inset' in sys.argv:
            popt2, pcov2 = curve_fit(funcs.quadratic, self.height_arrays_mod[0][1:-1],
                                                        vx_chunkZ[0, :][1:-1])

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

            for i in range(len(datasets_x)):
                mflowrate = dd(datasets_x[i], datasets_z[i], skip).mflux()['mflowrate_stable']
                self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array, mflowrate, ls=lt, marker='o',label=input('Label:'), alpha=opacity)

        if 'mflowrate_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            mflowrate_avg = np.zeros([len(datasets_x)])

            for i in range(len(datasets_x)):
                mflowrate_t[i, :] = dd(datasets_x[i], datasets_z[i], skip).mflux()['mflowrate_stable']
                mflowrate_avg[i] = np.mean(mflowrate_t[i])
                self.ax.plot(self.time*1e-6,  mflowrate_t*1e18, ls='-', marker=' ',label=input('Label:'), alpha=0.5)
                print(self.ax.lines[i].get_color())

            for i in range (len(self.ax.lines)):
                self.ax.axhline(y=mflowrate_avg[i]*1e18, color= self.ax.lines[i].get_color(), linestyle='dashed', lw=1)


    def mflux(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[4])

        if 'jx_length' in sys.argv:
            self.ax.set_xlabel(labels[1])

            for i in range(len(datasets_x)):
                jx = dd(datasets_x[i], datasets_z[i], skip).mflux()[0]
                self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array, jx, ls=lt, marker='o',label=input('Label:'), alpha=opacity)

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
            self.ax.set_xlabel(labels[1])

            for i in range(len(datasets_x)):
                denX = dd(datasets_x[i], datasets_z[i], skip).density()['den_chunkX']
                self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array[1:-1], denX[1:-1],
                            ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'den_height' in sys.argv:
            self.ax.set_xlabel(labels[0])
            for i in range(len(datasets_z)):
                denZ = dd(datasets_x[i], datasets_z[i], skip).density()['den_chunkZ']
                self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).height_array, denZ,
                            ls=lt, label=input('Label:'), marker=mark, alpha=0.5)

        if 'den_time' in sys.argv:
            self.ax.set_xlabel(labels[2])
            bulk_den_avg = np.zeros_like(gap_height_avg)
            for i in range(len(datasets_x)):
                denT = dd(datasets_x[i], datasets_z[i], skip).density()['den_t']
                bulk_den_avg[i] = np.mean(denT[i])
                self.ax.plot(self.time*1e-6, denT, ls=lt, label=input('Label:'),
                            marker=mark, alpha=opacity)


    def vir_len(self, lab=None, err=None, lt='-', mark='o', opacity=1.0, legend=None,
                lab_lines=None, draw_vlines=None):

        pds = plot_from_ds()
        self.ax.set_ylabel(labels[7])
        self.ax.set_xlabel(labels[1])

        vir_chunkX = np.zeros([len(datasets_x), self.Nx-2])
        pump_size = 0.2 #np.float(input('pump size:'))

        for i in range(len(datasets_x)):
            vir_chunkX[i, :] = dd(datasets_x[i], datasets_z[i], skip).virial()['vir_chunkX']

        if err is not None:
            vir_err = np.zeros_like(vir_chunkX)
            for i in range(len(datasets_x)):
                vir_err[i, :] = dd(datasets_x[i], datasets_z[i], skip).virial()['vir_err']
                markers, caps, bars= self.ax.errorbar(dd(datasets_x[i],
                                        datasets_z[i], skip).length_array[1:-1], vir_chunkX[i],
                                        yerr=vir_err, ls=lt, fmt=mark, label=input('label:'),
                                        capsize=1.5, markersize=1.5, alpha=opacity)
                [bar.set_alpha(0.3) for bar in bars]
                [cap.set_alpha(0.3) for cap in caps]

        else:
            for i in range(len(datasets_x)):
                plot = self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array[1:-1], vir_chunkX[i],
                                     ls=lt, label=input('label:'), marker=mark, alpha=0.7)

        if legend is not None:
            self.ax.legend()
        if lab_lines is not None:
            pds.inline(self)
        if draw_vlines is not None:
            pds.draw_vlines(self)


        if 'sigwall' in sys.argv:
            sigzz_chunkX = np.zeros_like([len(datasets_x), self.Nx])
            sigzz_err = np.zeros_like(sigzz_chunkX)


            for i in range(len(datasets_x)):
            sigzz_chunkX[i, :] = dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigzz_chunkX']
            sigzz_err[i, :] = dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigzz_err']


                if 'uncertain' in sys.argv:
                    markers, caps, bars = self.ax.errorbar(dd(datasets_x[i], datasets_z[i], skip).length_array[:-1], sigzz_chunkX,
                                     yerr=sigzz_err, ls=lt, fmt=mark, label=input('Label:'),
                                     capsize=1.5, markersize=1.5, alpha=opacity)

                    [marker.set_alpha(0.3) for bar in bars]
                    [cap.set_alpha(0.3) for cap in caps]

                else:
                    self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array[:-1], sigzz_chunkX,
                                ls=lt, marker='x', label=input('Label:'), alpha=opacity)


        if 'both' in sys.argv:
            for i in range(len(datasets_x)):

                if 'uncertain' in sys.argv:
                    vir_err = dd(datasets_x[i], datasets_z[i], skip).virial()['vir_err']
                    sigzz_err = dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigzz_err']
                    self.ax.errorbar(dd(datasets_x[i], datasets_z[i], skip).length_array[1:-1], vir_chunkX,
                                     yerr=vir_err, ls=lt, fmt=mark, label=input('Label:'),
                                     capsize=1.5, markersize=1.5, alpha=0.5)
                    self.ax.errorbar(dd(datasets_x[i], datasets_z[i], skip).length_array[:-1], sigzz_chunkX,
                                     yerr=sigzz_err, ls=lt, fmt='x', label=input('Label:'),
                                     capsize=1.5, markersize=3, alpha=0.5)
                else:
                    self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array[1:-1], vir_chunkX,
                                ls=lt, marker=' ', label=input('Label:'), alpha=opacity)
                    self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array[:-1], sigzz_chunkX,
                                ls='--', marker=' ', label=input('Label:'),
                                color=self.ax.lines[i].get_color(), alpha=opacity)

        if 'inset' in sys.argv:
            inset_ax = fig.add_axes([0.62, 0.57, 0.2, 0.28]) # X, Y, width, height
            inset_ax.axvline(x=0, color='k', linestyle='dashed')
            inset_ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
            inset_ax.set_ylim(220, 280)
            inset_ax.plot(dd(datasets_x[0], datasets_z[0], skip).length_array[0][1:29],
                         vir_chunkX[0, :][1:29] , ls= lt, color=ax.lines[0].get_color(),
                         marker=None, alpha=opacity, label=label)
            inset_ax.plot(dd(datasets_x[0], datasets_z[0], skip).length_array[0][1:29],
                         vir_chunkX[1, :][1:29] , ls= ' ', color=ax.lines[1].get_color(),
                         marker='x', alpha=opacity, label=label)

        if 'sigxz' in sys.argv:
            self.ax.set_xlabel(labels[1])
            self.ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')

            for i in range(len(datasets_x)):
                sigxz_chunkX = dd(datasets_x[i], datasets_z[i], skip).sigwall()['sigxz_chunkX']
                self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).length_array[1:-1],  sigxz_chunkX[1:-1],
                            ls='-', marker=' ',label=input('Label:'), alpha=0.5)


        if 'press_height' in sys.argv:
            self.ax.set_xlabel(labels[0])

            for i in range(len(datasets_z)):
                vir_chunkZ = dd(datasets_x[i], datasets_z[i], skip).virial()['vir_chunkZ']
                self.ax.plot(dd(datasets_x[i], datasets_z[i], skip).bulk_height_array[1:-1], vir_chunkZ[1:-1],
                            ls=lt, marker=None, label=input('Label:'), alpha=opacity)

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

        if 'pdiff_pumpsize' in sys.argv:

            self.ax.set_xlabel('Normalized pump length')

            pump_size = []
            vir_pdiff, sigzz_pdiff, vir_err, sigzz_err = [], [], [], []
            for i in range(len(datasets_x)):
                pump_size.append(np.float(input('pump size:')))
                vir_pdiff.append(dd(datasets_x[i], datasets_z[i], skip).virial(pump_size[i])['pDiff'])
                vir_err.append(dd(datasets_x[i], datasets_z[i], skip).virial(pump_size[i])['pDiff_err'])
                sigzz_pdiff.append(dd(datasets_x[i], datasets_z[i], skip).sigwall(pump_size[i])['pDiff'])
                sigzz_err.append(dd(datasets_x[i], datasets_z[i], skip).sigwall(pump_size[i])['pDiff_err'])

            markers, caps, bars= self.ax.errorbar(pump_size, vir_pdiff, yerr=vir_err, ls=lt, fmt=mark,
                    label='Virial (Fluid)', capsize=1.5, markersize=1.5, alpha=1)
            markers2, caps2, bars2= self.ax.errorbar(pump_size, sigzz_pdiff, yerr=sigzz_err, ls=lt, fmt='x',
                    label='$\sigma_{zz}$ (Solid)', capsize=1.5, markersize=3, alpha=1)

            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            [bar2.set_alpha(0.5) for bar2 in bars2]
            [cap2.set_alpha(0.5) for cap2 in caps2]

        if 'pdiff_err' in sys.argv:

            self.ax.set_xlabel(labels[7])
            self.ax.set_ylabel('Uncertainty $\zeta$ %')

            pressures = np.array([0.5, 5, 50, 250, 500])
            vir_err, sigzz_err = [], []
            for i in range(len(datasets_x)):
                vir_err.append(dd(datasets_x[i], datasets_z[i], skip).virial(0.2)['pDiff_err'])
                # sigzz_err.append(dd(datasets_x[i], datasets_z[i], skip).sigwall(0.2)['pDiff_err'])

            vir_err_rel = vir_err / pressures
            self.ax.plot(pressures, vir_err_rel, ls='-', marker='o',label='Virial (Fluid)', alpha=1)
            # self.ax.plot(pressures, sigzz_err, ls='-', marker='o',label='$\sigma_{zz}$ (Solid)', alpha=1)

    def temp(self, label=None, err=None, lt='-', mark='o', opacity=1.0):

        self.ax.set_ylabel(labels[6])

        if 'temp_length' in sys.argv:
            tempX = np.zeros([len(datasets_x), self.Nx])
            self.ax.set_xlabel(labels[1])
            for i in range(len(datasets_x)):
                tempX[i, :] = dd(datasets_x[i], datasets_z[i], skip).temp()[0]
                self.ax.plot(self.length_arrays[i, :][1:-1], tempX[i, :][1:-1], ls=lt, label=input('Label:'), marker=mark, alpha=opacity)

        if 'temp_height' in sys.argv:
            tempZ = np.zeros([len(datasets_z), self.Nz])
            self.ax.set_xlabel(labels[0])
            for i in range(len(datasets_z)):
                tempZ[i, :] = dd(datasets_z[i], datasets_z[i], skip).temp()[1]
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
            self.pGrad[i] = dd(datasets_x[i], datasets_z[i], skip).virial(np.float(input('pump size:')))[2]
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

        gap_height_avg = np.zeros([len(datasets_x)])

        for i in range(len(datasets_x)):
            gap_height = dd(datasets_x[i], datasets_z[i], skip).h
            gap_height_avg[i] = dd(datasets_x[i], datasets_z[i], skip).avg_gap_height

            self.ax.plot(self.time*1e-6, gap_height[i, :], ls='-', marker=' ', alpha=opacity, label=input('Label:'))
            self.ax.axhline(y= gap_height_avg, color= 'k', linestyle='dashed', lw=1)

    def weight(self, label=None, err=None, lt='-', mark='o', opacity=1.0):
        self.ax.set_xlabel(labels[1])
        self.ax.set_ylabel('Weight')

        length_padded = np.pad(self.length_arrays[0], (1,0), 'constant')

        self.ax.plot(length_padded, funcs.quartic(length_padded), ls=lt, marker=None, alpha=opacity)
        self.ax.plot(length_padded, funcs.step(length_padded), ls='--', marker=None, alpha=opacity)



#
