#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4
import re
import numpy as np
import sys
import os
import get_variables_210810 as get
import funcs
import sample_quality as sq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import label_lines
from cycler import cycler
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
labels=('Height (nm)','Length (nm)', r'Density (g/${\rm cm^3}$)','Time (ns)',
#           4                                    5             6                    7
        r'${\rm j_x}$ (g/${\rm m^2}$.ns)', 'Vx (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\rm(F_x)}$ (pN)', r'${\rm \partial p / \partial x}$ (MPa/nm)',
#           10                              11
        r'${\rm \dot{m}}$ (g/ns)', r'${\rm \dot{\gamma} (s^{-1})}$')


datasets = []
skip = np.int(sys.argv[1])

for i in sys.argv:
    if i.endswith('.nc'):
        datasets.append(i)

if 'from_txt' in sys.argv:
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (MPa)')

    data = np.loadtxt('results.txt',skiprows=2, dtype=float)

    xdata = data[:,0]

    ydata1 = data[:,1]
    err1 = data[:,2]

    ydata2 = data[:,3]
    err2 = data[:,4]


    ydata3 = data[:,7]
    err3 = data[:,8]

    ax.errorbar(xdata, ydata1 , yerr=err1, ls='--', fmt=mark, label= 'Expt. (Gehrig et al. 1978)' , capsize=3,alpha=opacity)
    ax.errorbar(xdata, ydata2 , yerr=err2, ls=lt, fmt=mark, label= 'All-atom',capsize=3,alpha=opacity)
    ax.errorbar(xdata, ydata3 , yerr=err3, ls=lt, fmt=mark, label= 'United-atom',capsize=3,alpha=opacity)

if 'from_txt2' in sys.argv:
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

    # fig.tight_layout()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.set_xlabel('Height (nm)')
    ax.set_ylabel(r'${\rm u}$ (m/s)')

    data = np.loadtxt('vx.txt',skiprows=2, dtype=float)

    xdata = data[:,0]
    ydata = data[:,1]

    ax.plot(xdata, ydata, ls=' ', marker='o', alpha=1)

    popt, pcov = curve_fit(funcs.linear, xdata, ydata)
    ax.plot(xdata, funcs.linear(xdata, *popt))


if 'arr' in sys.argv:
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

    ax.set_xscale('log', nonpositive='clip')
    # fig.tight_layout()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    ax.set_xlabel(r'${\rm \dot{\gamma} (s^{-1})} $')
    ax.set_ylabel(r'${\rm \mu}$ (mPa.s)')

    shear_rate_norton = [1.65664630e+08, 8.83541339e+08, 6.55806555e+09, 4.45694829e+10 ,1.22908271e+11]
    shear_rate_thevenin = [6.64248230e+08, 1.46916459e+09, 7.03289920e+09, 4.48788061e+10, 1.24037524e+11]

    viscosity_norton = [0.87595943, 0.51073626, 0.58379492, 0.44121496, 0.27991519]
    viscosity_thevenin = [0.87595943, 0.51073626, 0.58379492, 0.44121496, 0.27991519]

    ax.plot(shear_rate_norton, viscosity_norton,ls='-',marker='o', label='FF')
    ax.plot(shear_rate_thevenin, viscosity_thevenin,ls='-',marker='o', label='FC')

    ax.legend()
    fig.savefig('shearrate_visco.png')


def plot(outfile, label=None, err=None, lt='-', mark='o', opacity=1.0):

    Ly = get.get_data(datasets[0], skip)[27]

    Nx = len(get.get_data(datasets[0], skip)[0])
    Nz = len(get.get_data(datasets[0], skip)[9])
    Nz_mod = len(get.get_data(datasets[0], skip)[4])

    time = get.get_data(datasets[0], skip)[15]

    lengths = np.zeros([len(datasets), Nx])
    heights = np.zeros([len(datasets), Nz])
    heights_mod = np.zeros([len(datasets), Nz_mod])

    height = np.zeros([len(datasets), len(time)])
    den_t = np.zeros_like(height)
    sigzz_t = np.zeros_like(height)
    vir_t = np.zeros_like(height)
    sigxz_t = np.zeros_like(height)
    jx_t = np.zeros_like(height)

    # Pressure difference with mass flux
    avg_sigxz_t, mflux_avg, fx_avg, pGrad, mflowrate_avg, bulk_density_avg = [], [], [], [], [], []

    for i in range(len(datasets)):
        lengths[i, :] = get.get_data(datasets[i], skip)[0]
        heights[i, :] = get.get_data(datasets[i], skip)[9]
        heights_mod[i, :] = get.get_data(datasets[i], skip)[4]

        avg_gap_height = get.get_data(datasets[i], skip)[10]
        print('h = %g' %avg_gap_height)

        bulk_density_avg.append(get.get_data(datasets[i], skip)[28])

        if 'sigxz' in sys.argv:
            avg_sigxz_t.append(get.get_data(datasets[i], skip)[12])
        if 'mflux' in sys.argv:
            mflux_avg.append(get.get_data(datasets[i], skip)[11])
        if 'fx' in sys.argv:
            fx_avg.append(get.get_data(datasets[i], skip)[13])
        if 'pgrad' in sys.argv:
            pGrad.append(get.get_data(datasets[i], skip)[14])
        if 'mflowrate' in sys.argv:
            mflowrate_avg.append(get.get_data(datasets[i], skip)[26])
        if 'sigxz-time' in sys.argv:
            sigxz_t[i, :] =  get.get_data(datasets[i], skip)[24]
        if 'jx-time' in sys.argv:
            jx_t[i, :] = get.get_data(datasets[i], skip)[25]
        if 'press-time' in sys.argv:
            sigzz_t[i, :] =  get.get_data(datasets[i], skip)[23]
            vir_t[i, :] =  get.get_data(datasets[i], skip)[18]
        if 'denx-time' in sys.argv:
            den_t[i, :] = get.get_data(datasets[i], skip)[22]
        if 'height-time' in sys.argv:
            height[i, :] = get.get_data(datasets[i], skip)[16]

    pump_length = 0.2 * np.max(lengths)
    smoothed_pump_length = pump_length * 15/8
    length_padded = np.pad(lengths[0], (1,0), 'constant')

    # if 'vx-height' in sys.argv:
    vx_chunkZ = np.zeros([len(datasets), Nz_mod])
    for i in range(len(datasets)):
        vx_chunkZ[i, :] = get.get_data(datasets[i], skip)[5]
        print('Uc = %g' %np.max(vx_chunkZ[i, :]))

    if 'jx' in sys.argv:
        mpl.rcParams.update({'lines.markersize': 4})
        jx = np.zeros([len(datasets), Nx])
        for i in range(len(datasets)):
            jx[i, :] = get.get_data(datasets[i], skip)[21]

    if 'tempx' in sys.argv:
        tempX = np.zeros([len(datasets), Nx])
        for i in range(len(datasets)):
            tempX[i, :] = get.get_data(datasets[i], skip)[19]

    if 'tempz' in sys.argv:
        tempZ = np.zeros([len(datasets), Nz])
        for i in range(len(datasets)):
            tempZ[i, :] = get.get_data(datasets[i], skip)[20]

    if 'sigwall' in sys.argv and 'evolution' in sys.argv:
        sigzz_chunkXi = get.get_data(datasets[0], skip)[3]        # load

    # if 'virial' in sys.argv and 'evolution' in sys.argv:
    #     vir_chunkXi = get.get_data(datasets[0], skip)[8]        # load

    if 'press' in sys.argv:
        sigzz_chunkX = np.zeros([len(datasets), Nx])
        vir_chunkX = np.zeros([len(datasets), Nx])
        for i in range(len(datasets)):
            vir_chunkX[i, :] = get.get_data(datasets[i], skip)[1]
            sigzz_chunkX[i, :] = get.get_data(datasets[i], skip)[2]

    if 'denz' in sys.argv:
        den_chunkZ = np.zeros([len(datasets), Nz])
        for i in range(len(datasets)):
            den_chunkZ[i, :] = get.get_data(datasets[i], skip)[6]

    if 'denx' in sys.argv:
        den_chunkX = np.zeros([len(datasets), Nx])
        for i in range(len(datasets)):
            den_chunkX[i, :] = get.get_data(datasets[i], skip)[7]


    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=300)

    # fig.tight_layout()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # Name the output files according to the input
    # base = os.path.basename(infile)
    # global filename
    # filename = os.path.splitext(base)[0]
    # figure(num=None, dpi=80, facecolor='w', edgecolor='k')

    if 'tempz' in sys.argv:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[6])

        for i in range(len(datasets)):
            ax.plot(heights[i, :][16:-16], tempZ[i, :][16:-16], ls=lt, marker='o', alpha=opacity)

    if 'tempx' in sys.argv:
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[6])

        for i in range(len(datasets)):
            ax.plot(lengths[i, :][1:-1], tempX[i, :][1:-1], ls=lt, label=input('Label:'), alpha=opacity)

        # label_lines.label_line(ax.lines[0], 5, yoffset= -5, label= r'${\rm \Delta P}$ = 0.5 MPa', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[1], 15, yoffset= -4, label= r'${\rm \Delta P}$ = 5 MPa', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[2], 25, yoffset= -4, label= r'${\rm \Delta P}$ = 50 MPa', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[3], 20, yoffset= -5, label= r'${\rm \Delta P}$ = 500 MPa', fontsize= 8, rotation= -34)
        # label_lines.label_line(ax.lines[4], 30, yoffset= 0, label= r'${\rm \Delta P}$ = 750 MPa', fontsize= 8, rotation= 0)

    if 'jx' in sys.argv:
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[4])

        for i in range(len(datasets)):
            ax.plot(lengths[i, :][1:-1], jx[i, :][1:-1], ls=lt, marker='o', alpha=opacity)

        label_lines.label_line(ax.lines[0], 30, yoffset= 0, label= 'Propane', fontsize= 8, rotation= 0)
        label_lines.label_line(ax.lines[1], 30, yoffset= 0, label= 'Pentane', fontsize= 8, rotation= 0)
        label_lines.label_line(ax.lines[2], 30, yoffset= 0, label= 'heptane', fontsize= 8, rotation= 0)

    if 'pgrad-mflowrate' in sys.argv:

        pGrad_thev = [0.200720545053957, 0.31895678884637785, 1.797746384500307, 8.506924998729314, 16.777136704292456]
        pGrad_thev = np.array(pGrad_thev)

        pGrad_norton = [0.07956470973388886, 0.2181253813406526, 1.6638443639192444, 8.475125094083188, 16.738613944645593]
        pGrad_norton = np.array(pGrad_norton)

        shear_rates = []


        for i in range(len(datasets)):
            # Nearest point to the wall (to evaluate the sheat rate at (For Posieuille flow))
            z1 = heights_mod[i][1]
            # Velocity at the wall
            vx_wall = vx_chunkZ[i][1]
            # Shear rate
            coeffs = np.polyfit(heights_mod[i, :][1:-1], vx_chunkZ[i, :][1:-1], 2)
            shear_rates.append(funcs.quad_slope(z1,coeffs[0],coeffs[1]) *1e9)      # S-1

        avg_sigxz_t = np.array(avg_sigxz_t)
        shear_rates = np.array(shear_rates)
        bulk_density_avg = np.array(bulk_density_avg)

        mu = avg_sigxz_t / shear_rates

        mflowrate_hp =  ((bulk_density_avg*1e3) * (Ly*1e-10) / (12 * (mu*1e6))) \
                                    * pGrad_thev*1e6*1e9 * (avg_gap_height*1e-10)**3
        mflowrate_hp =  np.array(mflowrate_hp) * 1e3 * 1e-9

        mflowrate_avg = np.array(mflowrate_avg)

        mflowrate_norton = [1.9800476e-21, 6.8659246e-21, 4.9087789e-20, 4.4434162e-19, 1.6169272e-18]
        mflowrate_norton = np.array(mflowrate_norton)

        mpl.rcParams.update({'lines.markersize': 4})
        mpl.rcParams.update({'figure.figsize': (12,12)})
        # ax.set_xscale('log', nonpositive='clip')
        #ax.set_yscale('log', nonpositive='clip')

        ax.ticklabel_format(axis='y', style='sci', useOffset=False)

        ax.set_xlabel(labels[9])
        ax.set_ylabel(r'${\rm \dot{m}} \times 10^{-18}}$ (g/ns)')

        ax.plot(pGrad_thev, mflowrate_hp*1e18, ls='--', marker='o', alpha=opacity, label=input('Label:'))
        ax.plot(pGrad_thev, mflowrate_avg*1e18, ls=lt, marker='o', alpha=opacity, label=input('Label:'))
        ax.plot(pGrad_thev, mflowrate_norton*1e18, ls=lt, marker='o', alpha=opacity, label=input('Label:'))

        ax2 = ax.twiny()
        ax2.set_xscale('log', nonpositive='clip')

        ax2.plot(shear_rates, mflowrate_avg, ls= ' ', marker= ' ',
                    alpha=opacity, label=label, color=color[0])

        ax2.set_xlabel(labels[11])


        # err=[]
        # for i in range(len(datasets)):
        #     err.append(sq.get_err(get.get_data(datasets[i], skip)[17])[2])
        #
        # ax.errorbar(pGrad, mflux_avg,yerr=err,ls=lt,fmt=mark,capsize=3,
        #            alpha=opacity)
        # ax.set_ylim(bottom=1e-4)

    if 'pgrad-viscosity' in sys.argv:
        mpl.rcParams.update({'lines.markersize': 4})

        # Shear rate
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(labels[9])
        ax.set_ylabel('Slip Length (nm)')
        ax.plot(pGrad, slip, ls=lt, marker='o', alpha=opacity)


    if 'pgrad-slip' in sys.argv:


        pGrad = [0.0748369639128527,0.18427666125154304,1.6485319785488304, 7.8125, 15.460572160633404]
        shear_rates, vx_wall= [], []




        shear_rates, vx_wall= np.array(shear_rates), np.array(vx_wall)
        Ls = vx_wall/shear_rates

        Ls = [0, 2.42761408e-2, 2.19492683e-1, 4.381412565561818e-1, 7.24803543e-1]

        ax.set_xlabel(labels[9])
        ax.set_ylabel('b (nm)')
        ax.plot(pGrad, Ls, ls=lt, marker='o', alpha=opacity)


    if 'vx-height' in sys.argv:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[5])

        mpl.rcParams.update({'lines.markersize': 3})

        shear_rates, Ls, line_colors, shear_rates_bulk = [], [], [], []
        j = 0

        for i in range(len(datasets)):
            # Nearest point to the wall (to evaluate the sheat rate at (For Posieuille flow))
            z1 = heights_mod[i][1]
            # In the bulk to measure viscosity
            z2 = heights_mod[i][20]
            # velocity at the wall for evaluating slip
            vx_wall = vx_chunkZ[i][1]

            xdata = heights_mod[i, :][1:-1]
            ydata = vx_chunkZ[i, :][1:-1]
            npoints = len(xdata)

            print('Velocity at the wall %g m/s at a distance %g nm from the wall' %(vx_wall,z1))

            label=input('Label:')

            ax.plot(xdata, ydata, ls=' ', label=label, marker='o', alpha=opacity)

            line_colors.append(ax.lines[j].get_color())

            if 'label-lines' in sys.argv:
                label_xpos= np.float(input('Label x-pos:'))
                y_offset = np.float(input('Y-offset for label: '))
                label_lines.label_line(ax.lines[j], label_xpos, yoffset= y_offset, \
                        label= label, fontsize= 8, rotation= 0)

            # Shear Rates
            coeffs_fit = np.polyfit(xdata, ydata, 2)        #returns the polynomial coefficients
            shear_rates.append(funcs.quad_slope(z1,coeffs_fit[0],coeffs_fit[1]) *1e9)      # S-1

            shear_rates_bulk.append(funcs.quad_slope(z2,coeffs_fit[0],coeffs_fit[1]) *1e9)      # S-1

            # Get the slip lengths
            # --------------------
            # construct the polynomial
            f = np.poly1d(coeffs_fit)
            # x-data to fit the parabola
            x1 = np.linspace(xdata[0], xdata[-1], npoints)

            # Plot the fits
            ax.plot(x1, f(x1))

            #Positions ro inter/extrapolate
            x2 = np.linspace(-22, xdata[1], npoints)
            # spline order: 1 linear, 2 quadratic, 3 cubic ...
            order = 1
            # do inter/extrapolation
            extrapolate = InterpolatedUnivariateSpline(heights_mod[i, :][1:-1], f(x1), k=order)

            coeffs_extrapolate = np.polyfit(x2, extrapolate(x2), 1)
            # Slip lengths (extrapolated)
            roots = np.roots(coeffs_extrapolate)
            Ls.append(np.abs(roots[-1])*1e-9)      #  m

            if 'extrapolate' in sys.argv:
                ax.set_ylim(bottom = 0)
                # Plot the extrapolated lines
                ax.plot(x2, extrapolate(x2), color='sienna')

            j = len(ax.lines)

            # if 'fit' in sys.argv:
            #     popt, pcov = curve_fit(funcs.quadratic, xdata, ydata)
            #     ax.plot(xdata, ydata, *popt))

        if 'inset' in sys.argv:
            popt2, pcov2 = curve_fit(funcs.quadratic, heights_mod[0][1:-1], vx_chunkZ[0, :][1:-1])

            inset_ax = fig.add_axes([0.6, 0.48, 0.2, 0.28]) # X, Y, width, height
            inset_ax.plot(heights_mod[0][-31:-1], vx_chunkZ[0, :][-31:-1] , ls= ' ', marker=mark,
                        alpha=opacity, label=label)
            inset_ax.plot(heights_mod[0][-31:-1], funcs.quadratic(heights_mod[0], *popt2)[-31:-1])

        # plot vertical lines for the walls
        if 'walls' in sys.argv:
            ax.axvline(x=0, color='k', linestyle='dashed', lw=1)
            if len(datasets) == 1:
                ax.axvline(x= heights[0][-1], color= 'k', linestyle='dashed', lw=1)
            else:
                for i in range(len(datasets)):
                    ax.axvline(x= heights[i][-1], color= line_colors[i], linestyle='dashed', lw=1)

        Ls = np.array(Ls)
        shear_rates = np.array(shear_rates)
        shear_rates_bulk = np.array(shear_rates_bulk)
        # Viscosities
        mu = avg_sigxz_t / shear_rates
        # Slip velocities according to Navier boundary
        Vs = Ls * shear_rates

        print('Viscosity (mPa.s) -----')
        print(mu*1e9)       # mPa.s
        print('Slip Length (nm) -----')
        print(Ls*1e9)       # nm
        print('Shear rate (s^-1) -----')
        print(shear_rates)  # s^-1
        print('Slip velocity: Navier boundary (m/s) -----')
        print(Vs)           # m/s

        # ax.axvline(x= heights[1][-1], color= ax.lines[2].get_color(), linestyle='dashed', lw=1)
        # ax.axvline(x= heights[2][-1], color= ax.lines[4].get_color(), linestyle='dashed', lw=1)

        # label_lines.label_line(ax.lines[0], label_xpos[0], yoffset= -3, label='Non-wetting', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[2], label_xpos[1], yoffset= -3.2, label='Non-wetting ($P_{ext}$=100 MPa)', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[4], label_xpos[2], yoffset= -2.8, label='Wetting', fontsize= 8, rotation= 0)

    if 'vx-distrib' in sys.argv:
        ax.set_xlabel(labels[0])
        ax.set_ylabel('Probability')
        values = get.get_data(datasets[0], skip)[6]
        probabilities = get.get_data(datasets[0], skip)[7]

        ax.plot(values, probabilities, ls=' ', marker='o', alpha=opacity)


    if 'height-time' in sys.argv:
        ax.set_xlabel(labels[3])
        ax.set_ylabel(labels[0])
        for i in range(len(datasets)):
            ax.plot(time*1e-6, height[i], ls='-', marker=' ', alpha=opacity)

        # label_xpos = [2,2,2]
        #
        # label_lines.label_line(ax.lines[0], label_xpos[0], yoffset= 0, label='Propane', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[1], label_xpos[1], yoffset= 0, label='Pentane', fontsize= 8, rotation= 0)
        # label_lines.label_line(ax.lines[2], label_xpos[2], yoffset= 0, label='Heptane', fontsize= 8, rotation= 0)

    if 'jx-time' in sys.argv:
        ax.set_xlabel(labels[3])
        ax.set_ylabel(labels[4])
        for i in range(len(datasets)):
            ax.plot(time*1e-6,  jx_t[i, :], ls='-', marker=' ',label=input('Label:'), alpha=0.5)

    if 'sigxz-time' in sys.argv:
        ax.set_xlabel(labels[3])
        ax.set_ylabel('Wall $\sigma_{xz}$ (MPa)')
        for i in range(len(datasets)):
            ax.plot(time[:5000]*1e-6,  sigxz_t[i, :5000], ls='-', marker=' ',label=input('Label:'), alpha=0.5)
            ax.axhline(y=avg_sigxz_t[i], color=color[i], linestyle='dashed')

    if 'press-time' in sys.argv:
        ax.set_xlabel(labels[3])
        ax.set_ylabel(labels[7])
        if 'virial' in sys.argv:
            for i in range(len(datasets)):
                ax.plot(time*1e-6,  vir_t[i, :], ls='-', marker=' ', alpha=0.5)
        if 'sigwall' in sys.argv:
            for i in range(len(datasets)):
                ax.plot(time*1e-6,  sigzz_t[i, :], ls='-', marker=' ', alpha=0.5)

    if 'press' in sys.argv:
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[7])

        if 'both' in sys.argv:
            for i in range(len(datasets)):
                ax.plot(lengths[i][1:-1], vir_chunkX[i, :][1:-1], ls=lt, marker=None, label=input('Label:'), alpha=opacity)
            for i in range(len(datasets)):
                ax.plot(lengths[i][1:-1], sigzz_chunkX[i, :][1:-1], ls='--', marker=None, label=input('Label:'), alpha=opacity)

            # plt.yticks(np.arange(30, 80, step=10))  # Set label locations.

        if 'sigwall' in sys.argv:
            for i in range(len(datasets)):
                ax.plot(lengths[i][1:-1], sigzz_chunkX[i, :][1:-1], ls=lt, marker=None, label=input('Label:'), alpha=opacity)

        if 'virial' in sys.argv:
            for i in range(len(datasets)):
                ax.plot(lengths[i][2:-2], vir_chunkX[i, :][2:-2], ls=lt, marker='o', label=input('Label:'), alpha=opacity)

            # ax.plot(lengths[1][1:-1], vir_chunkX[1, :][1:-1], ls=' ', color=ax.lines[0].get_color(), marker='x', alpha=opacity)
            # ax.plot(lengths[2][1:-1], vir_chunkX[2, :][1:-1], ls=lt, marker=None, label=input('Label:'), alpha=opacity)
            # ax.plot(lengths[3][1:-1], vir_chunkX[3, :][1:-1], ls=' ', color=ax.lines[2].get_color(), marker='x', alpha=opacity)
            # ax.plot(lengths[4][1:-1], vir_chunkX[4, :][1:-1], ls=lt, marker=None, label=input('Label:'), alpha=opacity)
            # ax.plot(lengths[5][1:-1], vir_chunkX[5, :][1:-1], ls=' ', color=ax.lines[4].get_color(), marker='x', label=input('Label:'), alpha=opacity)

        # if 'label-lines' in sys.argv:
            # label_lines.label_line(ax.lines[0], 15, yoffset= -1, label='Norton (Fluid)', rotation= -36)
            # label_lines.label_line(ax.lines[1], 25, yoffset= -1, label='Thevenin (Fluid)', rotation= -36)
            # label_lines.label_line(ax.lines[2], 20, yoffset= 1, label='Norton (Walls)', rotation= -36)
            # label_lines.label_line(ax.lines[3], 30, yoffset= 1, label='Thevenin (Walls)', rotation= -36)

        if 'label-lines' in sys.argv and 'virial-big' in sys.argv:
            label_lines.label_line(ax.lines[0], 30, yoffset= -3, label=r'${\rm \Delta P}$ = 50 MPa', rotation= -7)
            label_lines.label_line(ax.lines[2], 30, yoffset= -3, label=r'${\rm \Delta P}$ = 250 MPa', rotation= -28)
            label_lines.label_line(ax.lines[4], 30, yoffset= -3, label=r'${\rm \Delta P}$ = 500 MPa', rotation= -40)

        if 'inset' in sys.argv:
            inset_ax = fig.add_axes([0.62, 0.57, 0.2, 0.28]) # X, Y, width, height
            inset_ax.axvline(x=0, color='k', linestyle='dashed')
            inset_ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
            inset_ax.set_ylim(220, 280)
            inset_ax.plot(lengths[0][1:29], vir_chunkX[0, :][1:29] , ls= lt, color=ax.lines[0].get_color(), marker=None, alpha=opacity, label=label)
            inset_ax.plot(lengths[0][1:29], vir_chunkX[1, :][1:29] , ls= ' ', color=ax.lines[1].get_color(), marker='x', alpha=opacity, label=label)

    if 'denx' in sys.argv:
        ax.set_xlabel(labels[1])
        ax.set_ylabel(labels[2])
        for i in range(len(datasets)):
            ax.plot(lengths[i][1:-1], den_chunkX[i, :][1:-1], ls=lt, label=input('Label:'), marker=None, alpha=opacity)

    if 'denx-time' in sys.argv:
        ax.set_xlabel(labels[3])
        ax.set_ylabel(labels[2])
        for i in range(len(datasets)):
            ax.plot(time*1e-6, den_t[i, :], ls= lt, label=input('Label:'), marker='o', alpha=opacity)

    if 'denz' in sys.argv:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[2])
        for i in range(len(datasets)):
            print('ok')
            ax.plot(heights[i], den_chunkZ[i, :], ls=lt, marker='o', label=input('Label:') , alpha=0.7)
        if len(datasets) == 1:
            ax.axvline(x= heights[0][-1], color= 'k', linestyle='dashed', lw=1)

            ax.axvline(x=2,  color= 'r', linestyle='dashed', lw=1)
            ax.axvline(x=3,  color= 'r', linestyle='dashed', lw=1)

            ax.axvline(x=1.08,  color= 'g', linestyle='dashed', lw=1)
            ax.axvline(x=4.18,  color= 'g', linestyle='dashed', lw=1)
        else:
            for i in range(len(datasets)):
                ax.axvline(x= heights[i][-1], color= ax.lines[i].get_color(), linestyle='dashed', lw=1)
                print(heights[i][-1])

        # plot vertical lines for the walls
        ax.axvline(x=0, color='k', linestyle='dashed', lw=1)

    if 'triv' in sys.argv:
        ax.set_xlabel(labels[1])
        ax.set_ylabel('Weight')
        ax.plot(length_padded, funcs.quartic(length_padded), ls=lt, marker=None, alpha=opacity)
        ax.plot(length_padded, funcs.step(length_padded), ls='--', marker=None, alpha=opacity)

    if 'pump' in sys.argv:
        ax.axvline(x=0, color='k', linestyle='dashed')
        ax.axvline(x=0.2*np.max(lengths), color='k', linestyle='dashed')
        if 'smooth' in sys.argv:
            ax.axvline(x=0.375*np.max(lengths), color='k', linestyle='dashed')


    if 'legend' in sys.argv:
        ax.legend()


    fig.savefig(outfile + '.eps', format='eps')
    # ax.set_title('title')

    if 'plot_from_txt' in sys.argv:
        exit()


if __name__ == "__main__":
        plot(sys.argv[-1])





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
