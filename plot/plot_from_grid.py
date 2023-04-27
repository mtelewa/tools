#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys, os, logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from operator import itemgetter
import yaml
import funcs
import matplotlib as mpl
import sample_quality as sq
from compute_real import ExtractFromTraj as dataset
from plot_settings import Initialize, Modify
from scipy.integrate import trapezoid

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

# np.seterr(all='raise')

#           0             1                   2                    3
labels=('Height (nm)','Length (nm)', 'Time (ns)', r'Density $\rho$ (g/${\mathrm{cm^3}}$)',
#           4                                           5             6                    7
        r'Mass flux ${\mathrm{j_x}}$ (g/${\mathrm{m^2}}$.ns)', 'Velocity $u$ (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\mathrm{(Force)}}$ (pN)', r'${\mathrm{dP / dx}}$ (MPa/nm)',
#           10                                          11
        r'${\mathrm{\dot{m}}} \times 10^{-20}}$ (g/ns)', r'${\mathrm{\dot{\gamma}} (s^{-1})}$',
#           12
        r'${\mathrm{\eta}}$ (mPa.s)', r'$N_{\mathrm{\pump}}$', r'Energy (Kcal/mol)', '$R(t)$ (${\AA}$)')


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def colorFader(c1, c2, mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

# Color gradient
c1 = 'tab:blue' #blue
c2 = 'tab:red' #red
n = 5
colors_fader = []
for x in range(n+1):
    colors_fader.append(colorFader(c1,c2,x/n))

class PlotFromGrid:

    def __init__(self, skip, dim, datasets_x, datasets_z, mf, configfile, pumpsize):

        self.skip = skip
        self.dimension = dim
        self.mf = mf
        self.datasets_x = datasets_x
        self.datasets_z = datasets_z
        self.configfile = configfile
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)
        self.pumpsize = pumpsize
        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())

        first_dataset = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
        self.Nx = len(first_dataset.length_array)
        self.Nz = len(first_dataset.height_array)
        self.time = first_dataset.time
        self.Ly = first_dataset.Ly

    def plot_data(self, ax, x, y):
        """
        Plots the raw data
        """
        if self.dimension=='L' or self.dimension=='H': ax.plot(x, y)
        if self.dimension=='T':
            ax.plot(x, y)
            if input('Plot time-average?') == 'y':
                ax.axhline(y=np.mean(y))

    def plot_fit(self, ax, x, y):
        """
        Plots the data fit
        """
        fit_data = funcs.fit(x ,y, self.config[f'fit'])['fit_data']
        ax.plot(x, fit_data, lw=1.5)

    def plot_inset(self, xdata, xpos=0.64, ypos=0.23, w=0.23, h=0.17):
        """
        Adds an inset figure
        """
        inset_ax = self.fig.add_axes([xpos, ypos, w, h]) # X, Y, width, height

        # Inset x-axis limit
        inset_ax.set_xlim(left=0.2*np.max(xdata),right=np.max(xdata))#(right=0.2*np.max(xdata))
        # Inset axes labels
        inset_ax.set_xlabel(self.config['inset_xlabel'])
        inset_ax.set_ylabel(self.config['inset_ylabel'])

        return inset_ax

    def plot_uncertainty(self, ax, x, y, arr, color):
        """
        Plots the uncertainty of the data
        """
        if self.config['err_caps']:
            if self.dimension=='L':
                if len(arr.shape)>1:
                    err = sq.get_err(arr)['uncertainty']
                else:
                    err = arr
                markers, caps, bars= ax.errorbar(x, y, xerr=None, yerr=err,
                                            capsize=3.5, markersize=4, lw=2, alpha=0.8, color=color)
            if self.dimension=='H':
                err = sq.get_err(arr)['uncertainty']
                markers, caps, bars= ax.errorbar(x, y, xerr=None, yerr=err,
                                            capsize=3.5, markersize=4, lw=2, alpha=0.8, color=color)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        if self.config['err_fill']:
            lo, hi = sq.get_err(arr)['lo'], sq.get_err(arr)['hi']
            ax.fill_between(x, lo, hi, alpha=0.4)


    def extract_plot(self, *arr_to_plot, **kwargs):
        """
        Extracts data from the 'get_variables' module.
        An iteration plots all the quantities needed for a single dataset then moves
        on to plot all quantities for the next dataset.
        The quantites can be plotted on the same subplot if 'nsubplots variable' (in the config file)
        is set to 1 or they can be plotted separately (nsubplots>1).

        This function calls the plot_<data/fit/..> function and the returned plot is
        then modified with Modify class
        """

        variables = arr_to_plot[0]
        datasets = self.datasets_x

        self.nrows = self.config['nrows']
        self.ncols = self.config['ncols']

        # Label the bottom axis
        if self.dimension=='L': self.axes_array[-1].set_xlabel('Length (nm)')
        if self.dimension=='H': self.axes_array[-1].set_xlabel('Height (nm)')
        if self.dimension=='T': self.axes_array[-1].set_xlabel('Time (ns)')

        for i in range(len(datasets)):
            n=0    # subplot
            data = dataset(self.skip, self.datasets_x[i], self.datasets_z[i], self.mf, self.pumpsize)
            if self.dimension=='L': x = data.length_array   # nm
            if self.dimension=='H': x = data.height_array      # nm
            if self.dimension=='T': x = self.time[self.skip:] * 1e-6      # ns

            # Velocity - x component
            if any('vx' in var for var in variables):
                velocity = data.velocity()
                self.axes_array[n].set_ylabel(labels[5])
                if self.dimension=='L': arr, y = velocity['vx_full_x'], velocity['vx_X']
                if self.dimension=='H': arr, y = velocity['vx_full_z'], velocity['vx_Z']
                if self.dimension=='T': arr, y = None, velocity['vx_t']
                self.plot_data(self.axes_array[n], x, y)
                # Fitting the data
                if self.config[f'fit']:
                    self.plot_fit(self.axes_array[n], x, y)
                if self.config['extrapolate']:
                    slip_length = data.slip_length()
                    x_extrapolate = slip_length['xdata_left']
                    y_extrapolate = slip_length['extrapolate_left']
                    x_extrapolateR = slip_length['xdata_right']
                    y_extrapolateR = slip_length['extrapolate_right']
                    self.axes_array[n].plot(x_extrapolate, y_extrapolate, marker=' ', ls='--', color='k')
                    # self.axes_array[n].plot(x_extrapolateR, y_extrapolateR, marker=' ', ls='--', color='k')
                    # ax.set_xlim([slip_length['root_left'], 0.5*np.max(x)])
                    # ax.set_ylim([0, 1.1*np.max(y)])
                if self.nrows>1: n+=1
                if self.config['broken_axis'] and self.config['plot_on_all']:
                    for n in range(self.nrows): # plot the same data on all the axes
                        self.plot_data(self.axes_array[n], x, y)
                        if self.config[f'fit']: self.plot_fit(self.axes_array[n], x, y) #self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                        n+=1
                if self.config['broken_axis'] and not self.config['plot_on_all']:
                    while n<self.nrows-1: # plot the same data on all the axes except the last one
                        self.plot_data(self.axes_array[n], x, y)
                        if self.config[f'fit']: self.plot_fit(self.axes_array[n], x, y) #self.axes_array[n].plot(x[y!=0][1:-1], fit_data, 'k-',  lw=1.5)
                        n+=1

            # Gap height
            if any('gapheight' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[0])
                if self.dimension=='T': arr, y = None, data.h
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Gap height
            if any('gapdiv' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[0])
                if self.dimension=='T': arr, y = None, data.h_div
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Gap height
            if any('gapconv' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[0])
                if self.dimension=='T': arr, y = None, data.h_conv
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Mass flux - x component
            if any('jx' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[4])
                if self.dimension=='L': arr, y = data.mflux()['jx_full_x'], data.mflux()['jx_X']
                if self.dimension=='H': arr, y = data.mflux()['jx_fuVirialll_z'], data.mflux()['jx_Z']
                if self.dimension=='T': arr, y = None, data.mflux()['jx_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Heat flux - z component
            if any('je' in var for var in variables):
                self.axes_array[n].set_ylabel('J_e')
                if self.dimension=='T': arr, y = None, data.heat_flux()['jez_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Mass flowrate - x component
            if any('mflowrate' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[10])
                if self.dimension=='L': arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_X']*1e20
                if self.dimension=='T': arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_t']*1e20
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial Pressure - Scalar
            if any('virial' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L':
                    arr, y = virial['vir_full_x'], virial['vir_X']
                    # Save the pressure data to a txt file (to compare with the continuum)
                    ymin, ymax = np.argmin(y)-1, np.argmax(y)-1
                    # Include the part before the pump
                    xval = x[ymax:] - x[ymax]
                    # np.savetxt('press-profile-MD.txt', np.c_[xval, y[ymax:]],  delimiter=' ',\
                    #                                 header='Length (nm)              Pressure (MPa)')
                    np.savetxt('press-profile-MD-full.txt', np.c_[x, y],  delimiter=' ',\
                                    header='Length (nm)              Pressure (MPa)')
                if self.dimension=='H':
                    try:
                        x = data.bulk_height_array  # simulation with walls
                    except AttributeError:
                        x = data.height_array       # bulk simulations
                    arr, y = virial['vir_full_z'], virial['vir_Z']
                if self.dimension=='T': arr, y = None, virial['vir_t']

                if self.config['err_fill']:
                    self.plot_data(self.axes_array[n], x, y)
                    self.plot_uncertainty(self.axes_array[n], x, y, virial['vir_full_x'])
                if self.config['err_caps']:
                    self.plot_uncertainty(self.axes_array[n], x, y, virial['vir_full_x'])
                else:
                    self.plot_data(self.axes_array[n], x, y)

                if self.nrows>1: n+=1

            # Mass density
            if any('den' in var for var in variables):
                density = data.density()
                self.axes_array[n].set_ylabel(labels[3])
                if self.dimension=='L': arr, y = density['den_full_x'], density['den_X']
                if self.dimension=='H': arr, y = density['den_full_z'], density['den_Z']
                if self.dimension=='T': arr, y = None, densty['den_t']
                # self.axes_array[n].plot(np.roll(y[y!=0][1:-1],85), x[y!=0][1:-1], color=colors[i])
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial - xx component
            if any('virxx' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = virial['Wxx_full_x'], virial['Wxx_X']
                if self.dimension=='H': arr, y = virial['Wxx_full_z'], virial['Wxx_Z']
                if self.dimension=='T': arr, y = None, virial['Wxx_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial - xy component
            if any('virxy' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = virial['Wxy_full_x'], virial['Wxy_X']
                if self.dimension=='H': arr, y = virial['Wxy_full_z'], virial['Wxy_Z']
                if self.dimension=='T': arr, y = None, virial['Wxy_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial - xz component
            if any('virxz' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = virial['Wxz_full_x'], virial['Wxz_X']
                if self.dimension=='H': arr, y = virial['Wxz_full_z'], virial['Wxz_Z']
                if self.dimension=='T': arr, y = None, virial['Wxz_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial - yy component
            if any('viryy' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = virial['Wyy_full_x'], virial['Wyy_X']
                if self.dimension=='H': arr, y = virial['Wyy_full_z'], virial['Wyy_Z']
                if self.dimension=='T': arr, y = None, virial['Wyy_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial - yz component
            if any('viryz' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = virial['Wyz_full_x'], virial['Wyz_X']
                if self.dimension=='H': arr, y = virial['Wyz_full_z'], virial['Wyz_Z']
                if self.dimension=='T': arr, y = None, virial['Wyz_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            # Virial - zz component
            if any('virzz' in var for var in variables):
                virial = data.virial()
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = virial['Wzz_full_x'], virial['Wzz_X']
                if self.dimension=='H': arr, y = virial['Wzz_full_z'], virial['Wzz_Z']
                if self.dimension=='T': arr, y = None, virial['Wzz_t']
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

            if any('sigzz' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L':
                    arr, y = None, data.sigwall()['sigzz_X']
                    self.plot_data(self.axes_array[n], x, y)
                if self.dimension=='T':
                    arr, y = None, data.sigwall()['sigzz_t']
                    self.plot_data(self.axes_array[n], x, y)
                if self.config['err_caps']:
                    y = data.sigwall()['sigzz_X']
                    err =  data.sigwall()['sigzz_err']
                    self.plot_uncertainty(self.axes_array[n], x, y, err)
                if self.config['err_fill']:
                    lo =  data.sigwall()['sigzz_lo'][1:-1]
                    hi =  data.sigwall()['sigzz_hi'][1:-1]
                    self.plot_data(self.axes_array[n], x, y)
                    self.plot_uncertainty(self.axes_array[n], x, y, err)
                if self.nrows>1: n+=1

            if any('sigxz' in var for var in variables):
                self.axes_array[n].set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                if self.dimension=='L':
                    arr, y = None, data.sigwall()['sigxz_X']
                    self.plot_data(self.axes_array[n], x, y)
                if self.dimension=='T':
                    arr, y = None, data.sigwall()['sigxz_t']
                    self.plot_data(self.axes_array[n], x, y)
                if self.config['err_caps']:
                    # if self.dimension=='L': # Error in eachchunk
                    y = data.sigwall()['sigxz_X']
                    err =  data.sigwall()['sigxz_err']
                    # else:
                    #     err =  data.sigwall()['sigxz_err_t']
                    self.plot_uncertainty(self.axes_array[n], x, y, err)
                if self.config['err_fill']:
                    # if self.dimension=='L': # Error in eachchunk
                    lo =  data.sigwall()['sigxz_lo'][1:-1]
                    hi =  data.sigwall()['sigxz_hi'][1:-1]
                    # else:   # Error in timeseries
                    #     lo =  data.sigwall()['sigxz_lo_t']
                    #     hi =  data.sigwall()['sigxz_hi_t']
                    self.plot_data(self.axes_array[n], x, y)
                    self.plot_uncertainty(self.axes_array[n], x, y, err)
                if self.nrows>1: n+=1

            # Fluid temperature - x component
            if any('tempX' in var for var in variables):
                tempX = data.temp()
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = tempX['tempX_full_x'], tempX['tempX_len']
                if self.dimension=='H': arr, y = tempX['tempX_full_z'], tempX['tempX_height']
                if self.dimension=='T': arr, y = None, tempX['tempX_t']
                self.plot_data(self.axes_array[n], x, y)

            # Fluid temperature - y component
            if any('tempY' in var for var in variables):
                tempY_t = data.temp()
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = tempY['tempY_full_x'], tempY['tempY_len']
                if self.dimension=='H': arr, y = tempY['tempY_full_z'], tempY['tempY_height']
                if self.dimension=='T': arr, y = None, tempY['tempY_t']
                self.plot_data(self.axes_array[n], x, y)

            # Fluid temperature - z component
            if any('tempZ' in var for var in variables):
                tempZ = data.temp()
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = tempZ['tempZ_full_x'], tempZ['tempZ_len']
                if self.dimension=='H': arr, y = tempZ['tempZ_full_z'], tempZ['tempZ_height']
                if self.dimension=='T': arr, y = None, tempZ['tempZ_t']
                self.plot_data(self.axes_array[n], x, y)

            # Fluid temperature - Scalar
            if any('temp' in var for var in variables):
                temp = data.temp()
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = temp['temp_full_x'], temp['temp_X']
                if self.dimension=='H': arr, y = temp['temp_full_z'], temp['temp_Z']
                if self.dimension=='T': arr, y = None, temp['temp_t']
                if self.config['broken_axis'] is None and self.config['err_caps'] is None: self.plot_data(self.axes_array[n], x, y)
                # Fitting the data
                if self.config[f'fit']: self.plot_fit(self.axes_array[n], x, y)
                if self.config['err_fill']:
                    # self.plot_data(self.axes_array[n], x, y)
                    self.plot_uncertainty(self.axes_array[n], x, y,
                                                temp['temp_full_z'], color=plt.gca().lines[-1].get_color())
                if self.config['err_caps']:
                    self.plot_uncertainty(self.axes_array[n], x, y,
                                                temp['temp_full_z'], color=self.config[f'color_{n}'])
                if self.nrows>1: n+=1
                if self.config['broken_axis']:
                    try:
                        if self.config['plot_on_all']:
                            for n in range(self.nrows): # plot the same data on all the axes except the last one
                                self.axes_array[n].plot(x, y)
                                n+=1
                    except KeyError:
                        while n<self.nrows-1: # plot the same data on all the axes except the last one
                            self.axes_array[n].plot(x, y)
                            n+=1

                # np.savetxt('temp-height.txt', np.c_[x, y],  delimiter=' ',\
                #                 header='Length (nm)              Temperature (K))')
                # fit_data = funcs.fit(x[y!=0][2:-2] ,y[y!=0][2:-2], self.config[f'fit'])['fit_data']
                # np.savetxt('temp-height.txt', np.c_[x[y!=0][2:-2]/np.max(x[y!=0][2:-2]), y[y!=0][2:-2], fit_data],  delimiter=' ',\
                #                 header='Height (nm)              Temperature (K)            Fit')

            # Solid temperature - Scalar
            if any('tempS' in var for var in variables):
                tempS = data.temp()
                self.axes_array[n].set_ylabel(labels[6])
                try:
                    if self.dimension=='L': arr, y = tempS['temp_full_x_solid'], tempS['tempS_len']
                    if self.dimension=='H': arr, y = tempS['temp_full_z_solid'], tempS['tempS_height']
                    if self.dimension=='T': arr, y = None, tempS['tempS_t']
                except KeyError:
                    pass
                # if np.mean(tempS['tempS_len'])>1:     # Don't plot data where the solid temp. is zero (for example TF system)
                self.plot_data(self.axes_array[n], x, y)
                if self.nrows>1: n+=1

        if self.config['plot_inset']:

            # TODO : Generalize, data0 is wetting and data4 is non-wetting
            data0 = dataset(self.skip, self.datasets_x[0], self.datasets_z[0], self.mf, self.pumpsize)
            data4 = dataset(self.skip, self.datasets_x[4], self.datasets_z[4], self.mf, self.pumpsize)

            inset_ax = self.plot_inset(data0.length_array[1:-1])
            inset_ax.plot(data0.length_array[1:-1], data0.virial()['vir_X'][1:-1], color='tab:blue')
            inset_ax.plot(data4.length_array[1:-1], data4.virial()['vir_X'][1:-1], color='tab:orange')
            # inset_ax.plot(data4.length_array[1:-1], data4.sigwall()['sigzz_X'][1:-1], ls='--', color='k')

            self.plot_uncertainty(inset_ax, data0.length_array, data0.virial()['vir_X'],
                                            data0.virial()['vir_full_x'], color='tab:blue')
            self.plot_uncertainty(inset_ax, data4.length_array, data4.virial()['vir_X'],
                                            data4.virial()['vir_full_x'], color='tab:orange')


        try:
            Modify(x, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
