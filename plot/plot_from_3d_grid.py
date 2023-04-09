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
from compute_thermo_3d import ExtractFromTraj as dataset
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
        r'${\mathrm{j_x}}$ (g/${\mathrm{m^2}}$.ns)', 'Velocity $u$ (m/s)', 'Temperature (K)', 'Pressure (MPa)',
#           8                                   9
        r'abs${\mathrm{(Force)}}$ (pN)', r'${\mathrm{dP / dx}}$ (MPa/nm)',
#           10                                          11
        r'${\mathrm{\dot{m}}} \times 10^{-20}}$ (g/ns)', r'${\mathrm{\dot{\gamma}} (s^{-1})}$',
#           12
        r'${\mathrm{\eta}}$ (mPa.s)', r'$N_{\mathrm{\pump}}$', r'Energy (Kcal/mol)', '$R(t)$ (${\AA}$)')

def colorFader(c1, c2, mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

# Color gradient
c1 = 'tab:blue' #blue
c2 = 'tab:red' #red
n = 5
colors = []
for x in range(n+1):
    colors.append(colorFader(c1,c2,x/n))

class PlotFromGrid:

    def __init__(self, skip, dim, datasets, mf, configfile, pumpsize):

        self.skip = skip
        self.dimension = dim
        self.mf = mf
        self.datasets = datasets
        self.configfile = configfile
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)
        self.pumpsize = pumpsize
        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())

        # try:
        first_dataset = dataset(self.skip, self.datasets[0], self.mf, self.pumpsize)
        self.Nx = len(first_dataset.length_array)
        self.Ny = len(first_dataset.width_array)
        self.Nz = len(first_dataset.height_array)
        self.time = first_dataset.time
        self.Ly = first_dataset.Ly


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
        datasets = self.datasets

        self.nrows = self.config['nrows']
        self.ncols = self.config['ncols']

        # Label the axes
        if self.dimension=='LH':
            self.ax.set_xlabel('Length (nm)')
            self.ax.set_ylabel('Height (nm)')
        if self.dimension=='LW':
            self.ax.set_xlabel('Length (nm)')
            self.ax.set_ylabel('Width (nm)')
        if self.dimension=='LT':
            self.ax.set_xlabel('Length (nm)')
            self.ax.set_ylabel('Time (ns)')
        if self.dimension=='WH':
            self.ax.set_xlabel('Width (nm)')
            self.ax.set_ylabel('Height (ns)')
        if self.dimension=='WT':
            self.ax.set_xlabel('Width (nm)')
            self.ax.set_ylabel('Time (ns)')
        if self.dimension=='HT':
            self.ax.set_xlabel('Height (nm)')
            self.ax.set_ylabel('Time (ns)')

        for i in range(len(datasets)):
            n=0    # subplot
            data = dataset(self.skip, self.datasets[i], self.mf, self.pumpsize)
            if self.dimension=='LH':
                x = data.length_array   # nm
                y = data.height_array
            if self.dimension=='LW':
                x = data.length_array
                y = data.width_array
                print(x)
                print(y)
            if self.dimension=='LT':
                x = data.length_array
                y = self.time[self.skip:] * 1e-6      # ns
            if self.dimension=='WH':
                x = data.width_array
                y = data.height_array
            if self.dimension=='WT':
                x = data.width_array
                y = self.time[self.skip:] * 1e-6
            if self.dimension=='HT':
                x = data.height_array
                y = self.time[self.skip:] * 1e-6

            X, Y = np.meshgrid(x, y)

            # Velocity - x component
            if any('vx' in var for var in variables):
                if self.dimension=='LH': Z = data.velocity()['data_xz']
                if self.dimension=='LW': Z = data.velocity()['data_xy']
                if self.dimension=='LT': Z = data.velocity()['data_xt']
                if self.dimension=='WH': Z = data.velocity()['data_yz']
                if self.dimension=='WT': Z = data.velocity()['data_yt']
                if self.dimension=='HT': Z = data.velocity()['data_zt']

                if self.config['3d']:
                    surf = self.ax.plot_surface(X, Y, Z, cmap=mpl.cm.jet,
                        rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
                    self.fig.colorbar(surf, shrink=0.5, aspect=5)
                    self.ax.zaxis.set_rotate_label(False)
                    self.ax.set_zlabel(labels[5], labelpad=15)
                    self.ax.zaxis.set_label_coords(0.5, 1.15)
                    self.ax.view_init(35,60) #(35,60)
                    self.ax.grid(False)
                if self.config['heat']:
                    im = plt.imshow(Z, cmap='viridis', interpolation='lanczos',
                        extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower')
                    cbar = self.ax.figure.colorbar(im, ax=self.ax)

            # Gap height
            if any('gapheight' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[0])
                if self.dimension=='T': arr, y = None, data.h

            # Gap height
            if any('gapdiv' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[0])
                if self.dimension=='T': arr, y = None, data.h_div

            # Gap height
            if any('gapconv' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[0])
                if self.dimension=='T': arr, y = None, data.h_conv

            # Mass flux - x component
            if any('jx' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[4])
                if self.dimension=='L': arr, y = data.mflux()['jx_full_x'], data.mflux()['jx_X']
                if self.dimension=='H': arr, y = data.mflux()['jx_full_z'], data.mflux()['jx_Z']
                if self.dimension=='T': arr, y = None, data.mflux()['jx_t']

            # Heat flux - z component
            if any('je' in var for var in variables):
                self.axes_array[n].set_ylabel('J_e')
                if self.dimension=='T': arr, y = None, data.heat_flux()['jez_t']

            # Mass flowrate - x component
            if any('mflowrate' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[10])
                if self.dimension=='L': arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_X']*1e20
                if self.dimension=='T': arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_t']*1e20

            # Virial Pressure - Scalar
            if any('virial' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L':
                    arr, y = data.virial()['vir_full_x'], data.virial()['vir_X']
                    # Save the pressure data to a txt file (to compare with the continuum)
                    ymin, ymax = np.argmin(y)-1, np.argmax(y)-1
                    # Include the part before the pump
                    xval = x[ymax:] - x[ymax]
                if self.dimension=='H':
                    try:
                        x = data.bulk_height_array  # simulation with walls
                    except AttributeError:
                        x = data.height_array       # bulk simulations
                    arr, y = data.virial()['vir_full_z'], data.virial()['vir_Z']
                if self.dimension=='T': arr, y = None, data.virial()['vir_t']

            # Mass density
            if any('den' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[3])
                if self.dimension=='L': arr, y = data.density()['den_full_x'], data.density()['den_X']
                if self.dimension=='H': arr, y = data.density()['den_full_z'], data.density()['den_Z']
                if self.dimension=='T': arr, y = None, data.density()['den_t']
                # self.axes_array[n].plot(np.roll(y[y!=0][1:-1],85), x[y!=0][1:-1], color=colors[i])

            # Virial - xx component
            if any('virxx' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = data.virial()['Wxx_full_x'], data.virial()['Wxx_X']
                if self.dimension=='H': arr, y = data.virial()['Wxx_full_z'], data.virial()['Wxx_Z']
                if self.dimension=='T': arr, y = None, data.virial()['Wxx_t']

            # Virial - xy component
            if any('virxy' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = data.virial()['Wxy_full_x'], data.virial()['Wxy_X']
                if self.dimension=='H': arr, y = data.virial()['Wxy_full_z'], data.virial()['Wxy_Z']
                if self.dimension=='T': arr, y = None, data.virial()['Wxy_t']

            # Virial - xz component
            if any('virxz' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = data.virial()['Wxz_full_x'], data.virial()['Wxz_X']
                if self.dimension=='H': arr, y = data.virial()['Wxz_full_z'], data.virial()['Wxz_Z']
                if self.dimension=='T': arr, y = None, data.virial()['Wxz_t']

            # Virial - yy component
            if any('viryy' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = data.virial()['Wyy_full_x'], data.virial()['Wyy_X']
                if self.dimension=='H': arr, y = data.virial()['Wyy_full_z'], data.virial()['Wyy_Z']
                if self.dimension=='T': arr, y = None, data.virial()['Wyy_t']

            # Virial - yz component
            if any('viryz' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = data.virial()['Wyz_full_x'], data.virial()['Wyz_X']
                if self.dimension=='H': arr, y = data.virial()['Wyz_full_z'], data.virial()['Wyz_Z']
                if self.dimension=='T': arr, y = None, data.virial()['Wyz_t']

            # Virial - zz component
            if any('virzz' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L': arr, y = data.virial()['Wzz_full_x'], data.virial()['Wzz_X']
                if self.dimension=='H': arr, y = data.virial()['Wzz_full_z'], data.virial()['Wzz_Z']
                if self.dimension=='T': arr, y = None, data.virial()['Wzz_t']

            if any('sigzz' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[7])
                if self.dimension=='L':
                    arr, y = None, data.sigwall()['sigzz_X']

                if self.dimension=='T':
                    arr, y = None, data.sigwall()['sigzz_t']

                if self.config['err_caps']:
                    y = data.sigwall()['sigzz_X']
                    err =  data.sigwall()['sigzz_err']
                    self.plot_uncertainty(self.axes_array[n], x, y, err)
                if self.config['err_fill']:
                    lo =  data.sigwall()['sigzz_lo'][1:-1]
                    hi =  data.sigwall()['sigzz_hi'][1:-1]

                    self.plot_uncertainty(self.axes_array[n], x, y, err)

            if any('sigxz' in var for var in variables):
                self.axes_array[n].set_ylabel('Wall $\sigma_{xz}$ (MPa)')
                if self.dimension=='L':
                    arr, y = None, data.sigwall()['sigxz_X']

                if self.dimension=='T':
                    arr, y = None, data.sigwall()['sigxz_t']

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

                    self.plot_uncertainty(self.axes_array[n], x, y, err)

            # Fluid temperature - x component
            if any('tempX' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = data.temp()['tempX_full_x'], data.temp()['tempX_len']
                if self.dimension=='H': arr, y = data.temp()['tempX_full_z'], data.temp()['tempX_height']
                if self.dimension=='T': arr, y = None, data.temp()['tempX_t']

            # Fluid temperature - y component
            if any('tempY' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = data.temp()['tempY_full_x'], data.temp()['tempY_len']
                if self.dimension=='H': arr, y = data.temp()['tempY_full_z'], data.temp()['tempY_height']
                if self.dimension=='T': arr, y = None, data.temp()['tempY_t']

            # Fluid temperature - z component
            if any('tempZ' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = data.temp()['tempZ_full_x'], data.temp()['tempZ_len']
                if self.dimension=='H': arr, y = data.temp()['tempZ_full_z'], data.temp()['tempZ_height']
                if self.dimension=='T': arr, y = None, data.temp()['tempZ_t']

            # Fluid temperature - Scalar
            if any('temp' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[6])
                if self.dimension=='L': arr, y = data.temp()['temp_full_x'], data.temp()['temp_X']
                if self.dimension=='H': arr, y = data.temp()['temp_full_z'], data.temp()['temp_Z']
                if self.dimension=='T': arr, y = None, data.temp()['temp_t']

                if self.config['broken_axis']:
                    try:
                        if self.config['plot_on_all']:
                            for n in range(self.nrows): # plot the same data on all the axes except the last one
                                self.axes_array[n].plot(x, y[1:-1])
                                n+=1
                    except KeyError:
                        while n<self.nrows-1: # plot the same data on all the axes except the last one
                            self.axes_array[n].plot(x, y[1:-1])
                            n+=1
                # fit_data = funcs.fit(x[y!=0][2:-2] ,y[y!=0][2:-2], self.config[f'fit'])['fit_data']
                # np.savetxt('temp-height.txt', np.c_[x[y!=0][2:-2]/np.max(x[y!=0][2:-2]), y[y!=0][2:-2], fit_data],  delimiter=' ',\
                #                 header='Height (nm)              Temperature (K)            Fit')
            # Solid temperature - Scalar
            if any('tempS' in var for var in variables):
                self.axes_array[n].set_ylabel(labels[6])
                try:
                    if self.dimension=='L': arr, y = data.temp()['temp_full_x_solid'], data.temp()['tempS_len']
                    if self.dimension=='H': arr, y = data.temp()['temp_full_z_solid'], data.temp()['tempS_height']
                    if self.dimension=='T': arr, y = None, data.temp()['tempS_t']
                except KeyError:
                    pass

        try:
            Modify(x, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
