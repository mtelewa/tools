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
from compute_4d import ExtractFromTraj as dataset
from plot_settings import Initialize, Modify
from scipy.integrate import trapezoid
import matplotlib.animation as animation

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
            data = dataset(self.skip, self.datasets[i], self.mf, self.pumpsize)
            if self.dimension=='LH': x, y = data.length_array, data.height_array   # nm
            if self.dimension=='LW': x, y = data.length_array, data.width_array
            if self.dimension=='LT': x, y = data.length_array, self.time[self.skip:] * 1e-6      # ns
            if self.dimension=='WH': x, y = data.width_array, data.height_array
            if self.dimension=='WT': x, y  = data.width_array, self.time[self.skip:] * 1e-6
            if self.dimension=='HT': x, y = data.height_array, self.time[self.skip:] * 1e-6

            # if 'H' in self.dimension and any('virial' in var for var in variables) or any('den_bulk' in var for var in variables):
            #     print('Bulk height is used here!')
            #     try:
            #         x = data.bulk_height_array  # simulation with walls
            #     except AttributeError:
            #         x = data.height_array       # bulk simulations

            X, Y = np.meshgrid(x, y)

            # Velocity - x component
            if any('vx' in var for var in variables): array_to_plot = data.velocity()
            # Mass density
            if any('den' in var for var in variables): array_to_plot = data.density()
            # Bulk density
            if any('den_bulk' in var for var in variables): array_to_plot = data.density_bulk()
            # Virial Pressure - Scalar
            if any('virial' in var for var in variables): array_to_plot = data.virial()
            # Virial Pressure - Scalar
            if any('temp' in var for var in variables): array_to_plot = data.temp()

            full_array = array_to_plot['data']
            if self.dimension=='LH': Z = array_to_plot['data_xz']
            if self.dimension=='LW': Z = array_to_plot['data_xy']
            if self.dimension=='LT': Z = array_to_plot['data_xt']
            if self.dimension=='WH': Z = array_to_plot['data_yz']
            if self.dimension=='WT': Z = array_to_plot['data_yt']
            if self.dimension=='HT': Z = array_to_plot['data_zt']

            if self.config['3d']:
                surf = self.ax.plot_surface(X, Y, Z, cmap=mpl.cm.jet,
                    rcount=200, ccount=200 ,linewidth=0.2, antialiased=True)#, linewidth=0.2)
                self.fig.colorbar(surf, shrink=0.5, aspect=5)
                self.ax.zaxis.set_rotate_label(False)

                if any('vx' in var for var in variables): self.ax.set_zlabel(labels[5], labelpad=15)
                if any('den' in var for var in variables): self.ax.set_zlabel(labels[3], labelpad=15)
                if any('virial' in var for var in variables): self.ax.set_zlabel(labels[7], labelpad=15)
                if any('temp' in var for var in variables): self.ax.set_zlabel(labels[6], labelpad=15)

                self.ax.zaxis.set_label_coords(0.5, 1.15)
                self.ax.view_init(35,60) #(35,60)
                self.ax.grid(False)

            if self.config['heat']:
                im = plt.imshow(Z.T, cmap='viridis', interpolation='lanczos',
                    extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower')
                cbar = self.ax.figure.colorbar(im, ax=self.ax)

            if self.config['anim']:
                first_frame, last_frame, interval, fps = 7695, 9800, 30, 2 #30
                frames_list = np.arange(first_frame, last_frame, interval)
                print(f'No. of frames: {len(frames_list)} in the animation')

                #ArtistAnimation -----------------------------------------------
                ims = []
                for i in frames_list:
                    print(f'Rendering frame number: {i}')
                    print(f'Time={i*50/16666:.1f} ns')
                    im = self.ax.imshow(full_array[i,:,:,0].T, cmap='viridis', interpolation='lanczos',
                            extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower', animated=True)

                    if i == first_frame:
                        self.ax.imshow(full_array[i,:,:,0].T, cmap='viridis', interpolation='lanczos',
                            extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower', animated=True)  # show an initial one first
                    ims.append([im])

                anim = animation.ArtistAnimation(plt.gcf(), ims, interval=1000, blit=True)

                # FuncAnimation ------------------------------------------------
                # # plt.pcolormesh(X, Y, full_array[0,:,:,0].T, cmap='viridis')
                # im = plt.imshow(full_array[first_frame,:,:,0].T, cmap='viridis', interpolation='lanczos',
                #     extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto', origin='lower')
                #
                # def init():
                #     plt.title('Time=0')
                #     im.set_data(full_array[first_frame,:,:,0].T)
                #     return [im]
                #
                # def animate(frame_number):
                #     # plt.pcolormesh(X, Y, full_array[frame_number,:,:,0].T, cmap='viridis')
                #     print(f'Rendering frame number: {frame_number}')
                #     print(f'Time={i*50/16666:.1f} ns')
                #     plt.title(f'Time={i*50/16666:.1f} ns')
                #     im.set_array(full_array[frame_number,:,:,0].T)
                #     return [im]
                #
                # anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init, interval=1000, frames=frames_list)

                # plt.draw()
                plt.tight_layout()
                # saving to mp4 using ffmpeg writer
                writervideo = animation.FFMpegWriter(fps=fps, codec='mpeg2video', bitrate=2000)
                anim.save('bubble.mp4', writer=writervideo)
                plt.close()


            # # Mass flux - x component
            # if any('jx' in var for var in variables):
            #     self.axes_array[n].set_ylabel(labels[4])
            #     if self.dimension=='L': arr, y = data.mflux()['jx_full_x'], data.mflux()['jx_X']
            #     if self.dimension=='H': arr, y = data.mflux()['jx_full_z'], data.mflux()['jx_Z']
            #     if self.dimension=='T': arr, y = None, data.mflux()['jx_t']
            #
            # # Heat flux - z component
            # if any('je' in var for var in variables):
            #     self.axes_array[n].set_ylabel('J_e')
            #     if self.dimension=='T': arr, y = None, data.heat_flux()['jez_t']
            #
            # # Mass flowrate - x component
            # if any('mflowrate' in var for var in variables):
            #     self.axes_array[n].set_ylabel(labels[10])
            #     if self.dimension=='L': arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_X']*1e20
            #     if self.dimension=='T': arr, y = data.mflux()['mflowrate_full_x'], data.mflux()['mflowrate_t']*1e20
            #
            # # Solid temperature - Scalar
            # if any('tempS' in var for var in variables):
            #     self.axes_array[n].set_ylabel(labels[6])
            #     try:
            #         if self.dimension=='L': arr, y = data.temp()['temp_full_x_solid'], data.temp()['tempS_len']
            #         if self.dimension=='H': arr, y = data.temp()['temp_full_z_solid'], data.temp()['tempS_height']
            #         if self.dimension=='T': arr, y = None, data.temp()['tempS_t']
            #     except KeyError:
            #         pass

        try:
            Modify(x, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
