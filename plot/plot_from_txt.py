#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, logging
import matplotlib.pyplot as plt
from operator import itemgetter
from plot_settings import Initialize, Modify
import yaml

# Logger Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Import golbal plot configuration
plt.style.use('imtek')

kcalpermolA_to_N = 6.9477e-11

class PlotFromTxt:

    def __init__(self, skip, filename, txtfiles, configfile):

        self.skip = skip
        self.filename = filename
        self.txts = txtfiles
        self.configfile = configfile
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)
        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())


    def plot_data(self, ax, x, y):
        """
        Plots the raw data
        """

        ax.plot(x, y)
        # ax.axhline(y=np.mean(y))
        # print(np.mean(y))


    def extract_plot(self, *arr_to_plot):
        """
        Extracts data from log files or txt files and plots them.
        Uses the Modify class from 'plot_settings' module.
        """

        variables = arr_to_plot[0]
        nrows = self.config['nrows']

        if self.filename.startswith('log'): # Plot from LAMMPS log files
            for i in self.txts:
                n=0    # subplot
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' \
                | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)

                if 'energy' in variables:
                    x_pre, y_pre, x_col, y_col = 1e-6, 1, 0, 9 # ns, Kcal/mol
                    if nrows>1: n+=1
                if 'npump' in variables:
                    x_pre, y_pre, x_col, y_col = 1e-6, 1, 0, 13 # ns, count
                    if nrows>1: n+=1
                if 'fpump' in variables:
                    # self.axes_array[0].set_xlabel('Time (ns)')
                    # self.axes_array[0].set_ylabel('Force (pN)')
                    xdata, ydata = data[:,0]*1e-6, data[:,16]*kcalpermolA_to_N*1e12 # ns, pN
                    self.plot_data(self.axes_array[n], xdata, ydata)
                    if nrows>1: n+=1
                if 'fw' in variables:
                    # self.axes_array[0].set_xlabel('Time (ns)')
                    # self.axes_array[0].set_ylabel('Force (nN)')
                    xdata, ydata = data[:,0]*1e-6, data[:,14]*1e9 # ns, nN
                    self.plot_data(self.axes_array[n], xdata, ydata)
                    if nrows>1: n+=1
                if 'fp' in variables:
                    # self.axes_array[0].set_xlabel('Time (ns)')
                    # self.axes_array[0].set_ylabel('$f_{pump}$ (pN)')
                    xdata, ydata = data[:,0]*1e-6, data[:,15]*1e9 # ns, nN
                    self.plot_data(self.axes_array[n], xdata, ydata)
                    if nrows>1: n+=1
                if 'fin' in variables:
                    # self.axes_array[0].set_xlabel('Time (ns)')
                    # self.axes_array[0].set_ylabel('Force (nN)')
                    x_pre, y_pre, x_col, y_col = 1e-6, kcalpermolA_to_N*1e9, 0, 17 #ns, nN
                    if nrows>1: n+=1
                if 'fout' in variables:
                    # self.axes_array[0].set_xlabel('Time (ns)')
                    # self.axes_array[0].set_ylabel('Force (nN)')
                    x_pre, y_pre, x_col, y_col = 1e-6, kcalpermolA_to_N*1e9, 0, 18 #ns, nN
                if 'fmomnet' in variables:
                    # self.axes_array[0].set_xlabel('Time (ns)')
                    # self.axes_array[0].set_ylabel('Force (pN)')
                    x_pre, y_pre, x_col, y_col = 1e-6, kcalpermolA_to_N*1e12, 0, 19 #ns, pN

                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")

        else: # Plot from text files

            for idx, val in enumerate(self.txts):
                n=0    # subplot
                data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)

                if 'press_md-cont' in variables:
                    x_pre, y_pre, x_col, y_col = 1, 1, 0, 1            #nm, MPa
                if 'radius' in variables:
                    x_pre, y_pre, x_col, y_col = 1e-6, 1, 0, 1      #ns, Angstrom

                xdata = data[:,x_col] * x_pre
                if data[:,x_col][0]>10: #TODO : make md file start from zero
                    xdata = (data[:,x_col] - data[:,x_col][0]) * x_pre
                ydata = data[:,y_col] * y_pre
                self.plot_data(self.axes_array[n], xdata, ydata)


        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
