#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, logging
import matplotlib.pyplot as plt
from operator import itemgetter
from plot_settings import Initialize, Modify

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
        # Create the figure
        init = Initialize(os.path.abspath(configfile))
        self.fig, self.ax, self.axes_array = itemgetter('fig','ax','axes_array')(init.create_fig())

    def extract_plot(self, x_pre, y_pre, x_col, y_col):
        """
        Extracts data from log files or txt files and plots them.
        Uses the Modify class from 'plot_settings' module.
        """

        if self.filename.startswith('log'): # Plot from LAMMPS log files
            for i in self.txts:
                os.system(f"cat {os.path.dirname(i)}/log.lammps | sed -n '/Step/,/Loop time/p' \
                | head -n-1 > {os.path.dirname(i)}/thermo.out")
                with open(f'{os.path.dirname(i)}/thermo.out', 'r') as f:
                    for line in f:
                        if line.split()[0]!='Step' and line.split()[0]!='Loop':
                            with open(f"{os.path.dirname(i) + '/thermo2.out'}", "a") as o:
                                o.write(line)
                data = np.loadtxt(f"{os.path.dirname(i) + '/thermo2.out'}", skiprows=self.skip, dtype=float)
                xdata = data[:,x_col] * x_pre
                ydata = data[:,y_col] * y_pre
                self.ax.plot(xdata, ydata)
                os.system(f"rm {os.path.dirname(i)+'/thermo2.out'}")
                # self.ax.axhline(y = np.mean(ydata))
        else: # Plot from text files
            for idx, val in enumerate(self.txts):
                data = np.loadtxt(self.txts[idx], skiprows=self.skip, dtype=float)
                xdata = data[:,x_col] * x_pre
                ydata = data[:,y_col] * y_pre
                self.ax.plot(xdata, ydata)

        return {'xdata':xdata, 'ydata':ydata}


    def modify_plot(self, *arr_to_plot):
        """
        Sets the prefactors according to the input quantity and modifies the plot settings
        """

        variables = arr_to_plot[0]

        if 'energy' in variables:
            xdata = self.extract_plot(1e-6, 1, 0, 9) # ns, Kcal/mol
        if 'npump' in variables:
            xdata = self.extract_plot(1e-6, 1, 0, 13) # ns, count
        if 'fw' in variables:
            xdata = self.extract_plot(1e-6, 1e9, 0, 14) # ns, nN
        if 'fp' in variables:
            xdata = self.extract_plot(1e-6, 1e9, 0, 15) # ns, nN
        if 'fpump' in variables:
            xdata = self.extract_plot(1e-6, 1e12, 0, 16) #ns, pN
        if 'fin' in variables:
            xdata = self.extract_plot(1e-6, kcalpermolA_to_N, 0, 17) #ns, pN
        if 'fout' in variables:
            xdata = self.extract_plot(1e-6, kcalpermolA_to_N, 0, 18) #ns, pN
        if 'press_md-cont' in variables:
            xdata = self.extract_plot(1, 1, 0, 1)            #nm, MPa
        if 'radius' in variables:
            xdata = self.extract_plot(1e-6, 1, 0, 1)['xdata']            #nm, MPa

        try:
            Modify(xdata, self.fig, self.axes_array, self.configfile)
        except UnboundLocalError:
            logging.error('No data on the x-axis, check the quanity to plot!')

        return self.fig
