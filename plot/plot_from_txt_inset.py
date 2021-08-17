#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from math import atan2
import matplotlib as mpl
import matplotlib.pyplot as plt
import label_lines
from cycler import cycler
from scipy.stats import iqr
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

default_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )

labels=('Height $(nm)$','Length $(nm)$',
        'Density $(g/cm^3)$','Timestep',
        '$j_{x} \;(g/m^2.ns)$',
        'Vx $(m/s)$', 'Temperature $(K)$',
        'Pressure $(MPa)$')

mpl.rcParams.update({'font.size': 18})

mpl.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

mpl.rcParams["figure.figsize"] = (12,10) # the standard figure size
mpl.rcParams["lines.linewidth"] = 1     # line width in points
mpl.rcParams["lines.markersize"] = 8
mpl.rcParams["lines.markeredgewidth"]=1   # the line width around the marker symbol

matplotlib.rcParams.update({'text.usetex': True})

# mpl.rc('axes', prop_cycle=default_cycler)
# mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex
# mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# plt.style.use('default')
# print(plt.style.available)

def plot_from_txt(infile,skip,xdata,ydata,xlabel,ylabel,outfile,label,
                    err=None,lt='-',mark='o',opacity=1.0,
                    plottype='nofit'):

    # Name the output files according to the input
    base=os.path.basename(infile)
    global filename
    filename=os.path.splitext(base)[0]

    # figure(num=None, dpi=80, facecolor='w', edgecolor='k')

    data = np.loadtxt(infile,skiprows=skip,dtype=float)
    x_data = data[:,int(xdata)]
    y_data = data[:,int(ydata)]
    # err = data[:,int(err)]
    
    fig = plt.figure(dpi=100)
    axes1 = fig.add_axes([0.1, 0.1, 0.8, 1.5])
    axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes
    axes1.plot(x, 'r')
    axes1.set_xlabel('x')
    axes1.set_ylabel('y')
    axes1.set_title('title')

    axes2.plot(x, 'g')
    axes2.set_xlabel('x')
    axes2.set_ylabel('y')
    axes2.set_title('title')
