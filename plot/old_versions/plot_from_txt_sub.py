#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4

import numpy as np
import sys
import os
from math import atan2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
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

# print(type(default_cycler))

labels=('Height $(nm)$','Length $(nm)$',
        'Density $(g/cm^3)$','Timestep',
        '$j_{x} \;(g/m^2.ns)$',
        'Vx $(m/s)$', 'Temperature $(K)$',
        'Pressure $(MPa)$')


mpl.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# mpl.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex
mpl.rc('axes', prop_cycle=default_cycler)
mpl.rcParams["figure.figsize"] = (12,10) # the standard figure size
mpl.rcParams["lines.linewidth"] = 1     # line width in points
mpl.rcParams["lines.markersize"] = 8
mpl.rcParams["lines.markeredgewidth"]=1   # the line width around the marker symbol
# mpl.rcParams.update({'text.usetex': True})
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif':'Liberation Sans'})

# plt.style.use('default')
# print(plt.style.available)

def plot_from_txt(infile,skip,xdata,ydata,xlabel,ylabel,outfile,label=None,
                    err=None,lt='-',mark='o',opacity=1.0,xdata2=0,ydata2=1,infile2=None,
                    ylabel2=None,plottype='nofit',format=None,twin=None,inset=None):

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False)
    # fig.tight_layout()

    # Name the output files according to the input
    base=os.path.basename(infile)
    global filename
    filename=os.path.splitext(base)[0]

    # figure(num=None, dpi=80, facecolor='w', edgecolor='k')

    data = np.loadtxt(infile,skiprows=skip,dtype=float)
    x_data = data[:,int(xdata)]
    y_data = data[:,int(ydata)]

    scale_x = 1
    scale_y = 1

    def func(x,a,b,c):
        return a*x**2+b*x+c

    def func2(x,a,b):
        return a*x+b

    popt, pcov = curve_fit(func, x_data, y_data)
    popt2, pcov2 = curve_fit(func2, x_data, y_data)

    #Error bar with Quadratic fitting
    if plottype=='errquad':
        err = data[:,int(err)]
        ax.plot(x_data,func(x_data,*popt),ls=lt,marker=mark,
                   alpha=opacity,label=None)
        ax.errorbar(x_data,y_data,yerr=err,ls=lt,fmt=markerstyle,capsize=3,
                   alpha=opacity,label=label)

    #No errorbar With Quadratic fitting
    if plottype=='quad':
        ax.plot(x_data,func(x_data,*popt),ls=lt,marker=None,
                   alpha=opacity,label=None)
        ax.plot(x_data,y_data,marker=mark,
                   alpha=opacity,label=label)

    #No errorbar Without linear fitting
    if plottype=='linear':
        ax.plot(x_data,func2(x_data,*popt2),ls=lt,marker=None,
                   alpha=opacity,label=None)
        ax.plot(x_data,y_data,ls=None,marker=mark,
                   alpha=opacity,label=label)

    #No errorbar Without fitting
    if plottype=='nofit':
        plt.axhline(y=0.08663375, color='r', linestyle='dashed', label=' imposed')
        plt.axhline(y=0.08742073, color='g', linestyle='dashed', label=' measured')
        ax.plot(x_data*scale_x,y_data*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)

    #Erorbar Without fitting
    if plottype=='errnofit':
        plt.axvline(x=0.125, color='r', linestyle='dashed', label=' pump inlet')
        plt.axvline(x=2.38, color='b', linestyle='dashed', label=' pump outlet')
        err = data[:,int(err)]
        ax.plot(x_data*scale_x,y_data*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)
        ax.errorbar(x_data,y_data,yerr=err,ls=lt,fmt=mark,capsize=3,
                   alpha=opacity,label=label)

    if plottype=='log':
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(x_data*scale_x,y_data*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)

    if twin=='yes':
        data2 = np.loadtxt(infile2,skiprows=skip,dtype=float)
        y_data2 = data2[:,int(ydata2)]
        ax2 = ax.twinx()
        ax2.plot(x_data*scale_x,y_data2*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label,color=u'#ff7f0e')
        ax2.set_ylabel(r'$\rho \, (g/cm^3)$')
        # color_cycle = ax._get_lines.prop_cycler
        # for label in ax2.get_yticklabels():
        #     label.set_color()

    if inset=='yes':
        plt.tight_layout()
        data2 = np.loadtxt(infile2,skiprows=skip,dtype=float)
        x_data2 = data2[:,int(xdata2)]
        y_data2 = data2[:,int(ydata2)]
        inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height
        inset_ax.plot(x_data2*scale_x,y_data2*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)
        # set axis tick locations
        # inset_ax.set_yticks([0, 0.005, 0.01])
        # inset_ax.set_xticks([-0.1,0,.1]);

    if format=='power':
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.yaxis.set_major_formatter(formatter)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(outfile, dpi=100)
    # ax.set_title('title')
    ax.legend()

if __name__ == "__main__":
    if 'twin' in sys.argv:
        plot_from_txt(sys.argv[1],2,0,1,sys.argv[2],sys.argv[3],
                         'fig.png',opacity=0.6,twin='yes',infile2='density.txt')

    if 'inset' in sys.argv:
        plot_from_txt(sys.argv[1],2,0,1,sys.argv[2],sys.argv[3],
                         'fig.png',opacity=0.6,inset='yes',infile2='denX-time.txt')

    if 'err' in sys.argv:
        plot_from_txt(sys.argv[1],1,0,1,'Length $(nm)$','Pressure $(MPa)$', 'press-err.png',err=2,
                         plottype='errnofit',opacity=0.6)

    else:
        plot_from_txt(sys.argv[1],1,0,1,sys.argv[2],sys.argv[3],
                             'fig.png',label=None,opacity=0.6)

# infile,skip,xdata,ydata,xlabel,ylabel,outfile,label,
#                     err=None,lt='-',mark='o',opacity=1.0,
#                     plottype='nofit',**kwargs







#a
