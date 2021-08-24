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

# mpl.rcParams.update({'font.size': 18})

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
# mpl.rc('axes', prop_cycle=default_cycler)
# mpl.rcParams.update({'text.usetex': True})
# mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})

# mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# plt.style.use('default')
# print(plt.style.available)

def myticks(arr,interval,val):
    range_val = max(arr)-min(arr)
    amin = min(arr)-interval*range_val
    amax = max(arr)+interval*range_val
    #print(amin,amax)
    amin=truncate(amin,val)
    amax=truncate(amax,val)

    tick_labels = np.arange(amin,amax,interval)
    #tick_labels = tick_labels.astype(int)
    #print(amin,amax,tick_labels)
    return tick_labels


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

    linestyle = lt
    markerstyle = mark
    alpha = opacity

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
        plt.plot(x_data,func(x_data,*popt),ls=linestyle,marker=markerstyle,
                   alpha=alpha,label=None)
        plt.errorbar(x_data,y_data,yerr=err,ls=linestyle,fmt=markerstyle,capsize=3,
                   alpha=alpha,label=label)

    #No errorbar With Quadratic fitting
    if plottype=='quad':
        plt.plot(x_data,func(x_data,*popt),ls=linestyle,marker=None,
                   alpha=alpha,label=None)
        plt.plot(x_data,y_data,marker=markerstyle,
                   alpha=alpha,label=label)

    #No errorbar Without linear fitting
    if plottype=='linear':
        plt.plot(x_data,func2(x_data,*popt2),ls=linestyle,marker=None,
                   alpha=alpha,label=None)
        plt.plot(x_data,y_data,ls=None,marker=markerstyle,
                   alpha=alpha,label=label)

    #No errorbar Without fitting
    if plottype=='nofit':
        # plt.axhline(y=0.08663375, color='r', linestyle='dashed', label=' imposed')
        # plt.axvline(x=0.2*35.9, color='g', linestyle='dashed', label=' measured')
        # plt.xlim(0, 5e6)     # set the ylim to bottom, top
        plt.plot(x_data*scale_x,y_data*scale_y,ls=linestyle,marker=markerstyle,
                    alpha=alpha,label=label)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(outfile, dpi=100)



if __name__ == "__main__":
    if 'vx' in sys.argv:
        plot_from_txt('vx.txt',1,0,1,labels[0],labels[5],
                            'vx.png','6 $\AA$',opacity=0.6)
    if 'denz' in sys.argv:
        plot_from_txt('denZ.txt',1,0,1,labels[0],labels[2],
                            'denz.png','6 $\AA$',opacity=0.6)
    if 'virialx' in sys.argv:
        plot_from_txt('virialChunkX.txt',1,0,1,labels[0],labels[2],
                            'virialx.png','6 $\AA$',opacity=0.6)
    if 'dacf' in sys.argv:
        plot_from_txt('dacf.txt',1,0,1,labels[3],'DACF',
                            'dacf.png','12 $\AA$',mark=None,opacity=1.0)
    else:
        plot_from_txt(sys.argv[1],2,0,1,sys.argv[2],sys.argv[3],
                             'fig.png',label=None,opacity=0.6)

# ymin, ymax = ax.get_ylim()
# ax.set_ylim(ymin, ymax)
# plt.vlines(2.4, ymin, ymax, colors='k', linestyles='dashed', label='', data=None)

#xnew = np.linspace(densityZ_over_time[0], densityZ_over_time[-1], num=100, endpoint=True)
#f = interp1d(densityZ_over_time, height, kind='cubic')

#ax.plot(xnew,f(xnew), '-')
#ax.margins(x=None, y=1.0, tight=True)

#tick_labels=myticks(var_over_time,100,-1)
#ax.yaxis.set_ticks(tick_labels)

#slope, intercept, r_value, p_value, std_err = stats.linregress(height, vx)
#ax.plot(height[4:-4], intercept + slope*height[4:-4], 'r') #, label='fitted line')


# ax = fig.add_subplot(111)
# ymin,ymax = ax.get_ylim()
# ax.set_ylim(ymin, ymax)
# plt.vlines(2.4, ymin, ymax, colors='k', linestyles='dashed', label='', data=None)

#plt.errorbar(xdata*scale_x,ydata*scale_y,yerr=err)
#plt.text(xdata_i[-1], ydata_i[-1], input("Plot label: "), withdash=True)
