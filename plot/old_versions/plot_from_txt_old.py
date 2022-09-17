#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.stats import iqr
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

mpl.rcParams.update({'font.size': 18})
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

## TODO: KEEP THE COLOR OF THE LINE AND DOT THE SAME!!!
default_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )

mpl.rc('axes', prop_cycle=default_cycler)

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
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

#------------------------- Plot from text files --------------------------------#
def plot_from_txt(inputfile,skip,xdata,ydata,alpha,plottype,label,xlabel,ylabel,figname):

    #for i in range(int(nplots)):
    data = np.loadtxt(inputfile,skiprows=skip,dtype=float)
    x_data = data[:,int(xdata)]    #data[:,int(input("x-axis data: "))]
    y_data = data[:,int(ydata)]     #data[:,int(input("y-axis data: "))]
    #err =  data[:,int(input("Error data: "))]

    scale_x = 1     #float(input("scale x-axis: "))
    scale_y = 1      #float(input("scale y-axis: "))

    linestyle_val = '-'        #input("line style: " )
    markerstyle= None
    alpha_val = alpha
    #color=input("color: " )

    def func(x,a,b,c):
        return a*x**2+b*x+c

    def func2(x,a,b):
        return a*x+b

    popt, pcov = curve_fit(func, x_data, y_data)
    popt2, pcov2 = curve_fit(func2, x_data, y_data)

    #Error bar with Quadratic fitting
    if plottype=='errquad':
        plt.plot(x_data,func(x_data,*popt),ls=linestyle_val,marker=markerstyle,
                   alpha=alpha_val,label=None)
        plt.errorbar(x_data,y_data,yerr=err,ls=linestyle,fmt=markerstyle,capsize=3,
                   alpha=alpha_val,label=label)

    #No errorbar With Quadratic fitting
    if plottype=='quad':
        plt.plot(x_data,func(x_data,*popt),ls=linestyle_val,marker=None,
                   alpha=alpha_val,label=None)
        plt.plot(x_data,y_data,marker=markerstyle,
                   alpha=alpha_val,label=label)

    #No errorbar Without linear fitting
    if plottype=='linear':
        plt.plot(x_data,func2(x_data,*popt2),ls=linestyle_val,marker=None,
                   alpha=alpha_val,label=None)
        plt.plot(x_data,y_data,ls=None,marker=markerstyle,
                   alpha=alpha_val,label=label)

    #No errorbar Without fitting
    if plottype=='nofit':
        plt.plot(x_data*scale_x,y_data*scale_y,ls=linestyle_val,marker=markerstyle,
                    alpha=alpha,label=label)

        #plt.errorbar(xdata*scale_x,ydata*scale_y,yerr=err)
        #plt.text(xdata_i[-1], ydata_i[-1], input("Plot label: "), withdash=True)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figname)

if __name__ == "__main__":
   main()
