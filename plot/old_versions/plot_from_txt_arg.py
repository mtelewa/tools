#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import argparse
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
    cycler(
        color=[
            u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )

# mpl.rc('axes', prop_cycle=default_cycler)

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


def main(args):
    parser = argparse.ArgumentParser(
    description='Plots data from a text file with matplotlib')

    # Arguments ----------------
    parser.add_argument('infile', metavar='input_file', action='store',
                    help='Input file(s) name(s)')
    parser.add_argument('skip', metavar='skip_lines', action='store', type=int,
                    help='Nuber of lines to skip')
    parser.add_argument('xdata', metavar='x-axis_data', action='store', type=int,
                    help='X-axis data column index')
    parser.add_argument('ydata', metavar='y-axis_data', action='store', type=int,
                    help='Y-axis data column index')
    parser.add_argument('outfile', metavar='output_file', action='store',
                    help='Output file name')
    parser.add_argument('--err', metavar='error_data', action='store',
                    help='Error column data index, default: None')
    parser.add_argument('--lt', metavar='line_type', action='store',
                    help='Matplotlib line type, default: "-"')
    parser.add_argument('--mark', metavar='marker_type', action='store',
                    help='Matplotlib marker type, default: "o"')
    parser.add_argument('--opacity', metavar='opacity', action='store', type=int,
                    help='Transperency/Opacity of the lines/markers, default: "1"')
    parser.add_argument('--color', metavar='color', action='store',
                    help='Color of line and marker')
    parser.add_argument('--plottype', metavar='plot_type', action='store',
                    help='Plot with/out fit and/or error bars, default: No fit, no errorr bars')
    parser.add_argument('--label', metavar='label', action='store',
                    help='Plot label for legend')
    parser.add_argument('--xlabel', metavar='x_label', action='store',
                    help='x-axis label, default: "x"')
    parser.add_argument('--ylabel', metavar='y_label', action='store',
                    help='y-axis label, default: "y"')


    args = parser.parse_args(args)
    print(args)

    data = np.loadtxt(args.infile,skiprows=args.skip,dtype=float)
    x_data = data[:,int(args.xdata)]
    y_data = data[:,int(args.ydata)]

    if args.err:
        err = data[:,int(args.err)]
    else:
        pass

    linestyle = args.lt if args.lt else '-'
    markerstyle = args.mark if args.mark else 'o'
    alpha = args.opacity if args.opacity else 1.0
    label = args.label if args.label else '%s' %(args.infile)

    if args.color:
        color = args.color
    else:
        color = '#1f77b4'

    scale_x = 1
    scale_y = 1

    def func(x,a,b,c):
        return a*x**2+b*x+c

    def func2(x,a,b):
        return a*x+b

    popt, pcov = curve_fit(func, x_data, y_data)
    popt2, pcov2 = curve_fit(func2, x_data, y_data)

    #Error bar with Quadratic fitting
    if args.plottype=='errquad':
        plt.plot(x_data,func(x_data,*popt),ls=linestyle,marker=markerstyle,
                   alpha=alpha,label=None)
        plt.errorbar(x_data,y_data,yerr=err,ls=linestyle,fmt=markerstyle,capsize=3,
                   alpha=alpha,label=label)

    #No errorbar With Quadratic fitting
    if args.plottype=='quad':
        plt.plot(x_data,func(x_data,*popt),ls=linestyle,marker=None,
                   alpha=alpha,label=None)
        plt.plot(x_data,y_data,marker=markerstyle,
                   alpha=alpha,label=label)

    #No errorbar Without linear fitting
    if args.plottype=='linear':
        plt.plot(x_data,func2(x_data,*popt2),ls=linestyle,marker=None,
                   alpha=alpha,label=None)
        plt.plot(x_data,y_data,ls=None,marker=markerstyle,
                   alpha=alpha,label=label)

    #No errorbar Without fitting
    if args.plottype=='nofit':
        plt.plot(x_data*scale_x,y_data*scale_y,ls=linestyle,marker=markerstyle,
                    alpha=alpha,label=label,color=color)

    else:
        plt.plot(x_data*scale_x,y_data*scale_y,ls=linestyle,marker=markerstyle,
                    alpha=alpha,label=label,color=color)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ymin,ymax = ax.get_ylim()
    # ax.set_ylim(ymin, ymax)
    # plt.vlines(2.4, ymin, ymax, colors='k', linestyles='dashed', label='', data=None)

    #plt.errorbar(xdata*scale_x,ydata*scale_y,yerr=err)
    #plt.text(xdata_i[-1], ydata_i[-1], input("Plot label: "), withdash=True)

    plt.legend()

    if args.xlabel or args.ylabel:
        plt.xlabel(args.xlabel)
        plt.ylabel(args.ylabel)
    else:
        plt.xlabel('x')
        plt.ylabel('y')

    plt.savefig(args.outfile)

if __name__ == "__main__":
   main(sys.argv[1:])
