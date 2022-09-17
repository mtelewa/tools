    popt, pcov = curve_fit(quadratic, xdata, ydata)
    popt2, pcov2 = curve_fit(linear, xdata, ydata)

    if plottype=='log':
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(xdata*scale_x,ydata*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)

    if twin=='yes':
        data2 = np.loadtxt(infile2,skiprows=skip,dtype=float)
        ydata2 = data2[:,int(ydata2)]
        ax2 = ax.twinx()
        ax2.plot(xdata*scale_x,ydata2*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label,color=u'#ff7f0e')
        ax2.set_ylabel(r'$\rho \, (g/cm^3)$')
        # color_cycle = ax._get_lines.prop_cycler
        # for label in ax2.get_yticklabels():
        #     label.set_color()

    if inset=='yes':
        plt.tight_layout()
        data2 = np.loadtxt(infile2,skiprows=skip,dtype=float)
        xdata2 = data2[:,int(xdata2)]
        ydata2 = data2[:,int(ydata2)]
        inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height
        inset_ax.plot(xdata2*scale_x,ydata2*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)
        # set axis tick locations
        # inset_ax.set_yticks([0, 0.005, 0.01])
        # inset_ax.set_xticks([-0.1,0,.1]);

    if format=='power':
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.yaxis.set_major_formatter(formatter)


    #Erorbar Without fitting
    if plottype=='errnofit':
        plt.axvline(x=0.125, color='r', linestyle='dashed', label=' pump inlet')
        plt.axvline(x=2.38, color='b', linestyle='dashed', label=' pump outlet')
        err = data[:,int(err)]
        ax.plot(xdata*scale_x,ydata*scale_y,ls=lt,marker=mark,
                    alpha=opacity,label=label)
        ax.errorbar(xdata,ydata,yerr=err,ls=lt,fmt=mark,capsize=3,
                   alpha=opacity,label=label)

    #Error bar with Quadratic fitting
    if plottype=='errquad':
        err = data[:, int(err)]
        ax.plot(xdata, quadratic(xdata, *popt),ls = lt,marker = mark,
                   alpha = opacity, label=None)
        ax.errorbar(xdata, ydata, yerr=err, ls=lt, fmt=markerstyle, capsize=3,
                   alpha=opacity, label=label)

    #No errorbar With Quadratic fitting
    if plottype=='quad':
        ax.plot(xdata,quadratic(xdata,*popt),ls=lt,marker=None,
                   alpha=opacity,label=None)
        ax.plot(xdata,ydata,marker=mark,
                   alpha=opacity,label=label)

    #No errorbar Without linear fitting
    if plottype=='linear':
        ax.plot(xdata,linear(xdata,*popt2),ls=lt,marker=None,
                   alpha=opacity,label=None)
        ax.plot(xdata,ydata,ls=None,marker=mark,
                   alpha=opacity,label=label)
