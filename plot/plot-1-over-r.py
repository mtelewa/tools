import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

default_cycler = (
    cycler(color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
            u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )


mpl.rc('font', size=BIGGER_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=50)     # fontsize of the axes title
mpl.rc('axes', labelsize=25)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
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



def f(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(1/x)

fx_name = 'Linear Elasticity'

ax = plt.gca()
ax.xaxis.set_label_coords(1.0, -0.025)


plt.yticks(rotation=90)
plt.xticks([])
plt.yticks([])

x=np.linspace(-10,10,101)
y=f(x)


ax.fill_betweenx(y,-1,1,alpha=0.3,color='tab:red',label='Core region')
plt.plot(x, y, label=fx_name)
plt.axvline(x=0, color='k', linestyle='-')
plt.xlabel('r')

ax.set_xticks([1])
ax.set_xticklabels(['$r_0$'])

plt.ylabel(r'$\sigma(r)$')
plt.legend()
#plt.show()
plt.savefig('core.png')


