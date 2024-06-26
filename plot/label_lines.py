import numpy as np
from math import atan2

def label_line(line,x,yoffset=None,label=None,rotation=None,**kwargs):
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return
    #Find corresponding y coordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break
    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1]) + yoffset

    if not label:
        label = line.get_label()
    if rotation is not None:
        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((rotation,)),pt)[0]
    else:
        #Compute the slope
        dx = xdata[ip+10] - xdata[ip-10]
        dy = ydata[ip+10] - ydata[ip-10]
        ang = np.degrees(atan2(dy,dx))
        print(ang)
        #Transform to screen coordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]
        # print(trans_angle)

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()
    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'
    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'
    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()
    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5
    ax.text(x,y,label,rotation=trans_angle,rotation_mode='anchor', transform_rotates_text=True,
        bbox=dict(facecolor='none', edgecolor='none'),**kwargs)

def label_lines(lines,xvals=None,rotations=None,**kwargs):
    ax = lines[0].axes
    labLines = []
    labels = []
    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)
    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    if rotations is None:
        rotations = [None]*len(labLines)
    for line,x,label,rotation in zip(labLines,xvals,labels,rotations):
        label_line(line,x,yoffset,label,rotation,**kwargs)
