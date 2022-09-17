#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:44:05 2019

@authors: mtelewa & hannesh
"""

""" Script to plot equilibrium and non-equilibrium profiles after spatial binning
or "chunked" output from LAMMPS ".profile" files as wells as "fix ave/time .txt" files.
"""

import numpy as np
import sys
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_from_txt as ptxt
import get_stats as avg
from progressbar import progressbar
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

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

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

class profile:
    """
    Returns a variable averaged over time in spatial bins
    or over spatial bins in each timestep.
    The average is either in the flow direction (x) or in the wall-normal direction (z)
    """

    def __init__(self,inputfile,columns,xvalue,yvalue):

        # ------------------- Collecting the data -------------------------------#
        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='BoxHeight':
                    total_box_height=float(line.split()[1])
                if line.split()[0]=='gapHeight':
                    gapheight=float(line.split()[1])
                if line.split()[0]=='AChunk':
                    ChunkArea=float(line.split()[1])
                if line.split()[0]=='VChunkX':
                    ChunkVolX=float(line.split()[1])
                if line.split()[0]=='VChunkZ':
                    ChunkVolZ=float(line.split()[1])
                if line.split()[0]=='vShear':
                    vwall=float(line.split()[1])

        # Unit conversion
        atm_to_mpa=101325*1e-6
        kcalpermolA3_to_mpa=1/0.00014393
        if 'mflux' or 'all' in sys.argv:
            gramA_per_fs_to_gram_per_m2_ns = 1e26/(ChunkArea*gapheight)
        A_per_fs_to_m_per_s = 1e5

        A = []
        with open(inputfile, 'r') as f:
            for line in f:
                if len(line.split())==columns and line.split()[0]!='#':
                    A.append(line.split())

        numChunks=int(A[-1][0])

        tSteps=int(len(A)/numChunks)
        tSteps_list=[i for i in range(tSteps)]
        tSteps_array=np.asarray(tSteps_list)

        # ------------------- Editing the data (Remove outliers)--------------------#

        # All measured values (e.g. all densities) array

        def remove_chunks(chunks):

            values_arr=[]
            values_arr=np.asarray([A[i][3] for i in range(len(A))]).astype(np.float)
            values_arr=np.reshape(values_arr,(tSteps,chunks))
            #print(values_arr[0])

            # the std deviation and mean of each chunk for all time steps
            std_dev=np.std(values_arr,axis=0)
            mean=np.mean(values_arr,axis=0)

            # Get the chunk ids to delete based on their mean (if the mean is zero)
            index=0
            idx=[]

            return mean,index,idx

        # A. Remove chunks with zero mean
        #--------------------------------

        mean=remove_chunks(numChunks)[0]
        index=remove_chunks(numChunks)[1]
        idx=remove_chunks(numChunks)[2]

        for i in range(len(mean)):
            if mean[i]==0:
                idx.append(index)
            index+=1

        ignored_chunk_ids=[x+1 for x in idx]
        #print(ignored_chunk_ids)

        A=np.asarray(A)
        A=A.astype(np.float)

        # An array of chunks to delete
        delete_chunks=[]
        for i in range(len(A)):
            if A[i][0] in ignored_chunk_ids:
                delete_chunks.append(A[i])

        delete_chunks=np.asarray(delete_chunks)
        # Check how many chunks are deleted in one time step
        #print(len(delete_chunks)/self.tSteps)

        # The new chunk count
        newChunks=int((len(A)-len(delete_chunks))/tSteps)

        # The difference between the 'all chunks' and 'chunks to delete' array
        def array_diff(B,C):
            cumdims = (np.maximum(B.max(),C.max())+1)**np.arange(C.shape[1])
            return B[~np.in1d(B.dot(cumdims),C.dot(cumdims))]

        # The old chunks count has to be different than the new chunks count
        if newChunks != numChunks:
            A=array_diff(A,delete_chunks)

        # B. Remove chunks not in the Inter Quartile Range after removing the
        # zero mean chunks
        #-------------------------------------------------------------------

        mean2=remove_chunks(newChunks)[0]
        index2=remove_chunks(newChunks)[1]
        idx2=remove_chunks(newChunks)[2]

        if inputfile == 'vx.profile':
            iqr_value=iqr(mean2, axis=0)
            q1=np.quantile(mean2,0.25)
            q3=np.quantile(mean2,0.75)
            delete_lower= q1-1.5*iqr_value      # 1.5 is used to avoid excluding useful data
            delete_upper= q1+1.5*iqr_value
            #print(delete_lower,delete_upper)

            for i in range(len(mean2)):
                if mean2[i]<delete_lower or mean2[i]>delete_upper:
                    index2=A[i][0]
                    idx2.append(index2)
                #remove_chunks()[1]+=1

            ignored_chunk_ids2=[x for x in idx2]
            # print(ignored_chunk_ids2)

            # An array of chunks to delete
            delete_chunks2=[]
            for i in range(len(A)):
                if A[i][0] in ignored_chunk_ids2:
                    delete_chunks2.append(A[i])

            delete_chunks2=np.asarray(delete_chunks2)
            newChunks=int((len(A)-len(delete_chunks2))/tSteps)
            # print(delete_chunks2)

            if len(delete_chunks2) != 0.0:
                A=array_diff(A,delete_chunks2)

        # The final array of values to use for computation and plotting
        A=np.reshape(A,(tSteps,newChunks,columns))

        # ------------------- The averaging -------------------------------#

        # Array to fill with temporal average for each spatial bin
        time_avg_of_chunk=np.zeros((newChunks,columns-2))
        time_avg_of_chunk[:,0]=A[0,:,1]               # the first index denotes height/length

        # block_size = 100
        # no_of_blocks = len(A)/block_size
        # blocks = np.reshape(A,(round(no_of_blocks),block_size))
        # block_mean = np.mean(blocks,axis=1)

        # print(A[1299,:,3])

        if inputfile != 'virialChunkZ.profile' and inputfile != 'virialChunkX.profile' and inputfile != 'momentaX.profile':
            for i in range(tSteps):
                time_avg_of_chunk[:,1]+=A[i,:,3]          # second entry denotes the variable
            time_avg_of_chunk[:,1]/=tSteps                # the variable averaged over time for each spatial bin

            # print(A[i,:,3])

        elif inputfile == 'momentaX.profile':
            time_avg_of_chunk=np.zeros((newChunks,columns-1))
            time_avg_of_chunk[:,0]=A[0,:,1]               # the first index denotes the length
            for i in range(tSteps):
                time_avg_of_chunk[:,1]+=A[i,:,2]          # Count
                time_avg_of_chunk[:,2]+=A[i,:,3]          # momenta
            ChunkCountFluid = time_avg_of_chunk[:,1]/tSteps
            momenta = time_avg_of_chunk[:,2]/tSteps                # momenta averaged over time for each spatial bin
        # For the virial profiles
        else:
            time_avg_of_chunk=np.zeros((newChunks,columns-1))
            time_avg_of_chunk[:,0]=A[0,:,1]               # the first index denotes the length
            for i in range(tSteps):
                time_avg_of_chunk[:,1]+=A[i,:,2]  #  Count
                time_avg_of_chunk[:,2]+=A[i,:,3]  #  W1
                time_avg_of_chunk[:,3]+=A[i,:,4]  #  W2
                time_avg_of_chunk[:,4]+=A[i,:,5]  #  W3

            ChunkCountBulk = time_avg_of_chunk[:,1]/tSteps
            W1 = time_avg_of_chunk[:,2]/tSteps
            W2 = time_avg_of_chunk[:,3]/tSteps
            W3 = time_avg_of_chunk[:,4]/tSteps

        # Array to fill with average of all chunks at each time step
        time_avg_of_all_chunks=np.zeros((tSteps,columns-2))

        time_avg_of_all_chunks[:,0]=tSteps_array[:]       # first entry: time
        for i in range(newChunks):
            time_avg_of_all_chunks[:,1]+=(A[:,i,3])       # second entry: state variable
        time_avg_of_all_chunks[:,1]/=newChunks            # Average over the bins for each time step

        # Name the output files according to the input
        base=os.path.basename(inputfile)
        filename=os.path.splitext(base)[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('%s' %xvalue)
        ax.set_ylabel('%s' %yvalue)

        # temporal average for each chunk:
        if 'chunks' in sys.argv:
            # Dimension array in nm
            dim_list = time_avg_of_chunk[:,0]/10
            dim = np.asarray(dim_list)

            # Variable array averaged over time
            var_over_time_list=time_avg_of_chunk[:,1]
            var_over_time=np.asarray(var_over_time_list)

            # For the density profiles in x-direction, correct for LAMMPS bin volume calculation
            if inputfile == 'denX.profile':
                scale = float(total_box_height)/float(gapheight)
                var_over_time=np.asarray(var_over_time_list)*scale
            # Get the flux from momenta
            elif inputfile == 'momentaX.profile':
                mflux_list= ChunkCountFluid*momenta*gramA_per_fs_to_gram_per_m2_ns
                var_over_time=np.asarray(mflux_list)
            elif inputfile == 'vx.profile':
                var_over_time=np.asarray(var_over_time_list)*A_per_fs_to_m_per_s
                # The shear rate is averaged only in the bulk (excludes the boundary points)
                slope_vx=np.polyfit(time_avg_of_chunk[4:-4,0],time_avg_of_chunk[4:-4,1],1)
                self.shear_rate=slope_vx[0]*1e15                         # in s^-1
                vx_at_lower_wall=time_avg_of_chunk[0][1]
                self.slip_length=(vx_at_lower_wall/slope_vx[0])*0.1      # in nm
                vx_at_upper_wall=time_avg_of_chunk[-1][1]
                self.slip_velocity=vwall-vx_at_upper_wall
            elif inputfile == 'virialChunkX.profile':
                #dim = dim[:-1]
                virialx_list=-ChunkCountBulk*(W1+W2+W3)*atm_to_mpa/(3*ChunkVolX)
                var_over_time=np.asarray(virialx_list)
                self.virialx = var_over_time
            elif inputfile == 'virialChunkZ.profile':
                virialz_list=-ChunkCountBulk*(W1+W2+W3)*atm_to_mpa/(3*ChunkVolZ)
                var_over_time=np.asarray(virialz_list)
                self.virialz = var_over_time
            elif inputfile == 'fzL.profile':
                dim = dim[1:-1]
                sigzzL_list=-time_avg_of_chunk[1:-1,1]*kcalpermolA3_to_mpa/(ChunkArea)
                var_over_time=np.asarray(sigzzL_list)
                self.sigzzL=var_over_time
            elif inputfile == 'fzU.profile':
                dim = dim[1:-1]
                sigzzU_list=-time_avg_of_chunk[1:-1,1]*kcalpermolA3_to_mpa/(ChunkArea)
                var_over_time=np.asarray(sigzzU_list)
                self.sigzzU=var_over_time
                self.length=time_avg_of_chunk[1:-1,0]/10

            np.savetxt(filename+'.txt', np.c_[dim,var_over_time],
                        delimiter="  ",header="%s           %s" %(xvalue,yvalue))
            ax.plot(dim,var_over_time,'-o')

            # if inputfile == 'virialChunkX.profile':
            #     ymin, ymax = ax.get_ylim()
            #     ax.set_ylim(ymin, ymax)
            #     plt.vlines(2.4, ymin, ymax, colors='k', linestyles='dashed', label='', data=None)

            plt.savefig(filename+'.png')

        # time series
        # # TODO: Expand for other variables
        elif 'time' in sys.argv:
            var_each_chunk_list=time_avg_of_all_chunks[:,1]
            var_each_chunk=np.asarray(var_each_chunk_list)

            # Get the flux from momenta
            if inputfile == 'momentaX.profile':
                #print('lol')
                var_each_chunk=np.asarray(var_each_chunk_list)*gramA_per_fs_to_gram_per_m2_ns

            np.savetxt(filename+'-time.txt', np.c_[tSteps_array,var_each_chunk],
                        delimiter="  ",header="%s           %s" %(xvalue,yvalue))
            ax.plot(tSteps_array,var_each_chunk,'-')
            plt.savefig(filename+'-time.png')

        # for i in range(self.tSteps):
        #     progressbar(i+1,self.tSteps)

        #xnew = np.linspace(densityZ_over_time[0], densityZ_over_time[-1], num=100, endpoint=True)
        #f = interp1d(densityZ_over_time, height, kind='cubic')
        #ax.plot(xnew,f(xnew), '-')

        #ax.margins(x=None, y=1.0, tight=True)

        #tick_labels=myticks(var_over_time,100,-1)
        #ax.yaxis.set_ticks(tick_labels)

        #slope, intercept, r_value, p_value, std_err = stats.linregress(height, vx)
        #ax.plot(height[4:-4], intercept + slope*height[4:-4], 'r') #, label='fitted line')


def main():

    if 'denz' in sys.argv:
        plot = profile('denZ.profile',4,'Height $(nm)$','Density $(g/cm^3)$')#.rhoZ_height(fig)
    if 'denz-time' in sys.argv:
        plot = profile('denZ.profile',4,'Timestep','Density $(g/cm^3)$')
    if 'denx' in sys.argv:
        plot = profile('denX.profile',4,'Length $(nm)$','Density $(g/cm^3)$')
    if 'denx-time' in sys.argv:
        plot = profile('denX.profile',4,'Time step','Density $(g/cm^3)$')
    if 'mflux' in sys.argv:
        plot = profile('momentaX.profile',4,'Length $(nm)$','Mass flux $j_{x} \;(g/m^2.ns)$')
    if 'mflux-time' in sys.argv:
        ptxt.plot_from_txt('flux-stable-time.txt',2,0,1,'Time step','Mass flux $j_{x} \;(g/m^2.ns)$',
                'flux-time.png',mark=None)
    if 'vx' in sys.argv:
        plot = profile('vx.profile',4,'Height $(nm)$','Vx $(m/s)$')
        print("Shear rate %.3e s^-1 \nSlip length %.3f nm \nSlip velocity %.3f m/s"
                %(plot.shear_rate,plot.slip_length,plot.slip_velocity))
    if 'vx-time' in sys.argv:
        plot = profile('vx.profile',4,'Time step','Vx $(m/s)$')
    if 'tempx' in sys.argv:
        plot = profile('tempX.profile',4,'Length $(nm)$','Temperature $(K)$')
    if 'tempz' in sys.argv:
        plot = profile('tempZ.profile',4,'Height $(nm)$','Temperature $(K)$')
    if 'temp-time' in sys.argv:
        plot = profile('tempZ.profile',4,'Time step','Temperature $(K)$')
    if 'virialx' in sys.argv:
        plot = profile('virialChunkX.profile',6,'Length $(nm)$','Pressure $(MPa)$')
    if 'virialz' in sys.argv:
        plot = profile('virialChunkZ.profile',6,'Height $(nm)$','Pressure $(MPa)$')
    if 'virial-time' in sys.argv:
        ptxt.plot_from_txt('virial.txt',2,0,1,'Time step','Pressure $(MPa)$','virial_pressure.png')
    if 'sigzzL' in sys.argv:
        plot = profile('fzL.profile',4,'Length $(nm)$','Pressure $(MPa)$')
    if 'sigzzU' in sys.argv:
        plot = profile('fzU.profile',4,'Length $(nm)$','Pressure $(MPa)$')
    if 'sigzz' in sys.argv:
        sigzz_upper = profile('fzU.profile',4,'Length $(nm)$','Pressure $(MPa)$').sigzzU
        sigzz_lower = profile('fzL.profile',4,'Length $(nm)$','Pressure $(MPa)$').sigzzL
        length = profile('fzU.profile',4,'Length $(nm)$','Pressure $(MPa)$').length
        sigzz = 0.5*(sigzz_lower-sigzz_upper)
        np.savetxt("sigzz.txt", np.c_[length,sigzz],delimiter="  ",header="Length(nm)       Sigmazz(MPa)")
        os.system("plot_from_txt.py sigzz.txt 1 0 1 sigzz.png --xlabel 'Length $(nm)$' \
                        --ylabel 'Pressure $(MPa)$' --label 'Wall $\sigma_{zz}$' ")
    if 'sigzz-time' in sys.argv:
        ptxt.plot_from_txt('stress.txt',2,0,3,1,'nofit','Wall $\sigma_{zz}$','Time step','Pressure $(MPa)$','sigzz-time.png')
    if 'sigzzU-time' in sys.argv:
        ptxt.plot_from_txt('stressU.txt',2,0,3,1,'nofit',' Upper Wall $\sigma_{zz}$','Time step','Pressure $(MPa)$','sigzzU-time.png')
    if 'sigzzL-time' in sys.argv:
        ptxt.plot_from_txt('stressL.txt',2,0,3,1,'nofit',' Lower Wall $\sigma_{zz}$','Time step','Pressure $(MPa)$','sigzzL-time.png')
    if 'sigxz-time' in sys.argv:
        ptxt.plot_from_txt('stress.txt',2,0,1,1,'nofit','Wall $\sigma_{xz}$','Time step','Pressure $(MPa)$','sigxz.png')
    if 'viscosity' in sys.argv:
        shear_rate = profile('vx.profile',4,'Height $(nm)$','Vx $(m/s)$').shear_rate
        tauxz_lower = avg.columnStats('stressL.txt',1002,1)
        tauxz_upper = avg.columnStats('stressU.txt',1002,1)
        tauxz = 0.5*(tauxz_lower-tauxz_upper) #MPa
        print(tauxz)
        mu = tauxz/(shear_rate*1e-15)
        print(shear_rate*1e-15)
        print("Dynamic viscosity: %4.3f mPas" % (mu*1e-6))
    if 'all' in sys.argv:
        # plot1 = profile('denZ.profile',4,'Height $(nm)$','Density $(g/cm^3)$')
        plot2 = profile('denX.profile',4,'Length $(nm)$','Density $(g/cm^3)$')
        plot3 = profile('momentaX.profile',4,'Length $(nm)$','Mass flux $j_{x} \;(g/m^2.ns)$')
        plot4 = profile('vx.profile',4,'Height $(nm)$','Vx $(m/s)$')
        plot5 = profile('virialChunkX.profile',6,'Length $(nm)$','Pressure $(MPa)$')

if __name__ == "__main__":
   main()
