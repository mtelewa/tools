#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:44:05 2019

@authors: mtelewa & hannesh
"""

# Script to compute equilibrium and non-equilibrium profiles after spatial binning
# or "chunked" output from LAMMPS ".profile" files as wells as "fix ave/time .txt" files.
# Dynamic viscosity for Couette flow is calculated from the velocity profile
# and average thermodynamic quantities and stress tensor

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from scipy.stats import iqr
#from cycler import cycler
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
# default_cycler = (
#     cycler(
#         color=[
#             u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
#             u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] ) )
#
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

def progressbarstr(x, maxx, len=60):
    xx = x/maxx
    if xx != 1:
      return '|'+int(xx*len)*'#'+((len-int(xx*len))*'-')+'| {:>3}% ' \
             '({}/{})'.format(int(xx*100), x, maxx)
    else:
      return '|'+int(xx*len)*'#'+((len-int(xx*len))*'-')+'| {:>3}% ' \
             '({}/{})\n'.format(int(xx*100), x, maxx)

def progressbar(x, maxx, len=60):
    sys.stdout.write('{}\r'.format(progressbarstr(x, maxx, len=len)))
    sys.stdout.flush()


#------------------------- Get the column averages --------------------------------#
def columnStats(filename,skip,column):

    load_file = np.loadtxt(filename,skiprows=skip,dtype=float)
    num_columns = load_file.shape[1]
    num_rows = load_file.shape[0]
    #num_columns_list=list(range(1,num_columns))

    # Every 100,000 time steps
    block_size = 100
    whole_data = [load_file[i][column] for i in range(num_rows)]
    no_of_blocks = len(whole_data)/block_size
    #print(groups_of_data,len(values_arr))
    # Row is the values in the sample, column is the timesteps
    blocks = np.reshape(whole_data,(round(no_of_blocks),block_size))
    #print(values_arr[0])
    block_mean = np.mean(blocks,axis=1)
    #print(block_mean[0])
    mu = np.mean(block_mean)
    #print(whole_mean)
    #whole_mean = np.mean(load_file[:,column])
    #block_std_dev=np.std(blocks,axis=1)
    #print(block_std_dev)
    #sigma=np.mean(block_std_dev/np.sqrt(no_of_blocks))
    sigma=np.std(block_mean)
    #sigma=np.std(blocks,axis=1)/np.sqrt(no_of_blocks)

    #print(sigma)

    confidence_interval = 0.95      # 95% confidence interval
    alpha = 1.-confidence_interval
    cutoff = (alpha/2.)+confidence_interval       # 97.5th percentile

    z_score=norm.ppf(cutoff)
    #print(z_score)
    margin_error=z_score*sigma/np.sqrt(len(blocks))

    confidence_intervalU = mu+margin_error
    confidence_intervalL = mu-margin_error

    print("At a confidence interval of %g%%: \n\
          Margin Error: %.3f \n\
          Upper limit: %.3f \n\
          Lower limit: %.3f" %(confidence_interval*100.,margin_error,confidence_intervalL,
                                confidence_intervalU))

    # print("Column average taken every %g blocks: %.2f \nColumn average of the whole column: %.2f \n\
    # Upper confidence level: %.2f \nLower confidence level: %.2f " %(avg_every,mean_of_all_blocks,whole_mean,confidence_intervalU,confidence_intervalL))

    return mu


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


#------------------------- Plot from text files --------------------------------#
def plot_from_txt():

    plots = int(input("No. of plots on the same figure: "))
    #fig = plt.figure(figsize=(10.,8.))
    for i in range(plots):
        data = np.loadtxt(input("inputfile: "),skiprows=1,dtype=float)
        x_data = data[:,0]    #data[:,int(input("x-axis data: "))]
        y_data =data[:,1]     #data[:,int(input("y-axis data: "))]

        #err =  data[:,int(input("Error data: "))]

        scale_x = 1     #float(input("scale x-axis: "))
        scale_y = 1      #float(input("scale y-axis: "))

        linestyle_val = '-'        #input("line style: " )
        markerstyle= input("marker style: " )
        alpha_val = 1
        #color=input("color: " )

        # err = []
        # for i in range(len(xdata)):
        #     ele = float(input("err %g:  " %i))
        #     err.append(ele) # adding the element

        def func(x,a,b,c):
            return a*x**2+b*x+c

        def func2(x,a,b):
            return a*x+b

        popt, pcov = curve_fit(func, x_data, y_data)
        popt2, pcov2 = curve_fit(func2, x_data, y_data)
        #Error bar with Quadratic fitting
        #plt.plot(x_data,func(x_data,*popt),ls=linestyle,marker=markerstyle,
        #            alpha=alpha_val,label=None)
        #plt.errorbar(x_data,y_data,yerr=err,ls=linestyle,fmt=markerstyle,capsize=3,
        #            alpha=alpha_val,label=input("label: ")

        #No errorbar With Quadratic fitting
        plt.plot(x_data,func(x_data,*popt),ls=linestyle_val,marker=None,
                   alpha=alpha_val,label=None)
        plt.plot(x_data,y_data,marker=markerstyle,
                   alpha=alpha_val,label=input("label: "))

        #No errorbar Without linear fitting
        # plt.plot(x_data,func2(x_data,*popt2),ls=linestyle,marker=None,
        #            alpha=alpha_val,label=None)
        # plt.plot(x_data,y_data,ls=None,marker=markerstyle,
        #            alpha=alpha_val,label=input("label: "))

        #No errorbar Without fitting
        # plt.plot(x_data*scale_x,y_data*scale_y,ls=linestyle,marker=markerstyle,
        #             alpha=alpha_val,label=input("label: "))

        #plt.errorbar(xdata*scale_x,ydata*scale_y,yerr=err)
        plt.legend()
        #plt.text(xdata_i[-1], ydata_i[-1], input("Plot label: "), withdash=True)

    plt.xlabel(input("x-label: "))
    plt.ylabel(input("y-label: "))
    plt.savefig(input("Fig. name: "))

class profile:
    """
    Returns the density averaged over the bin and over time in the flow direction (x)
    or in the wall-normal direction (z)
    """

    def __init__(self,inputfile,columns):
        self.A = []

        with open(inputfile, 'r') as f:
            for line in f:
                if len(line.split())==columns and line.split()[0]!='#':
                    self.A.append(line.split())

        numChunks=int(self.A[-1][0])

        self.tSteps=int(len(self.A)/numChunks)
        tSteps_list=[i for i in range(self.tSteps)]
        self.tSteps_array=np.asarray(tSteps_list)

        # All measured values (e.g. all densities) array
        values_arr = []
        values_arr=[self.A[i][3] for i in range(len(self.A))]
        values_arr=np.asarray(values_arr)
        values_arr=values_arr.astype(np.float)
        values_arr=np.reshape(values_arr,(self.tSteps,numChunks))

        # the std deviation and mean of each chunk for all time steps
        std_dev=np.std(values_arr,axis=0)
        mean=np.mean(values_arr,axis=0)

        #print(values_arr[0])

        # Get the chunk ids to delete based on their mean (if the mean is zero)
        index=0
        idx=[]

        # Remove chunks with mean value of zero
        # CAREFUL WITH PROFILES WITH MORE THAN ONE OUTPUT COLUMN!!!

        #if inputfile != 'sigmazz.profile':
        for i in range(len(mean)):
            if mean[i]==0:
                idx.append(index)
            index+=1

        ## BUG: Count in chunks should all be the same
        # else:
        #     countAtomsInchunk=int(self.A[2][2])
        #     #print(countAtomsInchunk)
        #     for i in range(numChunks):
        #         if int(self.A[i][2]) != countAtomsInchunk:          ## BUG: which atom count to ignore?
        #             idx.append(index)
        #         index+=1

        ignored_chunk_ids=[x+1 for x in idx]

        # Check the deleted chunk ids
        #print(ignored_chunk_ids)

        #b=np.delete(values_arr,idx,axis=1)

        self.A=np.asarray(self.A)
        self.A=self.A.astype(np.float)

        # An array of chunks to delete
        delete_chunks=[]
        for i in range(len(self.A)):
            if self.A[i][0] in ignored_chunk_ids:
                delete_chunks.append(self.A[i])

        delete_chunks=np.asarray(delete_chunks)

        # Check how many chunks are deleted in one time step
        #print(len(delete_chunks)/self.tSteps)

        # The new chunk count
        self.newChunks=int((len(self.A)-len(delete_chunks))/self.tSteps)

        # The difference between the 'all chunks' and 'chunks to delete' array
        def array_diff(B,C):
            cumdims = (np.maximum(B.max(),C.max())+1)**np.arange(C.shape[1])
            return B[~np.in1d(B.dot(cumdims),C.dot(cumdims))]

        # The old chunks count has to be different than the new chunks count
        if self.newChunks != numChunks:
            self.A=array_diff(self.A,delete_chunks)


        # For some profiles, we need to delete chunks with Inter Quartile Range
        # (IQR) values exceeding upper and lower quartiles

        if inputfile == 'vx.profile' or inputfile == 'temp.profile':
            values_arr2= []
            values_arr2=[self.A[i][3] for i in range(len(self.A))]
            values_arr2=np.asarray(values_arr2)
            values_arr2=values_arr2.astype(np.float)
            values_arr2=np.reshape(values_arr2,(self.tSteps,self.newChunks))

            # the std deviation and mean of each chunk for all time steps
            std_dev2=np.std(values_arr2,axis=0)
            mean2=np.mean(values_arr2,axis=0)

            #print(len(mean2))

            index2=0
            idx2=[]

            iqr_value=iqr(mean2, axis=0)
            q1=np.quantile(mean2,0.25)
            q3=np.quantile(mean2,0.75)

            delete_lower= q1-2.5*iqr_value      # 2.5 is used to avoid excluding useful data
            delete_upper= q1+2.5*iqr_value
            # Check the upper and lower limits
            #print(delete_lower,delete_upper)

            for i in range(len(mean2)):
                if mean2[i]<delete_lower or mean2[i]>delete_upper:
                    index2=self.A[i][0]
                    idx2.append(index2)
                # index2+=1

            ignored_chunk_ids2=[x for x in idx2]

        # Check the deleted chunk ids
        # print(idx2)
        # An array of chunks to delete
            delete_chunks2=[]
            for i in range(len(self.A)):
                if self.A[i][0] in ignored_chunk_ids2:
                    delete_chunks2.append(self.A[i])

            delete_chunks2=np.asarray(delete_chunks2)
        # Check how many chunks are deleted in one time step
        #print(len(delete_chunks)/self.tSteps)
        # The new chunk count
            self.newChunks=int((len(self.A)-len(delete_chunks2))/self.tSteps)
            # print(delete_chunks2)
        # The difference between the 'all chunks' and 'chunks to delete' array
            def array_diff(B,C):
                cumdims = (np.maximum(B.max(),C.max())+1)**np.arange(C.shape[1])
                return B[~np.in1d(B.dot(cumdims),C.dot(cumdims))]
        # The old chunks count has to be different than the new chunks count
            if len(delete_chunks2) != 0.0:
                self.A=array_diff(self.A,delete_chunks2)

        # The final array of values to use for computation and plotting
        self.A=np.reshape(self.A,(self.tSteps,self.newChunks,columns))

        # Array to fill with temporal average for each spatial bin
        self.time_avg_of_chunk=np.zeros((self.newChunks,2))

        self.time_avg_of_chunk[:,0]=self.A[0,:,1]               # the first index denotes height
        for i in range(self.tSteps):
            self.time_avg_of_chunk[:,1]+=self.A[i,:,3]          # second index denotes variable

        self.time_avg_of_chunk[:,1]/=self.tSteps                # the variable averaged over time for each spatial bin

        # Array to fill with average of all chunks at each time step
        self.time_avg_of_all_chunks=np.zeros((self.tSteps,2))

        self.time_avg_of_all_chunks[:,0]=self.tSteps_array[:]       # first entry: time
        for i in range(self.newChunks):
            self.time_avg_of_all_chunks[:,1]+=(self.A[:,i,3])       # second entry: state variable

        self.time_avg_of_all_chunks[:,1]/=self.newChunks            # Average over the bins for each time step

        # for i in range(self.tSteps):
        #     progressbar(i+1,self.tSteps)

#------------------------- Density profiles --------------------------------#

    def rhoZ_height(self,fig):
        """ Samples the density in the normal-to-wall direction with the height.
            The LAMMPS output file is 'denZ.profile'"""

        height_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        height=np.asarray(height_list)

        densityZ_over_time_list=self.time_avg_of_chunk[:,1]*1e3                        # In kg/m3
        densityZ_over_time=np.asarray(densityZ_over_time_list)

        np.savetxt("denZ-height.txt", np.c_[densityZ_over_time,height],delimiter="  ",header="Density(kg/m3)    Height(nm)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Density $(kg/m^3)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(densityZ_over_time,height,'-o')

        #xnew = np.linspace(densityZ_over_time[0], densityZ_over_time[-1], num=100, endpoint=True)
        #f = interp1d(densityZ_over_time, height, kind='cubic')
        #ax.plot(xnew,f(xnew), '-')

    def rhoZ_time(self,fig):
        """ Samples the density in the normal-to-wall direction with time.
            The LAMMPS output file is 'denZ.profile'"""

        densityZ_each_chunk_list=self.time_avg_of_all_chunks[:,1]*1e3                        # In kg/m3
        densityZ_each_chunk=np.asarray(densityZ_each_chunk_list)

        np.savetxt("denZ-time.txt", np.c_[self.tSteps_array,densityZ_each_chunk],delimiter="  ",header="Timestep    Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Density $(kg/m^3)$')
        ax.plot(self.tSteps_array,densityZ_each_chunk,'-')

    def rhoX_length(self,fig):
        """ Samples the density in the streamwise direction with the length.
            The LAMMPS output file is 'denX.profile'"""

        length_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        length=np.asarray(length_list)

        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='BoxHeight':
                    total_box_height=float(line.split()[1])
                if line.split()[0]=='gapHeight':
                    gapheight=float(line.split()[1])

        global scale
        scale = total_box_height/float(gapheight)  # to correct for LAMMPS bin volume calculation
        densityX_over_time_list=self.time_avg_of_chunk[:,1]*1e3*scale                              # In kg/m3
        densityX_over_time=np.asarray(densityX_over_time_list)

        np.savetxt("denX-length.txt", np.c_[length,densityX_over_time],delimiter="  ",header="Length(nm)   Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Density $(kg/m^3)$')
        ax.plot(length,densityX_over_time,'-o')
        #ax.margins(x=None, y=1.0, tight=True)

        #tick_labels=myticks(densityX_over_time,100,-1)

        #ax.yaxis.set_ticks(tick_labels)

    def rhoX_time(self,fig):
        """ Samples the density in the streamwise direction with time.
            The LAMMPS output file is 'denX.profile'
        """

        densityX_each_chunk_list=self.time_avg_of_all_chunks[:,1]*1e3*scale                        # In kg/m3
        densityX_each_chunk=np.asarray(densityX_each_chunk_list)

        np.savetxt("denX-time.txt", np.c_[self.tSteps_array,densityX_each_chunk],delimiter="  ",header="Timestep    Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Desnity $(kg/m^3)$')
        ax.plot(self.tSteps_array,densityX_each_chunk,'-')


    def jX_length(self,fig):
        """ Samples the density in the streamwise direction with the length.
            The LAMMPS output file is 'denX.profile'"""

        length_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        length=np.asarray(length_list)

        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='gapHeight':
                    gapheight=float(line.split()[1])
                if line.split()[0]=='AChunk':
                    ChunkArea=float(line.split()[1])

        jX_over_time_list=self.time_avg_of_chunk[:,1]*1e26/(ChunkArea*gapheight)                       # In g/m2.n
        jX_over_time=np.asarray(jX_over_time_list)

        np.savetxt("denX-length.txt", np.c_[length,jX_over_time],delimiter="  ",header="Length(nm)   Mass flux(g/m^2.ns)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Mass flux $j_{x} \;(g/m^2.ns)$')
        ax.plot(length,jX_over_time,'-o')
        #ax.margins(x=None, y=1.0, tight=True)


#------------------------- Velocity profile --------------------------------#

    def Vx(self,fig):
        """
        Returns the velocity averaged over the bin in the flow direction (x)
        or in the wall-normal direction (z)
        """

        height_list=self.time_avg_of_chunk[:,0]/10
        height=np.asarray(height_list)

        vx_list=self.time_avg_of_chunk[:,1]*1e5
        vx=np.asarray(vx_list)

        # The shear rate is averaged only in the bulk (excludes the boundary points)
        slope_vx=np.polyfit(self.time_avg_of_chunk[4:-4,0],self.time_avg_of_chunk[4:-4,1],1)

        vx_at_lower_wall=self.time_avg_of_chunk[0][1]
        #print(self.time_avg_of_chunk[0])
        #print("Vx = %.10f" %vx_at_lower_wall)
        #print(self.time_avg_of_chunk[4:-4,0])

        ax = fig.add_subplot(111)
        ax.set_xlabel('Height $(nm)$')
        ax.set_ylabel('Vx $(m/s)$')
        ax.plot(height,vx,'o')

        #slope, intercept, r_value, p_value, std_err = stats.linregress(height, vx)
        #ax.plot(height[4:-4], intercept + slope*height[4:-4], 'r') #, label='fitted line')

        np.savetxt("velocityX.txt", np.c_[height,vx],delimiter="  ",header="Height(nm)   Vx(m/s)")

        return slope_vx[0],vx_at_lower_wall

#------------------------- Temperature profiles --------------------------------#

    def temp_height(self,fig):
        """ Samples the Temperature in the normal-to-wall direction with the height.
            The LAMMPS output file is 'denZ.profile'"""

        height_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        height=np.asarray(height_list)

        temp_over_time_list=self.time_avg_of_chunk[:,1]                        # In kg/m3
        temp_over_time=np.asarray(temp_over_time_list)

        np.savetxt("temp-height.txt", np.c_[height,temp_over_time],delimiter="  ",header="Height(nm)    Temperature(K)")

        ax = fig.add_subplot(111)
        ax.set_ylabel('Temperature $(K)$')
        ax.set_xlabel('Height $(nm)$')
        ax.plot(height,temp_over_time,'-o')
        ax.margins(x=None, y=1.0, tight=True)

    def temp_length(self,fig):
        """ Samples the Temperature in the normal-to-wall direction with the height.
            The LAMMPS output file is 'denZ.profile'"""

        length_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        length=np.asarray(length_list)

        temp_over_time_list=self.time_avg_of_chunk[:,1]                        # In kg/m3
        temp_over_time=np.asarray(temp_over_time_list)

        np.savetxt("temp-length.txt", np.c_[height,temp_over_time],delimiter="  ",header="Length(nm)    Temperature(K)")

        ax = fig.add_subplot(111)
        ax.set_ylabel('Temperature $(K)$')
        ax.set_xlabel('Length $(nm)$')
        ax.plot(length,temp_over_time,'-o')
        ax.margins(x=None, y=1.0, tight=True)


#------------------------- Stress profiles --------------------------------#

    def virial_pressureX(self,fig):
        """
        Returns the virial pressure in each fluid fluid averaged over time in the flow direction (x)
        """

        #ChunkVol= float(self.A[0,0,6])
        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='VChunkX':
                    ChunkVol=float(line.split()[1])
        atm_to_pa=101325
        kcalpermolA3_to_mpa=1/0.00014393

        avg=np.zeros((self.newChunks-1,5))

        # exclude the last chunk
        avg[:,0]=self.A[0,:-1,1]
        for i in range(self.tSteps):
            avg[:,1]+=self.A[i,:-1,2]         #  Count
            avg[:,2]+=self.A[i,:-1,3]         #  W1
            avg[:,3]+=self.A[i,:-1,4]         #  W2
            avg[:,4]+=self.A[i,:-1,5]         #  W3

        avg[:,1]/=self.tSteps
        avg[:,2]/=self.tSteps
        avg[:,3]/=self.tSteps
        avg[:,4]/=self.tSteps

        length_list=avg[:,0]/10
        length=np.asarray(length_list)

        virial_list=-avg[:,1]*(avg[:,2]+avg[:,3]+avg[:,4])*1e-6*atm_to_pa/(3*ChunkVol)
        virial=np.asarray(virial_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Pressure $(MPa)$')
        ax.plot(length,virial,'-o')

        #tick_labels=myticks(virial,10,-1)
        #ax.yaxis.set_ticks(tick_labels)

        np.savetxt("virialX.txt", np.c_[length,virial],delimiter="  ",header="Length(nm)    Pressure(MPa)")

    def virial_pressureZ(self,fig): # Change later
        """
        Returns the virial pressure in each fluid chunk averaged over time in the z-direction
        """
        #ChunkVol= float(self.A[0,0,6])
        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='VChunkZ':
                    ChunkVol=float(line.split()[1])
        atm_to_pa=101325
        kcalpermolA3_to_mpa=1/0.00014393

        avg=np.zeros((self.newChunks,5))

        avg[:,0]=self.A[0,:,1]
        for i in range(self.tSteps):
            avg[:,1]+=self.A[i,:,2]         #  Count
            avg[:,2]+=self.A[i,:,3]         #  W1
            avg[:,3]+=self.A[i,:,4]         #  W2
            avg[:,4]+=self.A[i,:,5]         #  W3

        avg[:,1]/=self.tSteps
        avg[:,2]/=self.tSteps
        avg[:,3]/=self.tSteps
        avg[:,4]/=self.tSteps

        height_list=avg[:,0]/10
        height=np.asarray(height_list)

        virial_list=-avg[:,1]*(avg[:,2]+avg[:,3]+avg[:,4])*1e-6*atm_to_pa/(3*ChunkVol)
        virial=np.asarray(virial_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Pressure $(MPa)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(virial,height,'-o')

        np.savetxt("virialZ.txt", np.c_[virial,height],delimiter="  ",header="Pressure(MPa)     Height(nm)")

    def sigzz(self,fig):
        """
        Returns the normal stress on the wall in each chunk averaged over time
        """
        with open('info.dat', 'r') as f:
            for line in f:
                if line.split()[0]=='AChunk':
                    ChunkArea=float(line.split()[1])

        #ChunkArea= 2.498*28.85 #float(self.A[-1,-1,4])
        avg=np.zeros((self.newChunks-2,4))
        atm_to_pa=101325
        kcalpermolA3_to_mpa=1/0.00014393

        avg[:,0]=self.A[0,1:-1,1]
        for i in range(self.tSteps):
            #avg[:,1]+=self.A[i,1:-1,2]         # Count
            avg[:,2]+=self.A[i,1:-1,3]         # Fz per atom

        #avg[:,1]/=self.tSteps
        avg[:,2]/=self.tSteps

        length_list=avg[:,0]/10
        length=np.asarray(length_list)

        sigmazz_list=-avg[:,2]*kcalpermolA3_to_mpa/(ChunkArea)
        sigmazz=np.asarray(sigmazz_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Pressure $(MPa)$')
        ax.plot(length,sigmazz,'-o')

        #ax.margins(x=None, y=1.0, tight=True)
        #tick_labels=myticks(sigmazz,100,-1)
        #ax.yaxis.set_ticks(tick_labels)

        #np.savetxt("sigmazz.txt", np.c_[length,sigmazz],delimiter="  ",header="Length(nm)       Sigmazz(MPa)")

        return length,sigmazz

#def main():
    #print("okay")

#if __name__ == "__main__":
#    main()
