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
from cycler import cycler

mpl.rcParams.update({'font.size': 18})
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

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

mpl.rcParams["figure.figsize"] = (16,10) # the standard figure size

mpl.rcParams["lines.linewidth"] = 1     # line width in points
mpl.rcParams["lines.markersize"] = 8
mpl.rcParams["lines.markeredgewidth"]=1   # the line width around the marker symbol


#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

#------------------------- Get the column averages --------------------------------#
def columnStats(filename,skip,column):

    load_file = np.loadtxt(filename,skiprows=skip,dtype=float)
    num_columns = load_file.shape[1]
    num_rows = load_file.shape[0]
    #num_columns_list=list(range(1,num_columns))

    # Every 100,000 time steps
    avg_every = 100
    values_arr = [load_file[i][column] for i in range(num_rows)]
    groups_of_data = len(values_arr)/avg_every
    #print(groups_of_data,len(values_arr))
    values_arr = np.reshape(values_arr,(round(groups_of_data),avg_every))
    #print(values_arr)
    block_mean = np.mean(values_arr,axis=1)
    #print(mean)
    mean_of_all_blocks = np.mean(block_mean)

    whole_mean = np.mean(load_file[:,column])
    std_dev=np.std(load_file[:,column])

    confidence_level = 0.95

    confidence_intervalU = whole_mean+(confidence_level*std_dev/np.sqrt(len(values_arr)))
    confidence_intervalL = whole_mean-(confidence_level*std_dev/np.sqrt(len(values_arr)))

    #print(confidence_intervalL,confidence_intervalU)

    # print("Column average taken every %g blocks: %.2f \nColumn average of the whole column: %.2f \n\
    # Upper confidence level: %.2f \nLower confidence level: %.2f " %(avg_every,mean_of_all_blocks,whole_mean,confidence_intervalU,confidence_intervalL))

    return mean_of_all_blocks

#------------------------- Plot from text files --------------------------------#
def plot_from_txt():

    plots = int(input("No. of plots on the same figure: "))
    #fig = plt.figure(figsize=(10.,8.))
    for i in range(plots):
        data = np.loadtxt(input("inputfile: "),skiprows=2,dtype=float)
        xdata_i = data[:,int(input("x-axis data: "))]
        ydata_i = data[:,int(input("y-axis data: "))]

        scale_x = float(input("scale x-axis: "))
        scale_y = float(input("scale y-axis: "))

        plt.plot(xdata_i*scale_x,ydata_i*scale_y,'-', alpha=float(input("alpha Value: ")))
        plt.text(xdata_i[-1], ydata_i[-1], input("Plot label: "), withdash=True)

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

        # the std deviation and mean within each chunk for all time steps
        std_dev=np.std(values_arr,axis=0)
        mean=np.mean(values_arr,axis=0)

        # Get the chunk ids to delete based on their mean (if the mean is zero)
        index=0
        idx=[]

        # check the mean of each chunk
        #print(mean)

        iqr_value=iqr(mean, axis=0)
        q1=np.quantile(mean,0.25)
        q3=np.quantile(mean,0.75)
        # For the profiles along length, we need to delete chunks with Inter Quartile Range
        # (IQR) values exceeding upper and lower quartiles
        # For the profiles along height, we need to remove chunks with mean value of zero
        if inputfile == 'denX.profile':
            delete_lower= q1-2.5*iqr_value      # 2.5 is used to avoid excluding useful data
            delete_upper= q1+2.5*iqr_value
            # Check the upper and lower limits
            #print(delete_lower,delete_upper)

            for i in range(len(mean)):
                if mean[i]<delete_lower or mean[i]>delete_upper:
                    idx.append(index)
                index+=1

        elif inputfile == 'sigmazz.profile':
            countAtomsInchunk=int(self.A[0][2])
            #print(countAtomsInchunk)
            for i in range(numChunks):
                if int(self.A[i][2]) == countAtomsInchunk:          ## BUG: which atom count to ignore?
                    idx.append(index)
                index+=1

        else:
            for i in range(len(mean)):
                if mean[i]==0:
                    idx.append(index)
                index+=1

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

    def rhoX_length(self,fig,gapHeight):
        """ Samples the density in the streamwise direction with the length.
            The LAMMPS output file is 'denX.profile'"""

        length_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        length=np.asarray(length_list)

        ## TODO: Replace the terminal input with reading volume from the profile file
        global scale
        scale = float(input("total box height: "))/float(gapHeight)  # to correct for LAMMPS bin volume calculation
        densityX_over_time_list=self.time_avg_of_chunk[:,1]*1e3*scale                              # In kg/m3
        densityX_over_time=np.asarray(densityX_over_time_list)

        mean_data = np.mean(densityX_over_time)
        #print(mean_data)

        np.savetxt("denX-length.txt", np.c_[length,densityX_over_time],delimiter="  ",header="Length(nm)   Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Density $(kg/m^3)$')
        ax.plot(length,densityX_over_time,'-o')

        range = max(densityX_over_time)-min(densityX_over_time)

        amin = min(densityX_over_time)-10*range
        amax = max(densityX_over_time)+10*range
        #print(amin,amax)

        interval = range*3
        #print(interval)

        tick_labels = np.arange(amin,amax,interval)
        tick_labels = tick_labels.astype(int)

        ax.yaxis.set_ticks(tick_labels)

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

        range_data = max(densityX_each_chunk)-min(densityX_each_chunk)

        amin = min(densityX_each_chunk)-100*range_data
        amax = max(densityX_each_chunk)+100*range_data
        #print(amin,amax)

        interval = range_data
        #print(interval)

        tick_labels = np.arange(amin,amax,interval)
        tick_labels = tick_labels.astype(int)

        ax.yaxis.set_ticks(tick_labels)


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

        slope=np.polyfit(self.time_avg_of_chunk[2:-2,0],self.time_avg_of_chunk[2:-2,1],1)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Vx $(m/s)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(vx,height,'-o')

        np.savetxt("velocityX.txt", np.c_[vx,height],delimiter="  ",header="Vx(m/s)    height(nm)")

        return slope[0]

#------------------------- Temperature profiles --------------------------------#

    def temp_height(self,fig):
        """ Samples the Temperature in the normal-to-wall direction with the height.
            The LAMMPS output file is 'denZ.profile'"""

        height_list=self.time_avg_of_chunk[:,0]/10                                     # In [nm]
        height=np.asarray(height_list)

        temp_over_time_list=self.time_avg_of_chunk[:,1]                        # In kg/m3
        temp_over_time=np.asarray(temp_over_time_list)

        np.savetxt("temp-height.txt", np.c_[temp_over_time,height],delimiter="  ",header="Temperature(K)    Height(nm)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Temperature $(K)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(temp_over_time,height,'-o')

#------------------------- Stress profiles --------------------------------#

    def virial_pressureX(self,fig):
        """
        Returns the virial pressure in each fluid fluid averaged over time in the flow direction (x)
        """

        global ChunkVol
        ChunkVol= float(self.A[0,0,6])
        global atm_to_pa
        atm_to_pa=101325
        global kcalpermolA3_to_mpa
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

        range_data = max(virial)-min(virial)

        amin = min(virial)-10*range_data
        amax = max(virial)+10*range_data
        #print(amin,amax)

        interval = range_data*3
        #print(interval)

        tick_labels = np.arange(amin,amax,interval)
        tick_labels = tick_labels.astype(int)

        ax.yaxis.set_ticks(tick_labels)

        np.savetxt("virialX.txt", np.c_[length,virial],delimiter="  ",header="Length(nm)    Pressure(MPa)")

    def virial_pressureZ(self,fig):
        """
        Returns the virial pressure in each fluid chunk averaged over time in the z-direction
        """

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

        ChunkArea= float(self.A[-1,-1,4])
        avg=np.zeros((self.newChunks,4))
        atm_to_pa=101325
        kcalpermolA3_to_mpa=1/0.00014393

        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,2]         # Count
                avg[i,2]+=self.A[j,i,3]         # Fz per atom

            avg[i,1]/=self.tSteps
            avg[i,2]/=self.tSteps

        length_list=avg[:,0]/10
        length=np.asarray(length_list)

        sigmazz_list=-avg[:,1]*avg[:,2]*kcalpermolA3_to_mpa/(ChunkArea)
        sigmazz=np.asarray(sigmazz_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Pressure $(MPa)$')
        ax.plot(length,sigmazz,'-o')

        range_data = max(sigmazz)-min(sigmazz)

        amin = min(sigmazz)-100*range_data
        amax = max(sigmazz)+100*range_data
        #print(amin,amax)

        interval = range_data
        #print(interval)

        tick_labels = np.arange(amin,amax,interval)
        tick_labels = tick_labels.astype(int)

        ax.yaxis.set_ticks(tick_labels)

        np.savetxt("sigmazz.txt", np.c_[length,sigmazz],delimiter="  ",header="Length(nm)       Sigmazz(MPa)")


#def main():
    #print("okay")

#if __name__ == "__main__":
#    main()
