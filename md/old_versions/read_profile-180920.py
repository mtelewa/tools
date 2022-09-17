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

mpl.rcParams.update({'font.size': 20})
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

#------------------------- Get the column averages --------------------------------#
def columnAVE(filename,skip,column,startValue,endValue):

    load_file = np.loadtxt(filename,skiprows=skip,dtype=float)
    num_columns = load_file.shape[1]
    num_columns_list=list(range(1,num_columns))

    #for i in range(len(num_columns_list)):
    col = load_file[:,column][startValue:endValue]
    col_avg = np.mean(col)

    return col_avg

#------------------------- Plot from text files --------------------------------#
def plot_from_txt():

    plots = int(input("No. of plots on the same figure: "))
    fig = plt.figure(figsize=(10.,8.))
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

#------------------------- Density class --------------------------------#

class density:
    """
    Returns the density averaged over the bin and over time in the flow direction (x)
    or in the wall-normal direction (z)
    """

    def __init__(self,inputfile,ignore_chunks):
        self.A = []
        self.inputfile = inputfile
        with open(inputfile, 'r') as f:
            for line in f:
                if len(line.split())==4 and line.split()[0]!='#':
                    self.A.append(line.split())

        self.ignored_chunks= ignore_chunks            # Remove chunk pair (upper and lower)

        self.numChunks=int(self.A[-1][0])

        self.upperChunk= self.numChunks-self.ignored_chunks
        self.newChunks=self.numChunks-2*(self.ignored_chunks)   # remove 1 bottom and 1 upper chunk

        self.B = self.A.copy()
        for i in range(len(self.B)):
            if int(self.B[i][0])>self.upperChunk:    # if the chunk value exceeds the chunk upper limit
                self.A.remove(self.B[i])            # remove the upper bound chunk
                self.A.remove(self.B[abs(self.upperChunk-i)])     # and the lower bound

        self.A=np.asarray(self.A)
        #self.numChunks=int(self.A[-1,0])

        self.tSteps=int(len(self.A)/self.newChunks)
        self.tSteps_list=[]
        for i in range(self.tSteps):
            self.tSteps_list.append(i)
        self.tSteps_array=np.asarray(self.tSteps_list)

        self.A=np.reshape(self.A,(self.tSteps,self.newChunks,4))       #timesteps, chunks, value
        self.A=self.A.astype(np.float)

    def rhoZ_height(self,fig):
        """ Computes the density in the normal-to-wall direction with the height.
            The LAMMPS output file is 'denZ.profile'"""

        # Temporal average for each spatial bin
        avg=np.zeros((self.newChunks,2))         # the matrix to be filled with [height-rhoZ]

        # Looping over density at each chunk for all time steps
        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]               # the first index denotes height
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,3]          # second index denotes density of the bin

            avg[i,1]/=self.tSteps                # the density averaged over time for each spatial bin

        height_list=avg[:,0]/10                                     # In [nm]
        height=np.asarray(height_list)

        densityZ_over_time_list=avg[:,1]*1e3                        # In kg/m3
        densityZ_over_time=np.asarray(densityZ_over_time_list)

        np.savetxt("denZ-height.txt", np.c_[densityZ_over_time,height],delimiter="  ",header="Density(kg/m3)    Height(nm)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Density $(kg/m^3)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(densityZ_over_time,height,'-^')

    def rhoZ_time(self,fig):
        """ Computes the density in the normal-to-wall direction with time.
            The LAMMPS output file is 'denZ.profile'"""

        # Average of the whole fluid density at each time step
        avg=np.zeros((self.tSteps,2))

        # Looping over density at each timestep for all chunks
        for i in range(self.tSteps):
            #average[i,0].append(A[i,0,3])
            avg[i,0]=self.tSteps_array[i]              # first entry: time
            for j in range(self.newChunks):
                avg[i,1]+=(self.A[i,j,3])       # second entry: density

            avg[i,1]/=self.numChunks                 # the density averaged over the bins for each time step

        densityZ_each_chunk_list=avg[:,1]*1e3                        # In kg/m3
        densityZ_each_chunk=np.asarray(densityZ_each_chunk_list)

        np.savetxt("denZ-time.txt", np.c_[self.tSteps_array,densityZ_each_chunk],delimiter="  ",header="Timestep    Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Desnity $(kg/m^3)$')
        ax.plot(self.tSteps_array,densityZ_each_chunk,'-^')

    def rhoX_length(self,fig):
        """ Computes the density in the streamwise direction with the length.
            The LAMMPS output file is 'denX.profile'"""

        # Temporal average for each spatial bin
        avg=np.zeros((self.newChunks,2))         # the matrix to be filled with [height-rhoZ]

        # Looping over density at each chunk for all time steps
        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]               # the first index denotes height
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,3]          # second index denotes density of the bin

            avg[i,1]/=self.tSteps                # the density averaged over time for each spatial bin

        length_list=avg[:,0]/10                                     # In [nm]
        length=np.asarray(length_list)

        scale = float(input("total height: "))/float(input("region height: "))  # to correct for LAMMPS bin volume calculation
        densityX_over_time_list=avg[:,1]*1e3*scale                              # In kg/m3
        densityX_over_time=np.asarray(densityX_over_time_list)

        np.savetxt("denX-length.txt", np.c_[length,densityX_over_time],delimiter="  ",header="Length(nm)   Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Density $(kg/m^3)$')
        ax.plot(length,densityX_over_time,'-^')

    def rhoX_time(self,fig):
        """ Computes the density in the streamwise direction with time.
            The LAMMPS output file is 'denX.profile'"""

        # Average of the whole fluid density at each time step
        avg=np.zeros((self.tSteps,2))

        # Looping over density at each timestep for all chunks
        for i in range(self.tSteps):
            #average[i,0].append(A[i,0,3])
            avg[i,0]=self.tSteps_array[i]              # first entry: time
            for j in range(self.newChunks):
                avg[i,1]+=(self.A[i,j,3])       # second entry: density

            avg[i,1]/=self.newChunks                 # the density averaged over the bins for each time step

        scale = float(input("total height: "))/float(input("region height: "))  # to correct for LAMMPS bin volume calculation
        densityX_each_chunk_list=avg[:,1]*1e3*scale                        # In kg/m3
        densityX_each_chunk=np.asarray(densityX_each_chunk_list)

        np.savetxt("denX-time.txt", np.c_[self.tSteps_array,densityX_each_chunk],delimiter="  ",header="Timestep    Density(kg/m^3)")

        ax = fig.add_subplot(111)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Desnity $(kg/m^3)$')
        ax.plot(self.tSteps_array,densityX_each_chunk,'-^')

        amin = min(densityX_each_chunk)-50
        amax = max(densityX_each_chunk)+50
        ax.yaxis.set_ticks(np.arange(amin,amax,25.))

#------------------------- Velocity class --------------------------------#

class velocity:
    """
    Returns the velocity averaged over the bin in the flow direction (x)
    or in the wall-normal direction (z)
    """

    def __init__(self,inputfile,ignore_chunks):
        self.A = []
        self.inputfile = inputfile
        with open(inputfile, 'r') as f:
            for line in f:
                if len(line.split())==4 and line.split()[0]!='#':
                    self.A.append(line.split())

        self.ignored_chunks= ignore_chunks            # Remove chunk pair (upper and lower)

        self.numChunks=int(self.A[-1][0])

        self.upperChunk= self.numChunks-self.ignored_chunks
        self.newChunks=self.numChunks-2*(self.ignored_chunks)   # remove 1 bottom and 1 upper chunk

        self.B = self.A.copy()
        for i in range(len(self.B)):
            if int(self.B[i][0])>self.upperChunk:    # if the chunk value exceeds the chunk upper limit
                self.A.remove(self.B[i])            # remove the upper bound chunk
                self.A.remove(self.B[abs(self.upperChunk-i)])     # and the lower bound

        self.A=np.asarray(self.A)

        self.tSteps=int(len(self.A)/self.newChunks)

        self.A=np.reshape(self.A,(self.tSteps,self.newChunks,4))       #timesteps, chunks, value
        self.A=self.A.astype(np.float)

    def Vx(self,fig):
        # Average of the Vx for each chunk
        avg=np.zeros((self.newChunks,2))

        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,3]

            avg[i,1]/=self.tSteps

        height_list=avg[:,0]/10
        height=np.asarray(height_list)

        vx_list=avg[:,1]*1e5
        vx=np.asarray(vx_list)

        slope=np.polyfit(avg[2:-2,0],avg[2:-2,1],1)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Vx $(m/s)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(vx,height,'-^')

        np.savetxt("velocityX.txt", np.c_[vx,height],delimiter="  ",header="Vx(m/s)    height(nm)")

        return slope[0]

#------------------------- Stress class --------------------------------#

class stress:
    """
    Returns the stress tensor averaged over time in the flow direction (x)
    or in the wall-normal direction (z)
    """

    def __init__(self,inputfile,columns):
        self.A = []
        self.inputfile = inputfile
        self.colums = columns
        self.atm_to_pa=101325
        self.kcalpermolA3_to_mpa=1/0.00014393

        with open(inputfile, 'r') as f:
            for line in f:
                if len(line.split())==columns and line.split()[0]!='#':
                    self.A.append(line.split())

        self.numChunks=int(self.A[-1][0])
        if inputfile != "sigmazz.profile":
            self.ChunkVol=float(self.A[0][6])
        self.ChunkArea=float(self.A[-1][-1])

        self.tSteps=int(len(self.A)/self.numChunks)

        # Remove the chunks that have different atom count
        if inputfile == "sigmazz.profile":
            self.B = self.A.copy()
            self.countAtomsInchunk=int(self.A[0][2])            # Use the first atoms in chunk count as reference
            count=0
            for i in range(len(self.B)):
                if int(self.B[i][2]) != self.countAtomsInchunk:
                    #print("ok")
                    self.A.remove(self.B[i])            # remove that chunk
                    count+=1
                    self.newChunks=int(self.numChunks-(count/self.tSteps))

            self.A=np.asarray(self.A)
            self.A=np.reshape(self.A,(self.tSteps,self.newChunks,columns))       #timesteps, chunks, value
        else:
            self.A=np.asarray(self.A)
            self.A=np.reshape(self.A,(self.tSteps,self.numChunks,columns))       #timesteps, chunks, value

        self.A=self.A.astype(np.float)

    def virial_pressureX(self,fig):
        self.columns=7
        avg=np.zeros((self.numChunks,5))

        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,2]         #  Count
                avg[i,2]+=self.A[j,i,3]         #  W1
                avg[i,3]+=self.A[j,i,4]         #  W2
                avg[i,4]+=self.A[j,i,5]         #  W3

            avg[i,1]/=self.tSteps
            avg[i,2]/=self.tSteps
            avg[i,3]/=self.tSteps
            avg[i,4]/=self.tSteps

        length_list=avg[:,0]/10
        length=np.asarray(length_list)

        virial_list=-avg[:,1]*(avg[:,2]+avg[:,3]+avg[:,4])*1e-6*self.atm_to_pa/(3*self.ChunkVol)
        virial=np.asarray(virial_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Pressure $(MPa)$')
        ax.plot(length,virial,'-^')

        np.savetxt("virialX.txt", np.c_[length,virial],delimiter="  ",header="Length(nm)    Pressure(MPa)")

    def virial_pressureZ(self,fig):
        self.columns=7
        avg=np.zeros((self.numChunks,5))

        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,2]         #  Count
                avg[i,2]+=self.A[j,i,3]         #  W1
                avg[i,3]+=self.A[j,i,4]         #  W2
                avg[i,4]+=self.A[j,i,5]         #  W3

            avg[i,1]/=self.tSteps
            avg[i,2]/=self.tSteps
            avg[i,3]/=self.tSteps
            avg[i,4]/=self.tSteps

        height_list=avg[:,0]/10
        height=np.asarray(height_list)

        virial_list=-avg[:,1]*(avg[:,2]+avg[:,3]+avg[:,4])*1e-6*self.atm_to_pa/(3*self.ChunkVol)
        virial=np.asarray(virial_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Pressure $(MPa)$')
        ax.set_ylabel('Height $(nm)$')
        ax.plot(virial,height,'-^')

        np.savetxt("virialZ.txt", np.c_[virial,height],delimiter="  ",header="Pressure(MPa)     Height(nm)")

    def sigzz(self,fig):
        self.columns=5
        avg=np.zeros((self.newChunks,4))

        for i in range(len(avg)):
            avg[i,0]=self.A[0,i,1]
            for j in range(self.tSteps):
                avg[i,1]+=self.A[j,i,2]         # Count
                avg[i,2]+=self.A[j,i,3]         # Fz per atom

            avg[i,1]/=self.tSteps
            avg[i,2]/=self.tSteps

        length_list=avg[:,0]/10
        length=np.asarray(length_list)

        sigmazz_list=-avg[:,1]*avg[:,2]*self.kcalpermolA3_to_mpa/(self.ChunkArea)
        sigmazz=np.asarray(sigmazz_list)

        ax = fig.add_subplot(111)
        ax.set_xlabel('Length $(nm)$')
        ax.set_ylabel('Pressure $(MPa)$')
        ax.plot(length,sigmazz,'-^')

        np.savetxt("sigmazz.txt", np.c_[length,sigmazz],delimiter="  ",header="Length(nm)       Sigmazz(MPa)")


#def main():
    #print("okay")

#if __name__ == "__main__":
#    main()
