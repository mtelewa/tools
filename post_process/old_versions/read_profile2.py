#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:44:05 2019

@author: hannesh
"""

# script to compute density and velocity profile from LAMMPS output
# compute dynamic viscosity for Couette flow
# tested with pentane h50

import numpy as np
import matplotlib.pyplot as plt

#------------------------- Density along the height --------------------------------#

def get_denProfileZ(fig):

    A=[]
    with open('denZ.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)

    avg=np.zeros((numChunks,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i,3]

        avg[i,1]/=tSteps

    height_list=avg[:,0]/10
    height=np.asarray(height_list)

    densityZ_list=avg[:,1]*1e3
    densityZ=np.asarray(densityZ_list)


    ax = fig.add_subplot(111)
    ax.set_xlabel('Density (kg/m^3)')
    ax.set_ylabel('Height (nm)')
    ax.plot(densityZ,height,'-^')

    np.savetxt("densityZ.txt", np.c_[densityZ,height])

fig1 = plt.figure(figsize=(10.,8.))
get_denProfileZ(fig1)
plt.savefig('densityZ.png')


#------------------------- Density along the length --------------------------------#

def get_denProfileX(fig):

    A=[]
    with open('denX.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    global numChunks
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)

    avg=np.zeros((numChunks-12,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i+6,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i+6,3]

        avg[i,1]/=tSteps

    length_list=avg[:,0]/10
    length=np.asarray(length_list)

    densityX_list=avg[:,1]*1e3
    densityX=np.asarray(densityX_list)

    ax = fig.add_subplot(111)
    ax.set_xlabel('length (nm)')
    ax.set_ylabel('Density (kg/m^3)')
    ax.plot(length,densityX,'-^')

    np.savetxt("densityX.txt", np.c_[length,densityX])

    #amin = 450
    #amax = 650
    #ax.yaxis.set_ticks(np.arange(amin,amax,25.))

fig2 = plt.figure(figsize=(10.,8.))
get_denProfileX(fig2)
plt.savefig('densityX.png')

#------------------------- Vz along the height --------------------------------#

def get_velProfileZ(fig):

    A=[]
    with open('vz.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)

    avg=np.zeros((numChunks,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i,3]

        avg[i,1]/=tSteps

    height_list=avg[:,0]/10
    height=np.asarray(height_list)

    vz_list=avg[:,1]*1e5
    vz=np.asarray(vz_list)

    ax = fig.add_subplot(111)
    ax.set_xlabel('Vz (m/s)')
    ax.set_ylabel('Height (nm)')
    ax.plot(vz,height,'-^')

    np.savetxt("velocityZ.txt", np.c_[vz,height])

fig4 = plt.figure(figsize=(10.,8.))
get_velProfileZ(fig4)
#plt.savefig('Vz.png')

#------------------------- Vx along the height --------------------------------#

def get_velProfileX(fig):

    A=[]
    with open('vx.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
    numChunks=int(A[-1,0])
    tSteps=int(len(A)/numChunks)

    A=np.reshape(A,(tSteps,numChunks,4))       #timesteps, chunks, value
    A=A.astype(np.float)
    
    global avg
    avg=np.zeros((numChunks-4,2))

    for i in range(len(avg)):
        avg[i,0]=A[0,i+2,1]
        for j in range(tSteps):
            avg[i,1]+=A[j,i+2,3]

        avg[i,1]/=tSteps


    height_list=avg[:,0]/10
    height=np.asarray(height_list)

    vx_list=avg[:,1]*1e5
    vx=np.asarray(vx_list)


    slope=np.polyfit(avg[2:-2,0],avg[2:-2,1],1)

    ax = fig.add_subplot(111)
    ax.set_xlabel('Vx (m/s)')
    ax.set_ylabel('Height (nm)')
    ax.plot(vx,height,'-^')

    np.savetxt("velocityX.txt", np.c_[vx,height])

    return slope[0]


fig5 = plt.figure(figsize=(10.,8.))
get_velProfileX(fig5)
plt.savefig('Vx.png')

#------------------------- Regional pressure --------------------------------#

num_columns = np.loadtxt('flux-virial.txt', dtype='str').shape[1]
chunks=list(range(1,num_columns))

plot_virial = np.loadtxt('flux-virial.txt',skiprows=2,dtype=float)
plot_sigzz = np.loadtxt('sigzz.txt',skiprows=2,dtype=float)

average_virial = []
average_sigzz = []

for i in chunks:
    average_virial.append(np.mean(plot_virial[:,i]))
    average_sigzz.append(np.mean(plot_sigzz[:,i]))

figVirial = plt.figure(figsize=(10.,8.))

plt.plot(np.asarray(chunks),np.asarray(average_virial)*1e-6,'-^', label= 'Virial pressure')
plt.plot(np.asarray(chunks),np.asarray(average_sigzz)*1e-6,'-^', label= r'$\sigma_{zz}$')


plt.xlabel('Region')
plt.ylabel('Pressure (MPa)')
plt.legend()
plt.show()
plt.savefig('pressure-region.png')

# ---------------------- Pressure with time --------------------------------#

t=np.loadtxt('h.txt',skiprows=2,dtype=float)
time=t[:,0]
time_start_avg=time[600:]

stressZZ= np.loadtxt('stress-tensor.txt',skiprows=2,dtype=float)
virial_press= np.loadtxt('surface.txt',skiprows=2,dtype=float)

a=stressZZ[:,3][600:]       # Last column is sigma_zz
b=virial_press[:,3]

avg_sigzz=[]
avg_virial=[]

for i in range(len(a)):
    avg_sigzz.append(np.mean(a))
    avg_virial.append(np.mean(b))

fig100 = plt.figure(figsize=(10.,8.))

plt.plot(time_start_avg*1e-6,np.asarray(a)*1e-6,'-', color='orange', label= r'$\sigma_{zz}$',alpha=0.5)
plt.plot(time_start_avg*1e-6,np.asarray(b)*1e-6,'-',  color='b', label= 'Virial pressure',alpha=0.5)
plt.plot(time_start_avg*1e-6,np.asarray(avg_virial)*1e-6,'-',color='b')
plt.plot(time_start_avg*1e-6,np.asarray(avg_sigzz)*1e-6,'-',color='orange')

plt.xlabel('Time (ns)')
plt.ylabel('Pressure (MPa)')
plt.legend()
plt.show()
plt.savefig('pressure-time.png')

#--------------------Flux with time----------------------------------------#

plot_flux_time = np.loadtxt('flux-time.txt',skiprows=2,dtype=float)

flux_theo = plot_flux_time[:,1]
flux_calc = plot_flux_time[:,2]

figFlux = plt.figure(figsize=(10.,8.))

plt.plot(np.asarray(time)*1e-6,np.asarray(flux_theo)[1:],'-', label= 'Flux imposed')
plt.plot(np.asarray(time)*1e-6,np.asarray(flux_calc)[1:],'-', label= 'Flux calculated')

plt.xlabel('time (ns)')
plt.ylabel('Flux (g/m^2.ns)')
plt.legend()
plt.show()
plt.savefig('flux-time.png')

#--------------------Regional flux----------------------------------------#

plot_flux_region = np.loadtxt('flux-region.txt',skiprows=2,dtype=float)

average_flux=[]

for i in chunks:
    average_flux.append(np.mean(plot_flux_region[:,i]))

figFlux = plt.figure(figsize=(10.,8.))

plt.plot(np.asarray(chunks),np.asarray(average_flux),'-^', label= 'Mass flux')

plt.xlabel('Region')
plt.ylabel('Flux (g/m^2.ns)')
plt.legend()
plt.show()
plt.savefig('flux-region.png')

# -------------------- Other stresses with time  ---------------#

plot_arrU = np.loadtxt('stressU.txt',skiprows=2,dtype=float)
plot_arrL = np.loadtxt('stressL.txt',skiprows=2,dtype=float)

sigxzU = plot_arrU[:,1]
sigyzU = plot_arrU[:,2]
sigzzU = plot_arrU[:,3]

mean_sigxzU = np.mean(plot_arrU[:,1])
mean_sigyzU = np.mean(plot_arrU[:,2])
mean_sigzzU = np.mean(plot_arrU[:,3])

sigxzL = plot_arrL[:,1]
sigyzL = plot_arrL[:,2]
sigzzL = plot_arrL[:,3]

mean_sigxzL = np.mean(plot_arrL[:,1])
mean_sigyzL = np.mean(plot_arrL[:,2])
mean_sigzzL = np.mean(plot_arrL[:,3])

mean_sigxz=-0.5*(np.asarray(mean_sigxzU)-np.asarray(mean_sigxzL))

avg_sigxzU=[]
avg_sigyzU=[]
avg_sigzzU=[]
avg_sigxzL=[]
avg_sigyzL=[]
avg_sigzzL=[]

for i in range(len(time)):
    avg_sigxzU.append(np.mean(mean_sigxzU))
    avg_sigyzU.append(np.mean(mean_sigyzU))
    avg_sigzzU.append(np.mean(mean_sigzzU))
    avg_sigxzL.append(np.mean(mean_sigxzL))
    avg_sigyzL.append(np.mean(mean_sigyzL))
    avg_sigzzL.append(np.mean(mean_sigzzL))

figstress = plt.figure(figsize=(10.,8.))


plt.plot(time*1e-6,np.asarray(sigxzU)*1e-6,'-', label= r'$\sigma_{xz}$ Upper',alpha=0.5,color='r')
plt.plot(time*1e-6,np.asarray(sigyzU)*1e-6,'-', label= r'$\sigma_{yz}$ Upper',alpha=0.5,color='g')
plt.plot(time*1e-6,np.asarray(sigxzL)*1e-6,'-', label= r'$\sigma_{xz}$ Lower',alpha=0.5,color='c')
plt.plot(time*1e-6,np.asarray(sigyzL)*1e-6,'-', label= r'$\sigma_{yz}$ Lower',alpha=0.5,color='gray')

plt.plot(time*1e-6,np.asarray(avg_sigxzU)*1e-6,'-',color='r')
plt.plot(time*1e-6,np.asarray(avg_sigyzU)*1e-6,'-',color='g')
plt.plot(time*1e-6,np.asarray(avg_sigxzL)*1e-6,'-',color='c')
plt.plot(time*1e-6,np.asarray(avg_sigyzL)*1e-6,'-',color='gray')

plt.xlabel('Time (ns)')
plt.ylabel('Pressure (MPa)')
plt.legend()
plt.show()
plt.savefig('other-stresses.png')

# --------------------------------------

def getTau():
    lower = np.loadtxt('stressL.txt',skiprows=2)
    upper = np.loadtxt('stressU.txt',skiprows=2)
    tau_lower = np.mean(lower,axis=0)[2]
    tau_upper = np.mean(upper,axis=0)[2]
    tau=0.5*(tau_lower-tau_upper)

    return tau

slope = get_velProfileX(fig5)
tau = getTau()
#tau = mean_sigmaxz 
eta = tau/slope

#eta = mean_sigxz/slope

with open('visocisty.txt', 'w') as f:
    print("viscosity: %4.3f mPas" % (eta*1e-12), file=f)


#def main():
#    get_denProfile(fig)

#if __name__ == "__main__":
#    main()
