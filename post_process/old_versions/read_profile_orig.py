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

def get_denProfile(fig):
    
    A=[]
    with open('den.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)
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
    
    ax = fig.add_subplot(121)
    ax.set_xlabel('Density (kg/m^3)')
    ax.set_ylabel('Height (nm)')
    ax.plot(avg[:,1]*1e3,avg[:,0]/10,'-^')

fig = plt.figure(figsize=(14.,6.))
get_denProfile(fig)
plt.savefig('density.png')

def get_velProfile(fig):
    
    A=[]
    with open('vel.profile', 'r') as f:
        for line in f:
            if len(line.split())==4 and line.split()[0]!='#':
                A.append(line.split())

    A=np.asarray(A)

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
        
    slope=np.polyfit(avg[20:-20,0],avg[20:-20,1],1)
       
    ax = fig.add_subplot(122)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Height (nm)')

    ax.plot(avg[:,1]/1e-5,avg[:,0]/10,'^')
    
    return slope[0]

fig = plt.figure(figsize=(14.,6.))
get_velProfile(fig)
plt.savefig('velocity.png')
    
def getTau():
    lower = np.loadtxt('stressL.txt',skiprows=2)
    upper = np.loadtxt('stressU.txt',skiprows=2)
    tau_lower = np.mean(lower,axis=0)[2]
    tau_upper = np.mean(upper,axis=0)[2]
    tau=0.5*(tau_lower-tau_upper)
    
    return tau

def main():
    fig = plt.figure(figsize=(14.,6.))
    
    get_denProfile(fig)
    
    slope = get_velProfile(fig)
    
    tau = getTau()

    eta = tau/slope
    print("viscosity: %4.3f mPas" % (eta*1e-12))
    
    
if __name__ == "__main__":
    main()

