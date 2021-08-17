#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021

@author: mtelewa
"""

# N: length of trajectory
# M: No. of blocks
# n: Block size

import numpy as np
from scipy.stats import norm

def block_ND_arr(tSample,N,newChunks,n):
    iter_block = n
    M = round(tSample/n)

    blocks=[]
    block_averages = []
    for i in range(M):
        blocks.append(np.reshape(N[(iter_block-n):iter_block],(n,newChunks)))
        iter_block+=n
    blocks=np.asarray(blocks)

    for i in range(M):
        block_averages.append(np.mean(blocks[i],axis=0))
    block_averages=np.asarray(block_averages)

    return block_averages


def block_1D_arr(N,n):
    M = np.int(len(N)/n)
    # Row is the variable, column is the timesteps
    blocks = np.reshape(N,(M,n))
    block_averages = np.mean(blocks,axis=1)

    average = np.mean(block_averages)

    return block_averages

# Test block size from Block Standard Error
def get_bse(N):
    n=[]

    for i in range(1, len(N)):
        if len(N) % i == 0:
            n.append(i)

    n=np.asarray(n)
    M=len(N)/n

    blocks,std_dev=[],[]
    for i in n:
        blocks.append(block_1D_arr(N,i))
        std_dev.append(np.std(blocks[-1], ddof=1))

    std_dev=np.asarray(std_dev)
    # Block standard error
    bse=std_dev/np.sqrt(M)

    return n, bse


# Confidence intervals
def get_err(N):

    if  N.ndim > 1:         # N.ndim
        avg = np.mean(N, axis=(0,1))
        std_dev = np.std(N, axis=(0,1), ddof=1)

    else:
        avg = np.mean(N)
        std_dev = np.std(N, ddof=1)

    sigma_lower, sigma_upper = avg-(3*std_dev), avg+(3*std_dev)

    dist = norm(avg, std_dev)
    values = [value for value in np.linspace(sigma_lower, sigma_upper, num=100, endpoint=True)]
    probabilities = [dist.pdf(value) for value in values]

    confidence_interval = 0.95      # 95% confidence interval
    alpha = 1.-confidence_interval
    cutoff = (alpha/2.)+confidence_interval       # 97.5th percentile
    z_score = norm.ppf(cutoff)
    margin_error = z_score*std_dev

    confidence_intervalL = avg-margin_error
    confidence_intervalU = avg+margin_error

    # print(confidence_intervalL, confidence_intervalU)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False)
    #plt.hist(avg_over_time, bins=30,normed=True)
    # plt.plot(values, probabilities, '-o', label='vx')
    # plt.axvline(x=avg_a, color='k', linestyle='dashed', label='$\mu$')
    # plt.axvline(x=confidence_intervalL, color='r', linestyle='dashed', label='$2.5^{th}$ percentile')
    # plt.axvline(x=confidence_intervalU, color='r', linestyle='dashed', label='$97.5^{th}$ percentile')
    # plt.legend()
    # plt.savefig('dist.png', dpi=100)

    # print("At a confidence interval of %g%%: \n\
    #       The averge value is \n\
    #       %.8f \n\
    #       Margin Error: %.3f \n\
    #       Upper limit: %.3f \n\
    #       Lower limit: %.3f" %(confidence_interval*100.,mu,margin_error,
    #                                confidence_intervalL, confidence_intervalU))

    return values, probabilities, margin_error



# def get_distrib(N):
#
#     avg_over_time=np.mean(N,axis=0)   # Average along time for each atom
#     std_dev=np.std(avg_over_time, ddof=1)
#
#     avg_arr=np.mean(avg_over_time)     # Mean of all velocities
#     sigma_lower,sigma_upper=avg_arr-(3*std_dev),avg_arr+(3*std_dev)
#
#     return avg_over_time,values



# def stats_allchunks(tSample,N,newChunks,n):
#     M = round(tSample/n)
#     arr=np.zeros((M,newChunks))
#     # print(arr.shape)
#     for i in range(newChunks):
#         arr[:,i][:]=block_data_profile(tSample,N,newChunks,n)[:,i][:]
#
#     print(arr)
#     return arr



# Autocorrelation functions
# ---------------------------------
#Axis 0 must be time
def acf(f_tq):
    n = f_tq.shape[0]
    var = np.var(f_tq, axis=0)
    f_tq -= np.mean(f_tq, axis=0)

    C = np.array([np.sum(f_tq * f_tq, axis=0) if i == 0 else np.sum(f_tq[i:] * f_tq[:-i], axis=0)
                  for i in range(n)]) / (n*var)

    return C
