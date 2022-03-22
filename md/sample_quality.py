#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021

@author: mtelewa
"""

import numpy as np
from scipy.stats import norm

def block_ND(tSample, f_tq, nChunks, n):
    """
    Block sampling of a 2D array through the trajectory. Each block contains n timesteps.
    parameters
    ----------
    tSample: int
        sampling time
    f_tq: 2-dim arr
        input array with shape (time, Nchunks)
    nChunks: int
        number of chunks
    n: block size
        number of timesteps in each block

    returns
    -------
    block_averages: arr
        average of each block of time
    """
    # No. of blocks
    M = round(tSample/n)

    iter_block = n
    blocks=[]
    block_averages = []

    for i in range(M):
        blocks.append(np.reshape(f_tq[(iter_block-n):iter_block],(n, nChunks)))
        iter_block+=n
    blocks=np.asarray(blocks)

    for i in range(M):
        block_averages.append(np.mean(blocks[i],axis=0))
    block_averages=np.asarray(block_averages)

    return block_averages


def block_1D(f_t, n):
    """
    Block sampling of a 1D array through the trajectory. Each block contains n timesteps.
    parameters
    ----------
    f_t: 1-dim arr
        input time-series array
    n: block size
        number of timesteps in each block

    returns
    -------
    block_averages: arr
        average of each block of time
    """
    # No. of blocks
    M = np.int(len(f_t)/n)
    # Row is the variable, column is the timesteps
    blocks = np.reshape(f_t, (M, n))
    block_averages = np.mean(blocks, axis=1)

    average = np.mean(block_averages)

    return block_averages


def get_bse(f_t):
    """
    Test block size by measuring the Block Standard Error (BSE)
    The chose block size is determined as we see a plateau in the BSE.
    parameters
    ----------
    f_t: 1-dim arr
        input time-series array

    returns
    -------
    n: int
        block size
    bse: int
        BSE corresponding to the block size
    """

    n=[]
    for i in range(1, len(f_t)):
        if len(f_t) % i == 0:
            n.append(i)
    n=np.asarray(n)

    # No. of blocks
    M=len(f_t)/n

    blocks,std_dev=[],[]
    for i in n:
        blocks.append(block_1D(f_t,i))
        std_dev.append(np.std(blocks[-1], ddof=1))
    std_dev=np.asarray(std_dev)

    # Block standard error
    bse=std_dev/np.sqrt(M)

    return n, bse


def acf(f_tq):
    """
    Autocorrelation function
    parameters
    ----------
    f_tq: arr
        input array with shape (time,)
    """
    n = f_tq.shape[0]
    var = np.var(f_tq, axis=0)
    f_tq -= np.mean(f_tq, axis=0)

    C_non_norm = np.array([np.sum(f_tq * f_tq, axis=0) if i == 0
            else np.sum(f_tq[i:] * f_tq[:-i], axis=0)
                  for i in range(n)])

    C_norm = C_non_norm / (n*var)


    return {'non-norm':C_non_norm, 'norm':C_norm}


def acf_conjugate(f):
    """
    Autocorrelation function
    parameters
    ----------
    f_tq: arr
        input array with shape (time,)
    """
    var = np.var(f)
    mean = np.mean(f)
    print(mean.shape)
    C = np.correlate(f - mean, f - mean, mode="full")
    C = C[C.size // 2:]
    C /= len(C)
    C /= var

    return {'C':C}



def acf_fft(f_tq):
    """
    Compute normalized time autocorrelation function of multidimensional time
    series using FFT (Wiener-Khinchin theorem). Since time is generally non-
    periodic, the array is zero padded and the result of the inverse FFT is
    divided by the size of the overlap.
    Parameters
    ----------
    f_tq : numpy.ndarray
        array containing the multidimensional time series, 1st axis is time
    Returns
    ----------
    C : numpy.ndarray
        ACF function
    """

    n = f_tq.shape[0]
    var = np.var(f_tq, axis=0)
    f_tq -= np.mean(f_tq, axis=0)

    # pad array with zeros
    ext_size = 2 * n - 1
    fsize = 2**np.ceil(np.log2(ext_size)).astype('int')


    # do fft and ifft
    f_wq = np.fft.fft(f_tq, fsize, axis=0)
    C_tq = np.fft.ifft(f_wq * f_wq.conjugate(), axis=0)[:n] / \
            (n - np.arange(n))[:, np.newaxis, np.newaxis, np.newaxis]

    var[var == 0] = 1.

    C_tq /= var

    return C_tq.real



def get_err(f):
    """
    Compute uncertainties

    parameters
    ----------
    f: arr
        input array of shape (time,)

    """
    if  f.ndim > 1:         # c.ndim
        n = 100 # samples/block
        if f.shape[1] > 1: blocks = block_ND(f.shape[0], f, f.shape[1], n)  #Qtts along length
        if f.shape[2] > 1: blocks = block_ND(f.shape[0], f, f.shape[2], n)  #Qtts along height
        # Discard chunks with zero time average
        # Get the time average in each chunk
        blocks_mean = np.mean(blocks, axis=0)
        blocks = np.transpose(blocks)
        blocks = blocks[blocks_mean !=0]
        blocks = np.transpose(blocks)

        avg = np.mean(blocks, axis=0)
        std_dev = np.std(blocks, axis=0, ddof=1)

    else:
        avg = np.mean(f)
        std_dev = np.std(f, ddof=1)

    # 3-sigma bounds
    sigma_lower, sigma_upper = avg-(3*std_dev), avg+(3*std_dev)

    # Normal distribution object
    dist = norm(avg, std_dev)
    # values from lower to upper 3-sigma bounds
    values = [value for value in np.linspace(sigma_lower, sigma_upper, num=100, endpoint=True)]
    # Probability density function
    probabilities = [dist.pdf(value) for value in values]

    # Confidence Interval
    confidence_interval = 0.95
    # Significance level (prob. of rejecting null hypothesis when it is true.)
    alpha = 1. - confidence_interval
    # 97.5th percentile
    cutoff = (alpha/2.) + confidence_interval
    # Percent point function
    z_score = norm.ppf(cutoff)
    # The uncertainty
    margin_error = z_score * std_dev

    confidence_intervalL = avg - margin_error
    confidence_intervalU = avg + margin_error

    return {'values':values, 'probs':probabilities, 'uncertainty':margin_error,
            'lo':confidence_intervalL, 'hi':confidence_intervalU}


def prop_uncertainty(f1,f2):
    """
    Compute propagated uncertainties from 2 arrays of the same shape

    parameters
    ----------
    f1: arr
        input array of shape (time, Nchunks)
    f2: arr
        input array of shape (time, Nchunks)
    """

    err1 = get_err(f1)['uncertainty']
    err2 = get_err(f2)['uncertainty']

    # The uncertainty in each chunk is the Propagation of uncertainty of L and U surfaces
    # Get the covariance in the chunk
    cov_in_chunk = np.zeros(f1.shape[1])
    for i in range(f1.shape[1]):
        cov_in_chunk[i] = np.cov(f1[:,i], f2[:,i])[0,1]

    # Needs to be corrected with a positive sign of the variance for sigxz in PD
    err = np.sqrt(0.5**2*err1**2 + 0.5**2*err2**2 - 2*0.5*0.5*cov_in_chunk)
    avg = 0.5 * (np.mean(f1, axis=0) - np.mean(f2, axis=0))
    lo = avg - err
    hi = avg + err

    return {'err':err, 'lo':lo, 'hi':hi}




#
