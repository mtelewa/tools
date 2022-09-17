#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 2020

@author: mtelewa
"""

import numpy as np
import sys
import os
from scipy.stats import iqr
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

#------------------------- Get the column averages --------------------------------#
def columnStats(infile,skip,column):
    """

    Parameters:
    -----------

    Returns:
    -------

    """

    load_file = np.loadtxt(infile,skiprows=skip,dtype=float)
    num_columns = load_file.shape[1]
    num_rows = load_file.shape[0]
    #num_columns_list=list(range(1,num_columns))

    whole_data = [load_file[i][column] for i in range(num_rows)]
    # M = len(whole_data)/n
    # Row is the values in the sample, column is the timesteps
    # blocks = np.reshape(whole_data,(round(M),n))
    # block_mean = np.mean(blocks,axis=1)
    #block_std_dev=np.std(blocks,axis=1)
    # print(block_mean)

    mu = np.mean(whole_data)
    sigma=np.std(whole_data, ddof=1)

    #Block standard error
    # bse=sigma/np.sqrt(M)
    # print(bse)

    #whole_mean = np.mean(load_file[:,column])
    #sigma=np.mean(block_std_dev/np.sqrt(M))
    #sigma=np.std(blocks,axis=1)/np.sqrt(M)

    # confidence_interval = 0.95      # 95% confidence interval
    # alpha = 1.-confidence_interval
    # cutoff = (alpha/2.)+confidence_interval       # 97.5th percentile
    # z_score=norm.ppf(cutoff)
    # margin_error=z_score*sigma/np.sqrt(len(blocks))
    #
    # confidence_intervalU = mu+margin_error
    # confidence_intervalL = mu-margin_error

    # print("At a confidence interval of %g%%: \n\
    #       The averge value of %s is \n\
    #       %.8f \n\
    #       Margin Error: %.3f \n\
    #       Upper limit: %.3f \n\
    #       Lower limit: %.3f" %(confidence_interval*100.,file_loc,mu,margin_error,confidence_intervalL,
    #                             confidence_intervalU))

    # print("Column average taken every %g blocks: %.2f \nColumn average of the whole column: %.2f \n\
    # # Upper confidence level: %.2f \nLower confidence level: %.2f " %(avg_every,mean_of_all_blocks,whole_mean,confidence_intervalU,confidence_intervalL))

    return mu


if __name__ == "__main__":
    columnStats(sys.argv[1],int(sys.argv[2]),1,int(sys.argv[3]))
