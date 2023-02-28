#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from sklearn.utils import resample

with open(sys.argv[1], 'r') as f:
    values = [float(x) for x in f.readlines()]

values = np.array(values)*3/2

# set the number of bootstrap samples
num_bootstraps = 1000

# create an empty array to store the means
bootstrap_means = np.zeros(num_bootstraps)

# loop through the number of bootstrap samples
for i in range(num_bootstraps):
    # generate a bootstrap sample using scikit-learn's bootstrap function
    bootstrap_sample = resample(values, replace=True)
    # calculate the mean of the bootstrap sample
    bootstrap_means[i] = np.mean(bootstrap_sample)

# calculate the mean and standard deviation of the bootstrap means
mean_of_means = np.mean(bootstrap_means)
std_of_means = np.std(bootstrap_means)

# calculate the 95% confidence interval
conf_int = np.percentile(bootstrap_means, [2.5, 97.5])

# print the results
print("Mean of means:", mean_of_means)
#print("Standard deviation of means:", std_of_means)
print("95% confidence interval:", conf_int)
