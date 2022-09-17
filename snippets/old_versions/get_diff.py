#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys

def get_diff(infile,expected):
    data = np.loadtxt(infile,skiprows=1,dtype=float)
    values = data[:,1]
    max_value =  np.max(values)
    min_value =  np.min(values)
    # print(min_value,max_value)
    observed = max_value-min_value
    abs_err = abs(expected-observed)
    rel_err = (abs(expected-observed)/expected)*100

    print('Absolute Error is {0:.4f} \nRelative Error is {1:.2f}% \
    \nObserved difference is {2:.4f}'.format(abs_err,rel_err,observed))
    # return absolute_error,relative_error

if __name__ == "__main__":
    get_diff(sys.argv[1],np.float(sys.argv[2]))
