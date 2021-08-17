#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 24 2020

@author: mtelewa
"""

"""
Determine the bulk region after loading the box
and modify the header file.
Input:
------
h.txt, header.LAMMPS, comZ.txt
Output:
-------
modified header.LAMMPS
"""

import numpy as np
import os
import re
import sys
import get_stats_txt as avg
import netCDF4



# Get the new bulk limits --------------
def get_com(infile, skip):

    data = netCDF4.Dataset(infile)

    com = np.array(data.variables["COM"])
    com = com.flatten()
    bulk_center = np.mean(com[skip:])

    h = np.array(data.variables["Height"])
    h = h.flatten()
    final_height = np.mean(h[skip:])

    print(bulk_center, final_height)

    # final_height = avg.columnStats('h.txt',skip,1) # skip the first 500 time steps
    # bulk_center = avg.columnStats('com.txt',skip,1)
    #
    bulk_upper = bulk_center+0.5*bulk_center
    bulk_lower = bulk_center-0.5*bulk_center
    #
    # print(bulk_center, final_height)

    # Modify the header

    fout = open("header2.LAMMPS", "w+")

    for line in open('header.LAMMPS','r').readlines():
      line = re.sub(r'bulkStartZ equal.+',r'bulkStartZ equal %g' %bulk_lower, line)
      line = re.sub(r'bulkEndZ equal.+',r'bulkEndZ equal %g' %bulk_upper, line)
      line = re.sub(r'gapHeight equal.+',r'gapHeight equal %g' %final_height, line)
      fout.write(line)

    os.rename("header2.LAMMPS", "header.LAMMPS")

if __name__ == "__main__":
    get_com(sys.argv[1], np.int(sys.argv[2]))
