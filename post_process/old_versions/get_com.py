#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:44:05 2019

@author: mtarek
"""

# script to determine the bulk region after compressing the box

import numpy as np
import os
import re
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
import get_stats as avg

#com = np.loadtxt('comZ.txt',skiprows=2,dtype=float)
#height = np.loadtxt('h.txt',skiprows=2,dtype=float)

final_height = avg.columnStats('h.txt',2,1) #(height[-1:,1])
bulk_center = avg.columnStats('comZ.txt',2,1)  #(com[-1:,1])
bulk_upper = bulk_center+0.5*bulk_center
bulk_lower = bulk_center-0.5*bulk_center

# Three methods to replace a string

# --------Method 1: Using Replace function----------
# Problem: need to specify the pattern to be replaced
# Not useful if we don't want to always set a value to
# the variables.

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

#replace("%s/header.LAMMPS" %os.getcwd(), "hAfterLoad equal ", "hAfterLoad equal %g" %final_height)
#replace("%s/header.LAMMPS" %os.getcwd(), "bulkStartZ equal ", "bulkStartZ equal %g" %bulk_lower)
#replace("%s/header.LAMMPS" %os.getcwd(), "bulkEndZ equal ", "bulkEndZ equal %g" %bulk_upper)


# --------Method 2: Searching and replacing the string (line.startswith)----------
# Problem: messes up the file structure.

#with open('header.LAMMPS', 'r') as f:
#    for line in f:
#        sline=line.split()
#        if len(sline)>2:
#            if sline[1].startswith("bulkStartZ"):
#                sline[3]="%g" %bulk_lower
#            if sline[1].startswith("bulkEndZ"):
#                sline[3]="%g" %bulk_upper
#            if sline[1].startswith("hAfterLoad"):
#                sline[3]="%g" %final_height
#        line=' '.join(sline)
#        fout = open("header3.LAMMPS", "a")
#        fout.write("%s" %line + '\n')

# --------Method 3: Using Regular expressions (re)----------

fout = open("header2.LAMMPS", "w+")

for line in open('header.LAMMPS','r').readlines():
  line = re.sub(r'bulkStartZ equal.+',r'bulkStartZ equal %g' %bulk_lower, line)
  line = re.sub(r'bulkEndZ equal.+',r'bulkEndZ equal %g' %bulk_upper, line)
  line = re.sub(r'gapHeight equal.+',r'gapHeight equal %g' %final_height, line)
  fout.write(line)

os.rename("header2.LAMMPS", "header.LAMMPS")
