#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 

def combine(*fnames):
"""
Merge files into one text file.
    Parameters
    ----------
    fnames : str
        Filenames
"""
    filenames = [fnames]
    with open('%s/merged.txt' %os.getcwd(), 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())
