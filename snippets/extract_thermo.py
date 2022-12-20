#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from os.path import exists

def extract_data(logfile, skip):
    logfile_dir = os.path.dirname(logfile)

    if exists(f"{logfile_dir}/thermo*.out"): os.system(f"rm {logfile_dir}/thermo*.out")
    
    os.system(f"cat {logfile_dir}/log.lammps | sed -n '/Step/,/Loop time/p' \
    | head -n-1 > {logfile_dir}/thermo.out")
    with open(f'{logfile_dir}/thermo.out', 'r') as f:
        for line in f:
            if line.split()[0]!='Step' and line.split()[0]!='Loop':
                with open(f"{logfile_dir + '/thermo2.out'}", "a") as o:
                    o.write(line)
            if line.split()[0]=='Step':
                thermo_variables = line.split()
    data = np.loadtxt(f"{logfile_dir + '/thermo2.out'}", skiprows=skip, dtype=float)

    return data, thermo_variables
