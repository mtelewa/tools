#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from os.path import exists

def extract_data(logfile, skip):
    logfile_dir = os.path.dirname(logfile)

    if exists(f"{logfile_dir}/thermo.out"):
        os.system(f"rm {logfile_dir}/thermo*.out")

    os.system(f"cat {logfile_dir}/log.lammps | sed -n '/Step/,/Loop time/p' \
    | head -n-1 > {logfile_dir}/thermo.out")
    with open(f'{logfile_dir}/thermo.out', 'r') as f:
        for line in f:
            if line.split()[0]!='Step' and line.split()[0]!='Loop':
                with open(f"{logfile_dir + '/thermo2.out'}", "a") as o:
                    o.write(line)
            if line.split()[0]=='Step':
                thermo_variables = line.split()

    f = open(f'{logfile_dir}/thermo2.out', 'r')

    lines = f.readlines()
    for i,j in enumerate(lines[:-1]):
        if lines[i].split()[0]!=lines[i+1].split()[0]:
            with open(f"{logfile_dir + '/thermo3.out'}", "a") as o:
                o.write(j)

    data = np.loadtxt(f"{logfile_dir + '/thermo3.out'}", skiprows=skip, dtype=float)

    if exists(f"{logfile_dir}/thermo.out"):
        os.system(f"rm {logfile_dir}/thermo*.out")

    return data, thermo_variables
