#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

def extract_data(logfile, skip):
    os.system(f"rm {os.path.dirname(logfile)}/thermo*.out")
    os.system(f"cat {os.path.dirname(logfile)}/log.lammps | sed -n '/Step/,/Loop time/p' \
    | head -n-1 > {os.path.dirname(logfile)}/thermo.out")
    with open(f'{os.path.dirname(logfile)}/thermo.out', 'r') as f:
        for line in f:
            if line.split()[0]!='Step' and line.split()[0]!='Loop':
                with open(f"{os.path.dirname(logfile) + '/thermo2.out'}", "a") as o:
                    o.write(line)
            if line.split()[0]=='Step':
                thermo_variables = line.split()
    data = np.loadtxt(f"{os.path.dirname(logfile) + '/thermo2.out'}", skiprows=skip, dtype=float)

    return data, thermo_variables
