#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

inp1=open("/work2/atom/sims/thermostated/main/steps/04-force/dump_nvt.lammpstrj","r")
out=open("track-atom.txt", "w+")
lines1=inp1.readlines()
val=0

out.write("#time   fx   fy  fz " + "\n")
out.close()


out=open("track-atom.txt", "a+")

for x in lines1:
    x.split()

    if x.startswith("4761 1"):
        val+=1
        out.write("%g" %val +  "  " + "%s" %(x.split()[6]) + "  " +"%s" %(x.split()[7]) + "  "  +"%s" %(x.split()[8]) + "\n")

inp1.close()

out.close()
