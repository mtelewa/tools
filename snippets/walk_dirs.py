#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from progressbar import progressbar as pb
from get_stats import columnStats as avg

# 1st level
dirs=[x[0] for x in os.walk(os.getcwd())][1:]

# 2nd level
subdirs=[]
for i in dirs:
    if 'out' in i:
        subdirs.append(i)

def exec_cmd(cmd):
    """
    Execute a command several times in the 'out' subdirectories
    Parameters
    ----------
    cmd : str
        Command.
    """
    n=0
    for i in sorted(subdirs):
        try:
            os.system('cd %s; %s' %(i,cmd))
            pb(n+1,len(subdirs))
            n+=1
        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError as err2:
            print("Value error: {0}".format(err2))

if __name__ == "__main__":
    command = ' '.join(sys.argv[1:])
    exec_cmd(command)
