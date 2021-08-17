import gc, glob, io, os, re, sys
import numpy as np
import matplotlib.pyplot as plt
#import ase, ase.io, ase.visualize # read and visualize LAMMPS trajectories
#import pandas as pd
#import scipy as scp # here for handling rotations as matrices and quaternions
#import scipy.spatial 
#from pprint import pprint

from fireworks import Firework, LaunchPad, ScriptTask, Workflow
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask
from fireworks.utilities.filepad import FilePad

prefix = os.getcwd()

# to start with
infile_prefix = os.path.join(prefix,'01-initial', '02-equi', '03-load', '04-flow')

# the FireWorks LaunchPad
lp = LaunchPad.auto_load()
#fp = FilePad.auto_load()
fp = FilePad.from_db_file('/home/mohamed/.fireworks/fireworks_launchpad_imtek.yaml')
#fp = FilePad.from_db_file('/home/mohamed/.fireworks/my_launchpad_testing.yaml')


project_id = 'pentane'

# input and data files for the current parametric study
files = {
    # one file describing the interatomic potential
    'potential':             'coeff.LAMMPS',
    # several LAMMPS input files for the above steps
    'initialize' : 'initialize.in',
    'equilibrate': 'equilib.in',
    'load':   'load.LAMMPS',
    'flow':   'flow.LAMMPS'
}

# metadata common to all these files
metadata = {
    'project': project_id,
    'type': 'input'
}

fp_files = []


# insert these input files into data base
for name, file in files.items():
    file_path = os.path.join(infile_prefix,file)
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    #fp.delete_file(identifier=identifier)
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )



