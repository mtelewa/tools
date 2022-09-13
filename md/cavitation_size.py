#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pandas as pd
from ovito.data import DataTable
from ovito.io import import_file, export_file
from ovito.modifiers import ConstructSurfaceModifier, TimeSeriesModifier

# def progressbarstr(x, maxx, len=60):
#     xx = x/maxx
#     if xx != 1:
#       return '|'+int(xx*len)*'#'+((len-int(xx*len))*'-')+'| {:>3}% ' \
#              '({}/{})'.format(int(xx*100), x, maxx)
#     else:
#       return '|'+int(xx*len)*'#'+((len-int(xx*len))*'-')+'| {:>3}% ' \
#              '({}/{})\n'.format(int(xx*100), x, maxx)
#
# def progressbar(x, maxx, len=60):
#     sys.stdout.write('{}\r'.format(progressbarstr(x, maxx, len=len)))
#     sys.stdout.flush()

# Import the trajectory
pipeline = import_file("flow.nc", multiple_frames = True)

# Run the pipleine
data = pipeline.compute()

# Number of frames in the trajectory
nframes = pipeline.source.num_frames
# Number of atoms in the trajectory
# Select type(s)
# pipeline.modifiers.append(SelectTypeModifier(types = {1,2}))
natoms = data.particles.count
print(f'Number of frames: {nframes} and Number of atoms: {natoms}')

# LAMMPS dumping frequency
lammps_dump_every = np.int64(pipeline.compute(1).attributes['time'] - pipeline.compute(0).attributes['time'])
print(f'LAMMPS dumped every: {lammps_dump_every} steps.')

# Process every nth frams
every_nth = 10 #100
rprobe = 6 # Angstrom
frames_to_process = np.arange(0, nframes, every_nth)
print(f'Ovito will process the trajectory every: {every_nth}th frame, a total of {np.int64(nframes/every_nth)} frames')

# Construct Surface mesh Modifier: Use the alpha method to create the triangulated surface mesh
pipeline.modifiers.append(ConstructSurfaceModifier(
    method = ConstructSurfaceModifier.Method.AlphaShape,
    radius = rprobe,
    smoothing_level=30,
    identify_regions = True))
    #only_selected = True)

####################################################################
# 1. A Global attribute of the surface modifier -- The Surface Area
####################################################################


def compute_every_frame():
    """
    Method A: Compute the global attribute for every frame then output to a txt file
    """
    # Run the data pipleine every nth frame
    for i in frames_to_process:
        data = pipeline.compute(i)
        # Query the dataset -----------------------------------------
        # Print the global attributes of the dataset after the construct surface mesh modifier
        print(pd.DataFrame.from_dict(data.attributes, orient='index'))
        print('-----------------------------------------------------')
        # List of properties produced by the current data pipeline
        print(pd.DataFrame.from_dict(data.particles, orient='index'))
        print('-----------------------------------------------------')

        # Global meshed surface area: Includes only interfaces between empty and filled regions
        global_surface_area = data.attributes['ConstructSurfaceMesh.surface_area']
        print(f"Surface area: {global_surface_area}")

    # Export the global mesh surface area to a text file
    export_file(pipeline, "surface_area.txt",
        format = "txt/attr", # format = "netcdf/amber" if NetCDF format is needed
        columns = ["time", "ConstructSurfaceMesh.surface_area"],
        every_nth_frame = every_nth)
        #multiple_frames = True)


# Method B: Compute for all the frames then output a table
#------------------------------------------------------------

def compute_time_series():
    """
    Method B: Compute the global attribute for all the frames using the
    TimeSeriesModifier, then output the table
    """
    # Sample the variable computed by the data pipeline over all frames
    pipeline.modifiers.append(TimeSeriesModifier(
        operate_on = ('ConstructSurfaceMesh.surface_area'),
        sampling_frequency = every_nth))

    #Run the data pipleine
    data = pipeline.compute()

    # Export the table to a text file
    export_file(data.tables['time-series'], f"{os.getcwd()}/surface_area.txt", format = "txt/table")
    # OR print it to terminal
    # print(data.tables['time-series'].xy())


####################################################################
# 2. A Local attribute of the surface modifier -- The surface regions, where
# one or few of the empty region(s) is/are the bubble(s)
####################################################################

# The modifier method
def modify(frame, data):
    """
    Modifier to write the properties of the local regions
    """
    # Regions are part of the surface mesh object
    regions = data.surfaces['surface'].regions
    # Convert the property container into a regular DataTable
    table = DataTable(identifier='regions', title='Volumetric regions')
    table.y = table.create_property('Filled', data=regions['Filled'])
    table.create_property('Surface Area', data=regions['Surface Area'])
    table.create_property('Volume', data=regions['Volume'])
    data.objects.append(table)



def void_surface_area(frame, data):
    """
    A method to get the void surface area based on filtering surfaces meshes with:
    Empty regions (Filled==0),
    Volumes (V>Vc) where Vc is a cutoff volume
    Surface Areas (A>Ac) where Ac is a cutoff surface area
    """
    # Regions are part of the surface mesh object
    regions = data.surfaces['surface'].regions

    dict = {'regions':np.array(regions['Filled']), 'Areas':np.array(regions['Surface Area']),
             'Volumes':np.array(regions['Volume'])}

    indeces_to_remove = []

    # Mask the non-empty regions (with Filled=1)
    for idx,val in enumerate(dict['regions']):
        if val==1:
            indeces_to_remove.append(idx)
    # print(indeces_to_remove)

    # Mask the empty regions with small areas
    for idx,val in enumerate(dict['Areas']):
        if val<100:
            indeces_to_remove.append(idx)
    # print(indeces_to_remove)

    # Mask the empty regions with small areas and small volumes
    for idx,val in enumerate(dict['Volumes']):
        if val<100:
            indeces_to_remove.append(idx)
    # print(indeces_to_remove)

    R=np.delete(dict['regions'], indeces_to_remove)
    A=np.delete(dict['Areas'], indeces_to_remove)
    V=np.delete(dict['Volumes'], indeces_to_remove)

    dict = {'regions':R, 'Areas':A, 'Volumes':V}

    # Total surface area
    area = dict['Areas']
    # The void in thin span-wise-direction (y-direction) boxes is more like a cylinder.
    # The Cylinder's height is the cell size in y-direction
    height = data.cell[:,1][1]
    # From the area and height, we get the cavity radius
    radius = (np.sqrt(height**2+(2*area/np.pi))-height)/2.

    return {'area':area,'radius':radius}



time_list, radius_list = [], []
# # Run the data pipleine
for i in frames_to_process:
    # progressbar(i,len(frames_to_process))
    data = pipeline.compute(i)
    r = void_surface_area(i,data)['radius']
    # area = void_surface_area(i,data)['area']
    time_list.append(i*lammps_dump_every)
    if r.size==0:
        radius_list.append(0)
    else:
        radius_list.append(np.max(r))

# TODO: Get the simulation nucleation rate
J_sim = tau*vol

np.savetxt('radius.txt', np.c_[time_list, radius_list],  delimiter=' ',\
                                header='Time (fs)              Radius (A)')


# Insert user-defined modifier function into the data pipeline.
# pipeline.modifiers.append(modify)
#
