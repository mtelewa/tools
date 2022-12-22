#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import ase
import ase.io as io
import ase.geometry.analysis as analysis
from ase import Atoms
from progressbar import progressbar

def rdf(traj, start, stop, Nevery, Nbins, cutoff, region):
    """
    Compute the radial distribution function (RDF) of a MD trajectory.
    Take the average of RDFs of <Nevery>th time frame.
    Parameters
    ----------
    Nevery : int
        Frequency of time frames to compute the RDF.
    Nbins : int
        Number of bins in radial direction i.e. resembles the RDF resolution.
    cutoff : float
        maximal distance up to which the RDF is calculated, in Ångström.
    Returns
    -------
    numpy.ndarray
        Array containing distance (r) and RDF values (g(r))
    """

    # Van Der Waals radius
    sigma = 3.405       # Ångström

    # io returns all atomic objects
    if stop == -1: stop = 510
    imageList = io.read(traj, index=f"{start}:{stop}:{Nevery}")

    # Count all atoms
    count=len(imageList[-1].get_masses())

    # Correct the atomic symbols
    for image in imageList:
        for atom in image:
            if atom.symbol=='Li': atom.symbol='Au'
            if atom.symbol=='He': atom.symbol='Si'  # CH2
            if atom.symbol=='H': atom.symbol='P'   # CH3

    # The atom counts
    atomic_symbols = np.array(imageList[-1].get_chemical_symbols())
    nCH2, nCH3, nAu = np.count_nonzero(atomic_symbols=='Si'), \
                      np.count_nonzero(atomic_symbols=='P'), \
                      np.count_nonzero(atomic_symbols=='Au')

    # Atom selection -----------------------------------------------

    if region == 'boundary':
        # Get bounday atoms
        for idx, image in enumerate(imageList):
            progressbar(idx, len(imageList))
            # Atom objects with only the fluid atoms
            del image[[atom.index for atom in image if atom.symbol=='Au']]
            # maximum zpos in each image
            max_zpos = np.max(imageList[idx].get_positions()[:,2])
            # delete atoms below the boundary atoms
            del image[[atom.index for atom in image if
                            atom.position[2] < (max_zpos - 0.5*sigma) ]]

        # Check the boundary atoms count
        count = len(imageList[-1].get_masses())
        print(f'{count} atoms are in the last image')

    if region == 'solid':
        for idx, image in enumerate(imageList):
            progressbar(idx, len(imageList))
            # Atom objects with only the solid atoms
            del image[[atom.index for atom in image if atom.symbol=='Si' or atom.symbol=='P']]
            # delete surfL atoms
            del image[[atom.index for atom in image if atom.position[2] < 10.0 ]]

        # Check the boundary atoms count
        count = len(imageList[-1].get_masses())
        print(f'{count} boundary atoms are in the last image')

    if region == 'bulk':
        # Get bounday atoms
        for idx, image in enumerate(imageList):
            print('Processing Images..')
            progressbar(idx, len(imageList))
            # Atom objects with only the CH3 atoms
            del image[[atom.index for atom in image if atom.symbol=='Au']]
            del image[[atom.index for atom in image if atom.symbol=='Si']]
            # maximum zpos in each image
            max_zpos = np.max(imageList[idx].get_positions()[:,2])
            min_zpos = np.min(imageList[idx].get_positions()[:,2])
            # Delete atoms above and below the bulk.
            # Bulk is defined as the distance of 3*sigma from the wall (or maxima
            # of fluid positions)
            del image[[atom.index for atom in image if
                            atom.position[2] > (max_zpos - 3*sigma) or
                            atom.position[2] > (min_zpos + 3*sigma)]]

        # Check the bulk atoms count
        count = len(imageList[-1].get_masses())
        print(f'{count} bulk atoms are in the last image')

    # RDF Calculation ----------------------------------------------

    # RDF for each image: Shape (len(images), Nbins)
    analyse = analysis.Analysis(imageList)
    print(f'{analyse.nImages} images will contribute to the RDF calculation.')

    rdf = np.array(analyse.get_rdf(cutoff, Nbins, return_dists=True))

    # Average RDF
    outRDF = np.mean(rdf, axis=0).T[:, ::-1]

    return outRDF
