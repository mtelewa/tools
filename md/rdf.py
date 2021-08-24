#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import ase
import ase.io as io
import ase.geometry.analysis as ana
from ase import Atoms

def rdf(traj, Nevery=20, Nbins=100, maxDist=10):
    """
    Compute the radial distribution function (RDF) of a MD trajectory.
    Take the average of RDFs of <Nevery>th time frame.
    Parameters
    ----------
    Nevery : int
        Frequency of time frames to compute the RDF (the default is 50).
    Nbins : int
        Number of bins in radial direction i.e. resembles the RDF resolution.
        (the default is 100)
    maxDist : float
        maximal distance up to which the RDF is calculated, in Ångström.
        (the default is 5.)
    Returns
    -------
    numpy.ndarray
        Array containing distance (r) and RDF values (g(r))
    """

    # Van Der Waals radius
    sigma = 3.405       # Angstrom

    # io returns all atomic objects
    imageList = io.read(traj, index=f"::{Nevery}")

    # Count all atoms
    count=len(imageList[-1].get_masses())

    # Correct the atomic symbols
    for image in imageList:
        for atom in image:
            if atom.symbol=='Li':
                atom.symbol='Au'
            if atom.symbol=='He':   # CH2
                atom.symbol='Si'
            if atom.symbol=='H':    # CH3
                atom.symbol='P'

    # The atom counts
    atomic_symbols = np.array(imageList[-1].get_chemical_symbols())
    nCH2, nCH3, nAu = np.count_nonzero(atomic_symbols=='Si'), \
                      np.count_nonzero(atomic_symbols=='P'), \
                      np.count_nonzero(atomic_symbols=='Au')

    # Atom selection -----------------------------------------------

    if 'boundary' in sys.argv:
        # Get bounday atoms
        for idx, image in enumerate(imageList):
            # Atom objects with only the fluid atoms
            del image[[atom.index for atom in image if atom.symbol=='Au']]
            # maximum zpos in each image
            max_zpos = np.max(imageList[idx].get_positions()[:,2])
            # delete atoms below the boundary atoms
            del image[[atom.index for atom in image if
                            atom.position[2] < (max_zpos - 1.5*sigma) ]]

        # Check the boundary atoms count
        count = len(imageList[-1].get_masses())
        print(f'{count} atoms are in the last image')

    if 'solid' in sys.argv:
        for idx, image in enumerate(imageList):
            # Atom objects with only the solid atoms
            del image[[atom.index for atom in image if atom.symbol=='Si' or
                                                       atom.symbol=='P']]
            # delete surfL atoms
            del image[[atom.index for atom in image if atom.position[2] < 10.0 ]]

        # Check the boundary atoms count
        count = len(imageList[-1].get_masses())
        print(f'{count} atoms are in the last image')

    if 'liquid' in sys.argv:
        # Get bounday atoms
        for idx, image in enumerate(imageList):
            # Atom objects with only the fluid atoms
            del image[[atom.index for atom in image if atom.symbol=='Au']]
            # maximum zpos in each image
            max_zpos = np.max(imageList[idx].get_positions()[:,2])
            min_zpos = np.min(imageList[idx].get_positions()[:,2])
            # Delete atoms above and below the bulk
            # bulk is defined as the distance of 3*sigma from the wall (or maxima
            # of fluid positions)
            del image[[atom.index for atom in image if
                            atom.position[2] > (max_zpos - 3*sigma) or
                            atom.position[2] > (min_zpos + 3*sigma)]]

        # Check the boundary atoms count
        count = len(imageList[-1].get_masses())
        print(f'{count} atoms are in the last image')

    # RDF Calculation ----------------------------------------------

    # RDF for each image: Shape (len(images), Nbins)
    analyse = ana.Analysis(imageList[1:])
    print(f'{analyse.nImages} images will contribute to the RDF calculation.')

    rdf = np.array(analyse.get_rdf(maxDist, Nbins, return_dists=True))

    # Average RDF
    outRDF = np.mean(rdf, axis=0).T[:, ::-1]

    if 'boundary' in sys.argv:
        np.savetxt('RDF_boundary.txt',outRDF)
    if 'solid' in sys.argv:
        np.savetxt('RDF_solid.txt',outRDF)
    if 'liquid' in sys.argv:
        np.savetxt('RDF_liquid.txt',outRDF)

    return outRDF


if __name__ == "__main__":
    rdf(sys.argv[-1])
