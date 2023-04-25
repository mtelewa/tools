#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys, os
import numpy as np
from rdf import rdf

def get_parser():
    parser = argparse.ArgumentParser(
    description='Plot quantities post-processed from a netcdf trajectory.')

    #Positional arguments
    #--------------------
    parser.add_argument('start', metavar='start', action='store', type=int,
                    help='Start from this frame')
    parser.add_argument('stop', metavar='stop', action='store', type=int,
                    help='Discard after this frame. If till the last frame, type -1.')
    parser.add_argument('nevery', metavar='nevery', action='store', type=int,
                    help='ASE will process every nth frame')
    parser.add_argument('nbins', metavar='nbins', action='store', type=int,
                    help='Number of bins in radial direction i.e. resembles the RDF resolution.')
    parser.add_argument('cutoff', metavar='cutoff', action='store', type=float,
                    help='Cutoff for the RDF.')
    parser.add_argument('region', metavar='region', action='store', type=str,
                    help='Region for computing the RDF (boundary/bulk/solid).')
    # Get the datasets
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-ds")):
            # pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str,
                                help='datasets for the analysis')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    datasets = []
    for key, value in vars(args).items():
        if key.startswith('ds'):
            datasets.append(os.path.abspath(value))

    trajactories = []
    for k in datasets:
        for root, dirs, files in os.walk(k):
            for i in files:
                if i == 'flow.nc' or i == 'equilib.nc' or i == 'nvt.nc':
                    trajactories.append(os.path.join(root, i))

    for i in range(len(trajactories)):

        outRDF = rdf(trajactories[i], args.start, args.stop, args.nevery, args.nbins, args.cutoff, args.region)

        if 'boundary' in sys.argv:
            np.savetxt('RDF_boundary.txt', outRDF, delimiter=' ', header='Distance (A)        RDF')
        if 'solid' in sys.argv:
            np.savetxt('RDF_solid.txt', outRDF, delimiter=' ', header='Distance (A)        RDF')
        if 'bulk' in sys.argv:
            np.savetxt('RDF_bulk.txt', outRDF, delimiter=' ', header='Distance (A)        RDF')
