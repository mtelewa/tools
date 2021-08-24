#!/usr/bin/env python

import numpy as np
from scipy.spatial import Voronoi, ConvexHull, Delaunay

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    # print(v.point_region) # v.point_region: Index of the Voronoi region for each input point.
    #If qhull option “Qc” was not specified, the list will contain -1 for points
    #that are not associated with a Voronoi region.

    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num] #Indices of the Voronoi vertices forming each Voronoi region.
        # -1 indicates vertex outside the Voronoi diagram. (some regions can be opened)

        if -1 in indices:
            vol[i] = np.inf
        else:   #coordinates of the voronoi verices
            vol[i] = ConvexHull(v.vertices[indices]).volume

    return vol


def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
    #Specifies the subscripts for summation as comma separated list of subscript labels.

def delaunay_volumes(points):
    # print(points)
    v = Delaunay(points)
    vol = np.zeros(v.npoints)
    # print(v.simplices.shape) #Indices of the points forming the simplices in the triangulation.
    tets = v.points[v.simplices]

    vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                    tets[:, 2], tets[:, 3]))

    return vol


# Example in post-proc script
    # # Delaunay volumes
    # bulk_xcoords = fluid_coords[:,:,0]*bulk_region
    # bulk_ycoords = fluid_coords[:,:,1]*bulk_region
    # bulk_zcoords = fluid_coords[:,:,2]*bulk_region
    #
    # bulk_coords = np.zeros((tSample,Nf,3))
    # bulk_coords[:,:,0],bulk_coords[:,:,1],bulk_coords[:,:,2]= \
    #         bulk_xcoords,bulk_ycoords,bulk_zcoords
    #
    # totVi = []
    # for i in range(tSample):
    #     totVi.append(tes.delaunay_volumes(bulk_coords[i]))        # A3
    #
    # totVi=np.asarray(totVi)
