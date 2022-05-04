#!/usr/bin/env python

import numpy as np
from scipy.spatial import Voronoi, ConvexHull, Delaunay


def voronoi_volumes(points):
    """
    Computes the Voronoi volumes by performing Vornoi tessellation on the points (coordinates) array.
    The volumes are computed from the ConvexHull method ---
        So large because of the unbounded cells.
    parameters
    ----------
    points: array of shape (Natoms, ndim)

    returns
    -------
    vol: array of shape (Natoms,volumes)

    returns:
    -------
    vor: vornoi diagram
    vol: volume of each cell
    """
    vor = Voronoi(points)
    vol = np.zeros(vor.npoints)
    # print(v.point_region) # Index of the Voronoi region for each input point.
    #If qhull option “Qc” was not specified, the list will contain -1 for points
    #that are not associated with a Voronoi region.

    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num] #Indices of the Voronoi vertices forming each Voronoi region.
        if -1 in indices: # -1 indicates vertex outside the Voronoi diagram. (some regions can be open)
            vol[i] = np.inf
        else:   #coordinates of the voronoi verices
            vol[i] = ConvexHull(vor.vertices[indices]).volume

    return {'vor':vor, 'vol':vol}



def tetrahedron_volume(a, b, c, d):
    """
    Calculates the volume of a tetrahedron, given vertices a,b,c and d (triplets)
    """
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6
    #Specifies the subscripts for summation as comma separated list of subscript labels



def delaunay_volumes(points):
    """
    Computes the volumes by performing Delaunay tessellation on the points (coordinates) array.
    Brute force method where a cutoff is defined for a volume of a Voronoi cell.
    This volume was defined based on convergence to LAMMPS calculated volumes.
    parameters
    ----------
    points: Atomic coordinates, array of shape (Natoms, ndim)

    returns
    -------
    vol: Total volume of the tetrahedrons from the triangulation, float
    """
    d = Delaunay(points)
    # print(d.simplices) #Indices of the points forming the simplices in the triangulation.
    tets = d.points[d.simplices]

    volumes = tetrahedron_volume(tets[:, 0], tets[:, 1],
                                 tets[:, 2], tets[:, 3])

    volumes = np.ma.masked_greater(volumes, 1.8e2)
    vol = np.ma.masked_invalid(volumes).sum()

    return vol


def tetravol(a,b,c,d):
    '''Calculates the volume of a tetrahedron, given vertices a,b,c and d (triplets)'''
    tetravol=abs(np.dot((a-d),np.cross((b-d),(c-d))))/6
    return tetravol


def vor_to_del(vor,p):
     """
     Switches the Voronoi tessellation to Delaunay triangulation and
     calculates the volume of 3d Voronoi cell based on point p. Voronoi diagram is passed in vor.

     Too slow !!

     parameters:
     -----------
     Vor: voronoi diagram
     p : a single point in the vornoi diagram

     returns:
     --------
     vol: volume of the triangle

     """
     dpoints=[]
     vol=0

     # vor.point_region: Indices of the Voronoi region for each input point
     # vor.regions: Indices of the Voronoi vertices forming each Voronoi region

     # The Delaunay points are the Voronoi vertices
     for v in vor.regions[vor.point_region[p]]:
         dpoints.append(list(vor.vertices[v]))      # Points for Delaunay triangulation

     tri=Delaunay(np.array(dpoints))                # Delaunay triangulation
     for simplex in tri.simplices:
         vol+=tetravol(np.array(dpoints[simplex[0]]),np.array(dpoints[simplex[1]]),
                       np.array(dpoints[simplex[2]]),np.array(dpoints[simplex[3]]))
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
