import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

def region(coords_to_mask, masking_coords, lower, upper, ylength=1, length=1):
    """
    Set a mask for a coordinates array based on the limits of a region

    Parameters
    ----------
    coords_to_mask : arr
        Coordinates to mask
    masking_coords : arr
        Coordinates for the boolean
    lower: int
        Lower limit of the region
    upper: int
        Upper limit of the region
    ylength: int
        The length of the region in the y-direction
    length: int
        The third length of the region (3-dim region)

    Returns
    -------
    interval : int
        The first length of the region
    mask : arr
        Boolean of Coordinates mask
    data:  arr
        Masked coordinates array
    vol:  int
        Volume of the region
    N: int
        Number of atoms in the region
    """
    mask_hi = np.greater_equal(masking_coords,  lower)
    mask_lo = np.less(masking_coords,  upper)
    interval = upper - lower
    mask = np.logical_and(mask_lo, mask_hi)
    data = coords_to_mask * mask
    vol = interval * ylength * length
    N = np.sum(mask, axis=1)

    return {'interval':interval, 'mask':mask , 'data':data, 'vol':vol, 'count':N}

def extrema(coords):
    """
    Function to get the global minimum of a coordinates array
    in a parallelized script. The local coordinates are handled by
    each rank and globalized.

    Parameters
    ----------
    coords : array of coordinates to get the minimum of

    Returns
    -------
    local_min : arr -- the minimum in each rank in each timestep
    global_min : scalar -- the minimum across all ranks in all timesteps of the slice
    global_min_avg : scalar -- average min of all timesteps in the slice
    """
    local_min = np.min(coords, axis=1)
    global_min = comm.allreduce(np.min(local_min), op=MPI.MIN)
    global_min_avg = np.mean(comm.allgather(np.mean(local_min)))

    local_max = np.max(coords, axis=1)
    global_max = comm.allreduce(np.max(local_max), op=MPI.MAX)
    global_max_avg = np.mean(comm.allgather(np.mean(local_max)))

    return {'local_min':local_min, 'global_min':global_min ,
            'local_max':local_max, 'global_max':global_max }

def cnonzero_min(coords):
    """
    Function to get the global non-zero minimum of a coordinates array
    in a parallelized script. The local coordinates are handled by
    each rank and globalized.

    Parameters
    ----------
    coords : array of coordinates to get the minimum of

    Returns
    -------
    local_min : arr -- the minimum in each rank in each timestep
    global_min : scalar -- the minimum across all ranks in all timesteps
    global_min_avg : scalar -- average min of all timesteps
    """
    # Mask zero elements
    coords_nonzero = np.ma.masked_equal(coords, 0.0, copy=False)
    local_min_nonzero = coords_nonzero.min(axis=1)
    global_min_nozero = comm.allreduce(np.min(local_min_nonzero), op=MPI.MIN)
    #global_min_avg_nozero = np.mean(np.array(comm.allgather(np.mean(local_min_nonzero))))

    return {'local_min':local_min_nonzero, 'global_min':global_min_nozero}


def bounds(boundsX, boundsY, boundsZ):

    xx, yy, zz = np.meshgrid(boundsX, boundsY, boundsZ)

    xx = np.transpose(xx, (1, 0, 2))
    yy = np.transpose(yy, (1, 0, 2))
    zz = np.transpose(zz, (1, 0, 2))

    dx = xx[1:, 1:, 1:] - xx[:-1, :-1, :-1]
    dy = yy[1:, 1:, 1:] - yy[:-1, :-1, :-1]
    dz = zz[1:, 1:, 1:] - zz[:-1, :-1, :-1]

    vol = dx * dy * dz

    return xx, yy, zz, vol
