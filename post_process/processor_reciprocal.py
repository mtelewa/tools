#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging
import numpy as np
import scipy.constants as sci
import time as timer
import netCDF4
from mpi4py import MPI
from operator import itemgetter
import utils

# Warnings Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# np.set_printoptions(threshold=sys.maxsize)

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Conversions
fs_to_ns = 1e-6
A_per_fs_to_m_per_s = 1e5
atmA3_to_kcal = 0.02388*1e-27

t0 = timer.time()

class TrajtoGrid:
    """
    Molecular domain decomposition into spatial bins
    """

    def __init__(self, data, start, end, nx, ny, nz, mf, A_per_molecule, fluid, TW_interface):
        """
        Parameters:
        -----------
        data: str, NetCDF trajectory file
        start, end: Start and end of the sampled time in each processor
        nx, ny, Nnz: int, Number of discrete wave vectors in the x-, y- and z-direcions
        mf: float, molecular mass of fluid molecule
        A_per_molecule: int, no. of atoms per molecule
        fluid: str, fluid material
        TW_interface: int, boolean to define the vibrating portion of atoms in the upper wall
                Default = 1: thermostat applied on the wall layers in contact with the fluid
        """

        self.data = data
        self.start, self.end = start, end
        self.chunksize = self.end - self.start
        self.nx, self.ny, self.nz = comm.bcast(nx, root=0), comm.bcast(ny, root=0), comm.bcast(nz, root=0)
        self.mf, self.A_per_molecule = comm.bcast(mf, root=0), comm.bcast(A_per_molecule, root=0)
        self.fluid = comm.bcast(fluid, root=0)
        self.TW_interface = comm.bcast(TW_interface, root=0)

    def get_dimensions(self):
        """
        Get the domain dimensions in the first timestep from "cell_lengths", shape (time, 3)
        Returns
        ----------
        cell_lengths: (3,) array
        """

        cell_lengths = self.data.variables["cell_lengths"]
        cell_lengths = np.array(cell_lengths[0, :]).astype(np.float32)

        return cell_lengths


    def get_indices(self):
        """
        Define the fluid and solid groups in the first timestep from "type", shape (time, Natoms)
        Returns
        ----------
        fluid_idx: (Natoms,) array, indeces of fluid atoms
        solid_idx: (Natoms,) array, indeces of solid atoms
        solid_start: int, first solid atom index
        Nf: int, no. of fluid atoms
        Nm: int, no. of fluid molecules
        Ns: int, no. of solid atoms
        """

        type = self.data.variables["type"]
        type = np.array(type[0, :]).astype(np.float32)

        fluid_idx, solid_idx = [],[]    # Should be of shapes: (Nf,) and (Ns,)

        # Lennard-Jones
        if np.max(type)==2:
            fluid_idx.append(np.where(type == 1))
            fluid_idx = fluid_idx[0][0]
            solid_idx.append(np.where(type == 2))
            solid_idx = solid_idx[0][0]
        # Hydrocarbons
        if np.max(type)==3:
            fluid_idx.append(np.where([type == 1, type == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where(type == 3))
            solid_idx = solid_idx[0][0]
        # Hydrocarbons with sticking and slipping walls
        if np.max(type)==4 and self.fluid=='pentane':
            fluid_idx.append(np.where([type == 1, type == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where([type == 3, type == 4]))
            solid_idx = solid_idx[0][1]

        if np.max(type)==4 and self.fluid=='squalane':
            fluid_idx.append(np.where([type == 1, type == 2, type == 3]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where([type == 4]))
            solid_idx = solid_idx[0][1]

        solid_start = np.min(solid_idx)

        Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/self.A_per_molecule, \
                                np.max(solid_idx)-np.max(fluid_idx)

        if rank == 0:
            logger.info('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,int(Nm),Ns))

        return fluid_idx, solid_idx, solid_start, Nf, Nm


    def get_chunks(self, fluid_Zstart, fluid_Zend, solid_Zstart, solid_Zend):
        """
        Partitions the box into regions (solid and fluid) as well as chunks in
        those regions
        Parameters (normalized by gap height)
        -------------------------------------
        fluid_Zstart: float, where the fluid region of interest starts
        fluid_Zend: float, where the fluid region of interest ends
        solid_Zstart: float, where the fluid region of interest starts
        solid_Zend: float, where the fluid region of interest ends
        Returns
        ----------
        Thermodynamic quantities (arrays) with different shapes. The 2 shapes are
        (time,) time series of the quantity
        (time, nx, ny) time series of the chunks along the x and the y-directions.
        """

        fluid_idx, solid_idx, solid_start, Nf, Nm = self.get_indices()
        Lx, Ly, Lz = self.get_dimensions()

        # Discrete Wavevectors
        nx = np.linspace(1, self.nx, self.nx, endpoint=True)
        ny = np.linspace(1, self.ny, self.ny, endpoint=True)
        nz = np.linspace(1, self.nz, self.nz, endpoint=True)

        # Wavevectors
        kx = 2. * np.pi * nx / Lx
        ky = 2. * np.pi * ny / Ly

        # The unaveraged quantity is that which is dumped by LAMMPS every N timesteps
        # i.e. it is a snapshot of the system at this timestep.
        # As opposed to averaged quantities which are the average of a few previous timesteps
        # Shape: (time, idx, dimesnion)
        coords_data_unavgd = self.data.variables["coordinates"]   # Used for ACF calculation

        # Cartesian Coordinates ---------------------------------------------
        coords = np.array(coords_data_unavgd[self.start:self.end]).astype(np.float32)
        # Shape: (time, Nf)
        fluid_xcoords, fluid_ycoords, fluid_zcoords = coords[:, fluid_idx, 0], \
                                                      coords[:, fluid_idx, 1], \
                                                      coords[:, fluid_idx, 2]
        # Shape: (time, Ns)
        solid_xcoords, solid_ycoords, solid_zcoords = coords[:, solid_start:, 0], \
                                                      coords[:, solid_start:, 1], \
                                                      coords[:, solid_start:, 2]

        # Instanteneous extrema (array Needed for the gap height at each timestep)
        fluid_begin, fluid_end = utils.extrema(fluid_zcoords)['local_min'], \
                                 utils.extrema(fluid_zcoords)['local_max']

        # TODO: Need to define the diverged and converged regions properly
        # Narrow range because of outliers in the fluid coordinates
        beginConvergeX, endConvergeX, beginDivergeX, endDivergeX = 0.1, 0.2, 0.7, 0.8
        fluid_zcoords_conv = utils.region(fluid_zcoords, fluid_xcoords, beginConvergeX*Lx, endConvergeX*Lx)['data']
        fluid_begin_conv = utils.cnonzero_min(fluid_zcoords_conv)['local_min']
        fluid_end_conv = utils.extrema(fluid_zcoords_conv)['local_max']

        fluid_zcoords_div = utils.region(fluid_zcoords, fluid_xcoords, beginDivergeX*Lx, endDivergeX*Lx)['data']
        fluid_begin_div = utils.cnonzero_min(fluid_zcoords_div)['local_min']
        fluid_end_div = utils.extrema(fluid_zcoords_div)['local_max']

        # Shape: int
        avg_fluid_begin_div = np.mean(comm.allgather(np.mean(fluid_begin_div)))
        avg_fluid_end_div = np.mean(comm.allgather(np.mean(fluid_end_div)))

        # Define the upper surface and lower surface regions
        # To avoid problems with logical-and later
        solid_xcoords[solid_xcoords==0] = 1e-5
        surfU = np.ma.masked_greater(solid_zcoords, utils.extrema(fluid_zcoords)['global_max']/2.)
        surfL = np.ma.masked_less(solid_zcoords, utils.extrema(fluid_zcoords)['global_max']/2.)
        # Indices of the the lower and upper, Shape: (time, Nsurf)
        surfU_indices = np.where(surfU[0].mask)[0]
        surfL_indices = np.where(surfL[0].mask)[0]

        # Shape: (time, NsurfU)
        surfU_xcoords, surfU_zcoords = solid_xcoords[:,surfU_indices], solid_zcoords[:,surfU_indices]
        # Shape: (time, NsurfL)
        surfL_xcoords, surfL_zcoords = solid_xcoords[:,surfL_indices], solid_zcoords[:,surfL_indices]

        # Check if surfU and surfL are not equal
        N_surfU, N_surfL = len(surfU_indices), len(surfL_indices)
        if N_surfU != N_surfL and rank == 0:
            logger.warning("No. of surfU atoms != No. of surfL atoms")

        # Shape: (time,)
        surfU_begin, surfL_end = utils.cnonzero_min(surfU_zcoords)['local_min'], \
                                 utils.extrema(surfL_zcoords)['local_max']

        surfU_end, surfL_begin = utils.extrema(surfU_zcoords)['local_max'], \
                                 utils.extrema(surfL_zcoords)['local_min']

        # Average of the timesteps in this time slice for the avg. height
        # Shape: int
        avg_surfU_end, avg_surfL_begin = np.mean(comm.allgather(np.mean(surfU_end))), \
                                         np.mean(comm.allgather(np.mean(surfL_begin)))

        # The converged and diverged regions surfU and surfL: conv and div
        surfU_zcoords_conv = utils.region(surfU_zcoords, surfU_xcoords, beginConvergeX*Lx, endConvergeX*Lx)['data']
        surfU_begin_conv = utils.cnonzero_min(surfU_zcoords_conv)['local_min']
        surfU_zcoords_div = utils.region(surfU_zcoords, surfU_xcoords, beginDivergeX*Lx, endDivergeX*Lx)['data']
        surfU_begin_div = utils.cnonzero_min(surfU_zcoords_div)['local_min']

        avg_surfU_begin_div = np.mean(comm.allgather(np.mean(surfU_begin_div)))

        # For vibrating walls
        if self.TW_interface == 1: # Thermostat is applied on the walls directly at the interface (half of the wall is vibrating)
            surfU_vib_end = utils.cnonzero_min(surfU_zcoords)['local_min'] + \
                                0.5 * utils.extrema(surfL_zcoords)['local_max'] - 1
        else: # Thermostat is applied on the walls away from the interface (2/3 of the wall is vibrating)
            surfU_vib_end = utils.cnonzero_min(surfU_zcoords)['local_min'] + \
                                0.667 * utils.extrema(surfL_zcoords)['local_max'] - 1
        avg_surfU_vib_end = np.mean(comm.allgather(np.mean(surfU_vib_end)))

        surfU_vib = np.ma.masked_less(surfU_zcoords, avg_surfU_vib_end)
        surfU_vib_indices = np.where(surfU_vib[0].mask)[0]
        # Positions
        surfU_vib_xcoords, surfU_vib_zcoords = solid_xcoords[:,surfU_vib_indices], \
        solid_zcoords[:,surfU_vib_indices]

        surfL_zcoords_conv = utils.region(surfL_zcoords, surfL_xcoords, beginConvergeX*Lx, endConvergeX*Lx)['data']
        surfL_end_conv = utils.extrema(surfL_zcoords_conv)['local_max']
        surfL_zcoords_div = utils.region(surfL_zcoords, surfL_xcoords, beginDivergeX*Lx, endDivergeX*Lx)['data']
        surfL_end_div = utils.extrema(surfL_zcoords_div)['local_max']

        avg_surfL_end_div = np.mean(comm.allgather(np.mean(surfL_end_div)))

        # Gap heights and COM (in Z-direction) at each timestep
        # Shape: (time,)
        gap_height = (fluid_end - fluid_begin + surfU_begin - surfL_end) / 2.
        gap_height_conv = (fluid_end_conv - fluid_begin_conv + surfU_begin_conv - surfL_end_conv) / 2.
        gap_height_div = (fluid_end_div - fluid_begin_div + surfU_begin_div - surfL_end_div) / 2.
        gap_heights = [gap_height, gap_height_conv, gap_height_div]
        comZ = (surfU_begin - surfL_end) / 2.

        # Shape: int
        avg_gap_height = np.mean(comm.allgather(np.mean(gap_height)))
        avg_gap_height_conv = np.mean(comm.allgather(np.mean(gap_height_conv)))
        avg_gap_height_div = np.mean(comm.allgather(np.mean(gap_height_div)))

        # Update the box domain dimensions
        Lz = avg_surfU_end - avg_surfL_begin
        cell_lengths_updated = [Lx, Ly, Lz]

        # Fluid minimum positions for the grid
        fluid_min = [utils.extrema(fluid_xcoords)['global_min'],
                     utils.extrema(fluid_ycoords)['global_min'],
                     (avg_surfL_end_div + avg_fluid_begin_div) /2.]

        fluid_max = [utils.extrema(fluid_xcoords)['global_max'],
                     utils.extrema(fluid_ycoords)['global_max'],
                     (avg_surfU_begin_div + avg_fluid_end_div) /2.]
        # Fluid domain dimensions
        fluid_lengths = [fluid_max[0]- fluid_min[0], fluid_max[1]- fluid_min[1], avg_gap_height_div]

        # Wave vectors in the z-direction
        kz = 2. * np.pi * nz / avg_gap_height_div

        #-------------------------------------------------------------------------
        # The Grid -------------------------------------------------------------
        # ----------------------------------------------------------------------
        # Mask the layer of interest for structure factor calculation
        maskx_layer = utils.region(fluid_xcoords, fluid_xcoords, 0, Lx)['mask']
        masky_layer = utils.region(fluid_ycoords, fluid_ycoords, 0, Ly)['mask']
        maskz_layer = utils.region(fluid_zcoords, fluid_zcoords, fluid_Zstart, fluid_Zend)['mask']
        mask_xy_layer = np.logical_and(maskx_layer, masky_layer)
        mask_layer = np.logical_and(mask_xy_layer, maskz_layer)

        maskx_layer_solid = utils.region(solid_xcoords, solid_xcoords, 0, Lx)['mask']
        masky_layer_solid = utils.region(solid_ycoords, solid_ycoords, 0, Ly)['mask']
        maskz_layer_solid = utils.region(solid_zcoords, solid_zcoords, solid_Zstart, solid_Zend)['mask']
        mask_xy_layer_solid = np.logical_and(maskx_layer_solid, masky_layer_solid)
        mask_layer_solid = np.logical_and(mask_xy_layer_solid, maskz_layer_solid)

        # Count particles in the fluid and solid cells
        N_layer_mask = np.sum(mask_layer, axis=1)
        N_layer_mask_solid = np.sum(mask_layer_solid, axis=1)

        # ----------------------------------------------------------------------
        # Cell Partition: arrays of shape (time, nx, ny)------------------------
        # ----------------------------------------------------------------------

        # initialize buffers to store the count 'N' and 'data_ch' of each chunk
        rho_k = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)
        sf = np.zeros_like(rho_k)
        sf_solid = np.zeros_like(rho_k)

        # Structure factor
        for i in range(len(kx)):
            if rank==0: print(i)
            for k in range(len(ky)):
                # Fourier components of the density
                rho_k[:, i, k] = np.sum( np.exp(1.j * (kx[i] * fluid_xcoords * mask_layer +
                                    ky[k] * fluid_ycoords * mask_layer)))
                sf[:, i, k] =  np.abs((np.sum( np.exp(1.j * (kx[i] * fluid_xcoords * mask_layer +
                                    ky[k] * fluid_ycoords * mask_layer)), axis=1)))**2 / N_layer_mask
                sf_solid[:, i, k] =  np.abs((np.sum( np.exp(1.j * (kx[i] * solid_xcoords * mask_layer_solid +
                                    ky[k] * solid_ycoords * mask_layer_solid)), axis=1)))**2 / N_layer_mask_solid

        return {'cell_lengths': cell_lengths_updated,
                'kx': kx,
                'ky': ky,
                'kz': kz,
                'gap_heights': gap_heights,
                'com': comZ,
                'rho_k': rho_k,
                'sf': sf,
                'sf_solid': sf_solid,
                'Nf': Nf,
                'Nm': Nm}
