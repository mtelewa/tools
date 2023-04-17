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
import tessellation

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

    def __init__(self, data, start, end, nx, ny, mf, A_per_molecule, fluid, nz=1, tessellate=0, TW_interface=1):

        self.data = data
        self.start, self.end = start, end
        self.chunksize = self.end - self.start
        self.nx, self.ny, self.nz = comm.bcast(nx, root=0), comm.bcast(ny, root=0), comm.bcast(nz, root=0)
        self.mf, self.A_per_molecule = mf, A_per_molecule
        self.ms = 196.96 # gold atom
        self.tessellate = tessellate
        self.TW_interface = TW_interface
        self.fluid = fluid

    def get_dimensions(self):
        # Box dimensions
        cell_lengths = self.data.variables["cell_lengths"]
        cell_lengths = np.array(cell_lengths[0, :]).astype(np.float32)

        return cell_lengths


    def get_indices(self):
        # Particle Types ------------------------------
        type = self.data.variables["type"]
        type = np.array(type[0, :]).astype(np.float32)

        fluid_idx, solid_idx = [],[]

        # Only Bulk
        # Lennard-Jones
        if np.max(type)==1:
            walls = 0
            fluid_idx.append(np.where(type == 1))
            fluid_idx = fluid_idx[0][0]
        # Hydrocarbons
        if np.max(type)==2 and self.A_per_molecule>1:
            walls = 0
            fluid_idx.append(np.where([type == 1, type == 2]))
            fluid_idx = fluid_idx[0][1]

        # With Walls
        # Lennard-Jones
        if np.max(type)==2 and self.A_per_molecule==1:
            walls = 1
            fluid_idx.append(np.where(type == 1))
            fluid_idx = fluid_idx[0][0]
            solid_idx.append(np.where(type == 2))
            solid_idx = solid_idx[0][0]
        # Hydrocarbons
        if np.max(type)==3:
            walls = 1
            fluid_idx.append(np.where([type == 1, type == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where(type == 3))
            solid_idx = solid_idx[0][0]
        # Hydrocarbons with sticking and slipping walls
        if np.max(type)==4 and self.fluid=='pentane':
            walls = 1
            fluid_idx.append(np.where([type == 1, type == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where([type == 3, type == 4]))
            solid_idx = solid_idx[0][1]

        if np.max(type)==4 and self.fluid=='squalane':
            walls = 1
            fluid_idx.append(np.where([type == 1, type == 2, type == 3]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where([type == 4]))
            solid_idx = solid_idx[0][1]

        if walls == 1:
            Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/self.A_per_molecule, \
                                    np.max(solid_idx)-np.max(fluid_idx)

            solid_start = np.min(solid_idx)
        else:
            Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/self.A_per_molecule, 0
            solid_start = None


        if rank == 0:
            logger.info('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,int(Nm),Ns))

        return fluid_idx, solid_start, Nf, Nm


    def get_chunks(self, fluid_Zstart, fluid_Zend, solid_Zstart, solid_Zend):
        """
        Partitions the box into regions (solid and fluid) as well as chunks in
        those regions
        Parameters
        ----------
        fluid_Zstart: float
                Value of Lx, where the fluid region of interest starts
        fluid_Zend: float
                Value of Lx, where the fluid region of interest ends
        solid_Zstart: float
                Value of Lx, where the fluid region of interest starts
        solid_Zend: float
                Value of Lx, where the fluid region of interest ends
        Returns
        ----------
        Thermodynamic quantities (arrays) with different shapes. The 4 shapes are
        (time,)
            time series of the quantity
        (time,nx,ny)
            time series of the chunks along the x and the y-directions.
        """

        fluid_idx, solid_start, Nf, Nm = self.get_indices()
        Lx, Ly, Lz = self.get_dimensions()

        # Discrete Wavevectors
        nx = np.linspace(1, self.nx, self.nx, endpoint=True)
        ny = np.linspace(1, self.ny, self.ny, endpoint=True)
        nz = np.linspace(1, self.nz, self.nz, endpoint=True)

        # Wavevectors
        kx = 2. * np.pi * nx / Lx
        ky = 2. * np.pi * ny / Ly

        # Array dimensions: (time, idx, dimesnion)
        try:
            coords_data = self.data.variables["f_position"]
        except KeyError:
            coords_data = self.data.variables["coordinates"]
        # The unaveraged quantity is that which is dumped by LAMMPS every N timesteps
        # i.e. it is a snapshot of the system at this timestep.
        # As opposed to averaged quantities which are the average of a few previous timesteps
        coords_data_unavgd = self.data.variables["coordinates"]   # Used for ACF calculation

        # for i in range(self.Nslices):
        # Cartesian Coordinates ---------------------------------------------
        coords = np.array(coords_data[self.start:self.end]).astype(np.float32)

        fluid_xcoords,fluid_ycoords,fluid_zcoords = coords[:, fluid_idx, 0], \
                                                    coords[:, fluid_idx, 1], \
                                                    coords[:, fluid_idx, 2]

        # Unaveraged positions
        coords_unavgd = np.array(coords_data_unavgd[self.start:self.end]).astype(np.float32)

        fluid_xcoords_unavgd, fluid_ycoords_unavgd, fluid_zcoords_unavgd = coords_unavgd[:, fluid_idx, 0], \
                                                                           coords_unavgd[:, fluid_idx, 1], \
                                                                           coords_unavgd[:, fluid_idx, 2]

        # Wall Stresses -------------------------------------------------
        if solid_start != None:

            # Instanteneous extrema (array Needed for the gap height at each timestep)
            fluid_begin, fluid_end = utils.extrema(fluid_zcoords)['local_min'], \
                                     utils.extrema(fluid_zcoords)['local_max']

            fluid_zcoords_conv = utils.region(fluid_zcoords, fluid_xcoords, 0.1*Lx, 0.2*Lx)['data']
            fluid_begin_conv = utils.cnonzero_min(fluid_zcoords_conv)['local_min']
            fluid_end_conv = utils.extrema(fluid_zcoords_conv)['local_max']

            fluid_zcoords_div = utils.region(fluid_zcoords, fluid_xcoords, 0.7*Lx, 0.8*Lx)['data']
            fluid_begin_div = utils.cnonzero_min(fluid_zcoords_div)['local_min']
            fluid_end_div = utils.extrema(fluid_zcoords_div)['local_max']

            # Define the solid region ---------------------------------------------
            solid_xcoords, solid_ycoords, solid_zcoords = coords[:, solid_start:, 0], \
                                                          coords[:, solid_start:, 1], \
                                                          coords[:, solid_start:, 2]

            solid_xcoords_unavgd, solid_ycoords_unavgd, solid_zcoords_unavgd = coords_unavgd[:, solid_start:, 0], \
                                                          coords_unavgd[:, solid_start:, 1], \
                                                          coords_unavgd[:, solid_start:, 2]

            # To avoid problems with logical-and later
            solid_xcoords[solid_xcoords==0] = 1e-5

            # Define the surfU and surfL regions  ---------------------------------
            surfL = np.ma.masked_less(solid_zcoords, utils.extrema(fluid_zcoords)['global_max']/2.)
            surfU = np.ma.masked_greater(solid_zcoords, utils.extrema(fluid_zcoords)['global_max']/2.)
            # Indices of the the lower and upper
            surfL_indices = np.where(surfL[0].mask)[0]
            surfU_indices = np.where(surfU[0].mask)[0]

            surfL_xcoords, surfL_zcoords = solid_xcoords[:,surfL_indices],\
                                           solid_zcoords[:,surfL_indices]

            surfU_xcoords, surfU_zcoords = solid_xcoords[:,surfU_indices], \
                                           solid_zcoords[:,surfU_indices]

            # Check if surfU and surfL are not equal
            N_surfL, N_surfU = len(surfL_indices), len(surfU_indices)
            if N_surfU != N_surfL and rank == 0:
                logger.warning("No. of surfU atoms != No. of surfL atoms")

            # Instanteneous extrema (array Needed for the gap height at each timestep)
            surfU_begin, surfL_end = utils.cnonzero_min(surfU_zcoords)['local_min'], \
                                     utils.extrema(surfL_zcoords)['local_max']

            surfU_end, surfL_begin = utils.extrema(surfU_zcoords)['local_max'], \
                                     utils.extrema(surfL_zcoords)['local_min']

            # Average of the timesteps in this time slice for the avg. height
            avg_surfU_end, avg_surfL_begin = np.mean(comm.allgather(np.mean(surfU_end))), \
                                             np.mean(comm.allgather(np.mean(surfL_begin)))

            # Narrow range because of outliers in the fluid coordinates
            surfU_zcoords_conv = utils.region(surfU_zcoords, surfU_xcoords, 0.1*Lx, 0.2*Lx)['data']
            surfL_zcoords_conv = utils.region(surfL_zcoords, surfL_xcoords, 0.1*Lx, 0.2*Lx)['data']
            surfU_begin_conv = utils.cnonzero_min(surfU_zcoords_conv)['local_min']
            surfL_end_conv = utils.extrema(surfL_zcoords_conv)['local_max']

            # The converged and diverged regions fluid, surfU and surfL coordinates
            surfU_zcoords_div = utils.region(surfU_zcoords, surfU_xcoords, 0.7*Lx, 0.8*Lx)['data']
            surfL_zcoords_div = utils.region(surfL_zcoords, surfL_xcoords, 0.7*Lx, 0.8*Lx)['data']
            surfU_begin_div = utils.cnonzero_min(surfU_zcoords_div)['local_min']
            surfL_end_div = utils.extrema(surfL_zcoords_div)['local_max']

            # For vibrating walls ----------------------------------------------
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

            # Gap heights and COM (in Z-direction) at each timestep
            gap_height = (fluid_end - fluid_begin + surfU_begin - surfL_end) / 2.
            avg_gap_height = np.mean(comm.allgather(np.mean(gap_height)))

            gap_height_conv = (fluid_end_conv - fluid_begin_conv + surfU_begin_conv - surfL_end_conv) / 2.
            avg_gap_height_conv = np.mean(comm.allgather(np.mean(gap_height_conv)))

            gap_height_div = (fluid_end_div - fluid_begin_div + surfU_begin_div - surfL_end_div) / 2.
            avg_gap_height_div = np.mean(comm.allgather(np.mean(gap_height_div)))

            comZ = (surfU_begin - surfL_end) / 2.

            gap_heights = [gap_height, gap_height_conv, gap_height_div]

            #Update the box height
            Lz = avg_surfU_end - avg_surfL_begin

            cell_lengths_updated = [Lx, Ly, Lz]

            avg_surfL_end_div = np.mean(comm.allgather(np.mean(surfL_end_div)))
            avg_fluid_begin_div = np.mean(comm.allgather(np.mean(fluid_begin_div)))
            fluid_min = [utils.extrema(fluid_xcoords)['global_min'],
                         utils.extrema(fluid_ycoords)['global_min'],
                         (avg_surfL_end_div + avg_fluid_begin_div) /2.]

            fluid_lengths = [Lx, Ly, avg_gap_height_div]

            # Wave vectors in the z-direction
            kz = 2. * np.pi * nz / avg_gap_height_div

            fluid_vol = None

        else:
            comZ = None
            gap_heights = None
            cell_lengths_updated = [Lx, Ly, Lz]

            # For box volume that changes with time (i.e. NPT ensemble)
            fluid_min = [utils.extrema(fluid_xcoords)['global_min'],
                         utils.extrema(fluid_ycoords)['global_min'],
                         utils.extrema(fluid_zcoords)['global_min']]

            fluid_max = [utils.extrema(fluid_xcoords)['global_max'],
                         utils.extrema(fluid_ycoords)['global_max'],
                         utils.extrema(fluid_zcoords)['global_max']]

            fluid_lengths = [utils.extrema(fluid_xcoords)['global_max'] - utils.extrema(fluid_xcoords)['global_min'],
                             utils.extrema(fluid_ycoords)['global_max'] - utils.extrema(fluid_ycoords)['global_min'],
                             utils.extrema(fluid_zcoords)['global_max'] - utils.extrema(fluid_zcoords)['global_min']]

            cell_lengths = self.data.variables["cell_lengths"]
            # print(np.array(cell_lengths[self.start:self.end, 0]).astype(np.float32))
            fluid_vol = np.array(cell_lengths[self.start:self.end, 0]).astype(np.float32) * \
                        np.array(cell_lengths[self.start:self.end, 1]).astype(np.float32) * \
                        np.array(cell_lengths[self.start:self.end, 2]).astype(np.float32)

            # Wave vectors in the z-direction
            kz = 2. * np.pi * nz / Lz

        # REGIONS within the fluid------------------------------------------------
        #-------------------------------------------------------------------------
        # The Grid -------------------------------------------------------------
        # ----------------------------------------------------------------------
        dim_reciproc = np.array([kx, ky, kz], dtype=object)

        # -------------------------------------------------------
        # Cell Partition ---------------------------------------
        # -------------------------------------------------------
        # initialize buffers to store the count 'N' and 'data_ch' of each chunk

        # rho_kx_ch = np.zeros([self.chunksize, len(kx)] , dtype=np.complex64)
        sf_x = np.zeros([self.chunksize, len(kx)] , dtype=np.complex64)
        sf_y = np.zeros([self.chunksize, len(ky)] , dtype=np.complex64)
        sf_x_solid = np.zeros([self.chunksize, len(kx)] , dtype=np.complex64)
        sf_y_solid = np.zeros([self.chunksize, len(ky)] , dtype=np.complex64)
        rho_k = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)
        sf = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)
        sf_solid = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)


        # Mask the layer of interest for structure factor calculation
        maskx_layer = utils.region(fluid_xcoords, fluid_xcoords, 0, Lx)['mask']
        masky_layer = utils.region(fluid_ycoords, fluid_ycoords, 0, Ly)['mask']
        # maskz_layer = utils.region(fluid_zcoords, fluid_zcoords, 0, 10)['mask']       # Rigid walls
        maskz_layer = utils.region(fluid_zcoords, fluid_zcoords, fluid_Zstart, fluid_Zend)['mask']        # Vibrating walls

        maskx_layer_solid = utils.region(solid_xcoords, solid_xcoords, 0, Lx)['mask']
        masky_layer_solid = utils.region(solid_ycoords, solid_ycoords, 0, Ly)['mask']
        # maskz_layer_solid = utils.region(solid_zcoords, solid_zcoords, 7, 10)['mask']     # Rigid walls
        maskz_layer_solid = utils.region(solid_zcoords, solid_zcoords, solid_Zstart, solid_Zend)['mask']       # Vibrating walls

        mask_xy_layer = np.logical_and(maskx_layer, masky_layer)
        mask_layer = np.logical_and(mask_xy_layer, maskz_layer)

        mask_xy_layer_solid = np.logical_and(maskx_layer_solid, masky_layer_solid)
        mask_layer_solid = np.logical_and(mask_xy_layer_solid, maskz_layer_solid)

        # Count particles in the fluid cell
        N_layer_mask = np.sum(mask_layer, axis=1)
        Nzero_stable = np.less(N_layer_mask, 1)
        N_layer_mask[Nzero_stable] = 1

        N_layer_mask_solid = np.sum(mask_layer_solid, axis=1)
        Nzero_stable_solid = np.less(N_layer_mask_solid, 1)
        N_layer_mask_solid[Nzero_stable_solid] = 1

        # Structure factor
        for i in range(len(kx)):
            sf_x[:, i] = np.abs((np.sum(np.exp(1.j * kx[i] * fluid_xcoords_unavgd * mask_layer), axis=1)))**2 / N_layer_mask
            sf_x_solid[:, i] = np.abs((np.sum(np.exp(1.j * kx[i] * solid_xcoords_unavgd * mask_layer_solid), axis=1)))**2 / N_layer_mask_solid
        for k in range(len(ky)):
            sf_y[:, k] = np.abs((np.sum(np.exp(1.j * ky[k] * fluid_ycoords_unavgd * mask_layer), axis=1)))**2 / N_layer_mask
            sf_y_solid[:, k] = np.abs((np.sum(np.exp(1.j * ky[k] * solid_ycoords_unavgd * mask_layer_solid), axis=1)))**2 / N_layer_mask_solid

        for i in range(len(kx)):
            for k in range(len(ky)):
                # Fourier components of the density
                rho_k[:, i, k] = np.sum( np.exp(1.j * (kx[i]*fluid_xcoords_unavgd*mask_layer +
                                    ky[k]*fluid_ycoords_unavgd*mask_layer)) )
                sf[:, i, k] =  np.abs((np.sum( np.exp(1.j * (kx[i]*fluid_xcoords_unavgd*mask_layer +
                                    ky[k]*fluid_ycoords_unavgd*mask_layer) ) , axis=1)))**2 / N_layer_mask
                sf_solid[:, i, k] =  np.abs((np.sum( np.exp(1.j * (kx[i]*solid_xcoords_unavgd*mask_layer_solid +
                                    ky[k]*solid_ycoords_unavgd*mask_layer_solid) ) , axis=1)))**2 / N_layer_mask_solid


        return {'cell_lengths': cell_lengths_updated,
                'kx': kx,
                'ky': ky,
                'kz': kz,
                'gap_heights': gap_heights,
                'com': comZ,
                'rho_k': rho_k,
                'sf': sf,
                'sf_solid': sf_solid,
                'sf_x': sf_x,
                'sf_x_solid': sf_x_solid,
                'sf_y': sf_y,
                'sf_y_solid': sf_y_solid,
                'Nf': Nf,
                'Nm': Nm,
                'fluid_vol':fluid_vol}
