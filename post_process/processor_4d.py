#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging
import warnings
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

# For the spatial bins with 1 or zero atoms. These are masked later in computing properties.
warnings.simplefilter('ignore', category=RuntimeWarning)

class TrajtoGrid:
    """
    Molecular domain decomposition into spatial bins
    """
    def __init__(self, data, start, end, Nx, Ny, Nz, mf, A_per_molecule, fluid, tessellate, TW_interface):
        """
        Parameters:
        -----------
        data: str, NetCDF trajectory file
        start, end: Start and end of the sampled time in each processor
        Nx, Ny, Nz: int, Number of chunks in the x-, y- and z-direcions
        mf: float, molecular mass of fluid molecule
        A_per_molecule: int, no. of atoms per molecule
        fluid: str, fluid material
        tessellate: int, boolean to perform Delaunay tessellation
        TW_interface: int, boolean to define the vibrating portion of atoms in the upper wall
                Default = 1: thermostat applied on the wall layers in contact with the fluid
        """

        self.data = data
        self.start, self.end = start, end
        self.chunksize = self.end - self.start
        self.Nx, self.Ny, self.Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)
        self.mf, self.A_per_molecule = comm.bcast(mf, root=0), comm.bcast(A_per_molecule, root=0)
        self.fluid = comm.bcast(fluid, root=0)
        self.tessellate = comm.bcast(tessellate, root=0)
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
        types = np.array(type[0, :]).astype(np.float32)

        fluid_idx, solid_idx = [],[]    # Should be of shapes: (Nf,) and (Ns,)

        # Lennard-Jones
        if np.max(types)==2:
            fluid_idx.append(np.where(types == 1))
            fluid_idx = fluid_idx[0][0]
            solid_idx.append(np.where(types == 2))
            solid_idx = solid_idx[0][0]
        # Hydrocarbons
        if np.max(types)==3:
            fluid_idx.append(np.where([types == 1, types == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where(types == 3))
            solid_idx = solid_idx[0][0]
        # Hydrocarbons with sticking and slipping walls
        if np.max(types)==4 and self.fluid=='pentane':
            fluid_idx.append(np.where([types == 1, types == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where([types == 3, types == 4]))
            solid_idx = solid_idx[0][1]
        # Squalane
        if np.max(types)==4 and self.fluid=='squalane':
            fluid_idx.append(np.where([types == 1, types == 2, types == 3]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where([types == 4]))
            solid_idx = solid_idx[0][1]

        solid_start = np.min(solid_idx)

        Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/self.A_per_molecule, \
                                np.max(solid_idx)-np.max(fluid_idx)

        if rank == 0:
            logger.info('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,int(Nm),Ns))

        return fluid_idx, solid_idx, solid_start, Nf, Nm


    def get_chunks(self, stable_start, stable_end, pump_start, pump_end):
        """
        Partitions the box into regions (solid and fluid) as well as chunks in
        those regions
        Parameters (normalized by Lx)
        -----------------------------
        stable_start: float, Stable region (for sampling) start
        stable_end: float, Stable region (for sampling) end
        pump_start: float, Pump region (pertrubation region) start
        pump_end: float, Stable region (pertrubation field) end

        Returns
        ----------
        Thermodynamic quantities (arrays) with different shapes. The 4 shapes are:
            (time,): time series of a quantity for the whole domain
                    (6 quantites: heat flux vector and volumes)
            (Nf,): time averaged quantity for each particle
                    (3 quantites: velocity vector)
            (time,Nx): time series of the chunks along the x-direction
            (time,Nx,Ny,Nz): time series of each chunk along the x and the z-directions
                    (10 quantites: Vx, density, temp, pressure)
        """

        fluid_idx, solid_idx, solid_start, Nf, Nm = self.get_indices()
        Lx, Ly, Lz = self.get_dimensions()

        # Position, velocity, virial (Wi) array dimensions: (time, Natoms, dimension)
        # The unaveraged quantity is that which is dumped by LAMMPS every N timesteps
        # i.e. it is a snapshot of the system at this timestep.
        # The averaged quantities which are the average of a few previous timesteps
        coords_data = self.data.variables["coordinates"]
        try:
            vels_data = self.data.variables["f_velocity"]    # averaged velocities
        except KeyError:
            vels_data = self.data.variables["velocities"]    # unaveraged velocities used for temp. calculation

        vels_data_unavgd = self.data.variables["velocities"]

        # Fluid Virial Pressure ------------------------------------------------
        virial_data = self.data.variables["f_Wi_avg"]
        virial = np.array(virial_data[self.start:self.end]).astype(np.float32)

        # Fluid mass -----------------------------------------------------------
        # Assign mass to type (Could be replaced by dumping mass from LAMMPS)
        type_array = self.data.variables["type"]
        type_array = np.array(type_array[self.start:self.end]).astype(np.float32)
        mass = np.zeros_like(type_array)

        types = np.array(type_array[0, :]).astype(np.float32)
        if np.max(types)==2: mass_map = {1: 39.948, 2: 196.96}
        if np.max(types)==3: mass_map = {1: 15.03462, 2: 14.02667, 3: 196.96}
        if np.max(types)==4 and self.fluid=='pentane': {1: 15.03462, 2: 14.02667, 3: 196.96, 4: 196.96}
        if np.max(types)==4 and self.fluid=='squalane': {1: 13.019, 2: 14.02667, 3: 15.03462, 4: 196.96}

        for i in range(type_array.shape[0]):
            for j in range(type_array.shape[1]):
                mass[i, j] = mass_map.get(type_array[i, j], 0)

        # Fluid mass
        mass_fluid = mass[:, fluid_idx]      # (time, Nf)

        # ---------------------------------------------------------------------------------
        # Whole domain: arrays of shape (time,), (Nf,), (time,Ns), ------------------------
        # (time,NsurfU), (time,NsurfL) and (time, Nf) -------------------------------------
        # ---------------------------------------------------------------------------------

        # Cartesian Coordinates ---------------------------------------------
        coords = np.array(coords_data[self.start:self.end]).astype(np.float32)

        # Shape: (time, Nf)
        fluid_xcoords, fluid_ycoords, fluid_zcoords = coords[:, fluid_idx, 0], \
                                                    coords[:, fluid_idx, 1], \
                                                    coords[:, fluid_idx, 2]

        # Shape: (time, Ns)
        solid_xcoords, solid_ycoords, solid_zcoords = coords[:, solid_start:, 0], \
                                                      coords[:, solid_start:, 1], \
                                                      coords[:, solid_start:, 2]

        # Fluid min and max z-coordinates in each timestep,  shape: (time,)
        fluid_begin, fluid_end = utils.extrema(fluid_zcoords)['local_min'], \
                                 utils.extrema(fluid_zcoords)['local_max']

        # Average of the timesteps in this time slice for the avg. height
        # Shape: int
        avg_fluid_begin, avg_fluid_end = np.mean(comm.allgather(np.mean(fluid_begin))), \
                                         np.mean(comm.allgather(np.mean(fluid_end)))

        # Define the upper surface and lower surface regions
        # To avoid problems with logical-and boolean
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
            logger.warning(f'No. of surfU atoms {N_surfU} != No. of surfL atoms {N_surfL}')

        # Shape: (time,)
        surfU_begin, surfL_end = utils.cnonzero_min(surfU_zcoords)['local_min'], \
                                 utils.extrema(surfL_zcoords)['local_max']

        surfU_end, surfL_begin = utils.extrema(surfU_zcoords)['local_max'], \
                                 utils.extrema(surfL_zcoords)['local_min']

        # Average of the timesteps in this time slice for the avg. height
        # Shape: int
        avg_surfU_begin, avg_surfL_end = np.mean(comm.allgather(np.mean(surfU_begin))), \
                                         np.mean(comm.allgather(np.mean(surfL_end)))
        avg_surfU_end, avg_surfL_begin = np.mean(comm.allgather(np.mean(surfU_end))), \
                                         np.mean(comm.allgather(np.mean(surfL_begin)))

        # For vibrating walls
        if self.TW_interface == 1: # Thermostat is applied on the walls directly at the interface (half of the wall is vibrating)
            surfU_vib_end = utils.extrema(surfU_zcoords)['global_max'] - \
                                0.5 * (utils.extrema(surfL_zcoords)['global_max'] - avg_surfL_begin)
        else: # Thermostat is applied on the walls away from the interface (2/3 of the wall is vibrating)
            surfU_vib_end = utils.extrema(surfU_zcoords)['global_max'] - \
                                0.167 * (utils.extrema(surfL_zcoords)['global_max'] - avg_surfL_begin)

        surfU_vib = np.ma.masked_less(surfU_zcoords, surfU_vib_end)
        surfU_vib_indices = np.where(surfU_vib[0].mask)[0]

        if rank == 0:
            logger.info(f'Number of vibraing atoms in the upper surface: {len(surfU_vib_indices)}')

        surfU_vib_xcoords, surfU_vib_ycoords, surfU_vib_zcoords = solid_xcoords[:,surfU_vib_indices], \
                    solid_ycoords[:,surfU_vib_indices], solid_zcoords[:,surfU_vib_indices]

        # Atomic mass of the upper vibrating region, Shape: (time, NsurfU_vib)
        mass_solid = mass[:, solid_start:]
        mass_surfU_vib = mass_solid[:, surfU_vib_indices]

        # Gap heights and COM (in Z-direction) at each timestep
        # Shape: (time,)
        gap_height = (fluid_end - fluid_begin + surfU_begin - surfL_end) / 2.
        comZ = (surfU_end + surfL_begin) / 2.
        # Shape: int
        avg_gap_height = np.mean(comm.allgather(np.mean(gap_height)))

        # Update the entire domain dimensions
        Lz = utils.extrema(surfU_zcoords)['global_max'] - avg_surfL_begin
        cell_lengths_updated = [Lx, Ly, Lz]
        domain_min = [0, 0, avg_surfL_begin]

        # Fluid domain dimensions
        fluid_lengths = [Lx, Ly, (avg_fluid_end + avg_surfU_begin) / 2. - (avg_fluid_begin + avg_surfL_end) / 2.]
        fluid_domain_min = [0, 0, (avg_fluid_begin + avg_surfL_end) / 2.]

        # Velocities -----------------------------------------------------------
        vels = np.array(vels_data[self.start:self.end]).astype(np.float32)
        # Unaveraged velocities
        vels_t = np.array(vels_data_unavgd[self.start:self.end]).astype(np.float32)

        # Velocity of each atom
        # Shape: (time, Nf)
        fluid_vx, fluid_vy, fluid_vz = vels[:, fluid_idx, 0], \
                                       vels[:, fluid_idx, 1], \
                                       vels[:, fluid_idx, 2]


        fluid_vx_t, fluid_vy_t, fluid_vz_t = vels_t[:, fluid_idx, 0], \
                                             vels_t[:, fluid_idx, 1], \
                                             vels_t[:, fluid_idx, 2]

        # Shape: (time, NsurfU_vib)
        surfU_vx_vib, surfU_vy_vib, surfU_vz_vib = vels_t[:, solid_start:, 0][:,surfU_vib_indices], \
                                                   vels_t[:, solid_start:, 1][:,surfU_vib_indices], \
                                                   vels_t[:, solid_start:, 2][:,surfU_vib_indices]

        # ---------------------------------------------------------------------------------
        # Regions within the fluid: arrays of shape (time,), (Nf,) and (time, Nf) ---------
        # ---------------------------------------------------------------------------------

        # Pump -----------------------------------------------------------------
        pumpStartX, pumpEndX = pump_start * Lx, pump_end * Lx
        # Shapes: float, Boolean (time, Nf), (time, Nf), (time,), (time,)
        pump_length, pump_region, pump_xcoords, pump_vol, pump_N = \
            itemgetter('interval', 'mask', 'data', 'vol', 'count')\
            (utils.region(fluid_xcoords, fluid_xcoords, pumpStartX, pumpEndX,
                                    ylength=Ly, length=avg_gap_height))

        # Bulk -----------------------------------------------------------------
        bulkStartZ, bulkEndZ = (0.4 * Lz) + avg_surfL_begin, (0.6 * Lz) + avg_surfL_begin     # Boffset is 1 Angstrom
        # Shapes: float, Boolean (time, Nf), (time, Nf), (time,), (time,)
        bulk_height, bulk_region, bulk_zcoords, bulk_vol, bulk_N = \
            itemgetter('interval', 'mask', 'data', 'vol', 'count')\
            (utils.region(fluid_zcoords, fluid_zcoords, bulkStartZ, bulkEndZ,
                                            ylength=Ly, length=Lx))
        # Shape: (time,)
        bulkStartZ_time = 0.4 * (surfU_end - surfL_begin) + surfL_begin
        bulkEndZ_time = 0.6 * (surfU_end - surfL_begin) + surfL_begin

        # Voronoi tesellation volume, performed during simulation (Å^3)
        try:
            voronoi_vol_data = self.data.variables["f_Vi_avg"]  # (time, Natoms)
            voronoi_vol = np.array(voronoi_vol_data[self.start:self.end]).astype(np.float32)
            Vi = voronoi_vol[:, fluid_idx] * bulk_region  # (time, Nf)
            totVi = np.sum(Vi, axis=1)                    # (time, )
        except KeyError:
            totVi = np.zeros([self.chunksize], dtype=np.float32)
            if rank == 0: logger.info('Voronoi volume was not computed during LAMMPS Run!')
            pass

        # Delaunay triangulation (performed in post-processing)
        if self.tessellate==1:
            bulk_xcoords = fluid_xcoords * bulk_region
            bulk_ycoords = fluid_ycoords * bulk_region
            bulk_zcoords = fluid_zcoords * bulk_region

            bulk_coords = np.zeros((self.chunksize, Nf, 3))
            bulk_coords[:,:,0], bulk_coords[:,:,1], bulk_coords[:,:,2] = \
            bulk_xcoords, bulk_ycoords, bulk_zcoords

            del_totVi = np.zeros([self.chunksize], dtype=np.float32)
            for i in range(self.chunksize):
                del_totVi[i] = tessellation.delaunay_volumes(bulk_coords[i])
        else:
            del_totVi = np.zeros([self.chunksize], dtype=np.float32)

        # Stable ---------------------------------------------------------------
        stableStartX, stableEndX = stable_start * Lx, stable_end * Lx
        # Shapes: float, Boolean (time, Nf), (time, Nf), (time,), (time,)
        stable_length, stable_region, stable_xcoords, stable_vol, stable_N = \
            itemgetter('interval', 'mask', 'data', 'vol', 'count')\
            (utils.region(fluid_xcoords, fluid_xcoords, stableStartX, stableEndX,
                                            ylength=Ly, length=avg_gap_height))

        # ----------------------------------------------------------------------
        # The Grid -------------------------------------------------------------
        # ----------------------------------------------------------------------
        dim = np.array([self.Nx, self.Ny, self.Nz])

        # Whole cell -----------------------------------------------------------
        # The minimum position has to be added to have equal Number of solid atoms in each chunk
        bounds = [np.arange(dim[i] + 1) / dim[i] * cell_lengths_updated[i] + domain_min[i]
                        for i in range(3)]
        xx, yy, zz, vol_cell = utils.bounds(bounds[0], bounds[1], bounds[2])

        # Fluid ----------------------------------------------------------------
        bounds_fluid = [np.arange(dim[i] + 1) / dim[i] * fluid_lengths[i] + fluid_domain_min[i]
                        for i in range(3)]
        xx_fluid, yy_fluid, zz_fluid, vol_fluid = utils.bounds(bounds_fluid[0],
                                                  bounds_fluid[1], bounds_fluid[2])

        # Bulk -----------------------------------------------------------------
        bulkRangeZ = np.arange(dim[2] + 1) / dim[2] * bulk_height + bulkStartZ
        bounds_bulk = [bounds_fluid[0], bounds_fluid[1], bulkRangeZ]
        xx_bulk, yy_bulk, zz_bulk, vol_bulk_cell = utils.bounds(bounds_bulk[0],
                                                   bounds_bulk[1], bounds_bulk[2])

        # Stable ---------------------------------------------------------------
        stableRangeX = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX
        bounds_stable = [stableRangeX, bounds_fluid[1], bounds_fluid[2]]
        xx_stable, yy_stable, zz_stable, vol_stable_cell = utils.bounds(bounds_stable[0],
                                                            bounds_stable[1], bounds_stable[2])

        # ----------------------------------------------------------------------
        # Cell Partition: arrays of shape (time, Nx, Ny, Nz) and (time, Nx)---------
        # ----------------------------------------------------------------------
        # initialize buffers to store the count 'N' and 'data_ch' of each chunk
        N_fluid_mask = np.zeros([self.chunksize, self.Nx, self.Ny, self.Nz], dtype=np.float32)
        N_fluid_mask_non_zero = np.zeros_like(N_fluid_mask)
        N_stable_mask = np.zeros_like(N_fluid_mask)
        N_bulk_mask = np.zeros_like(N_fluid_mask)
        vx_ch = np.zeros_like(N_fluid_mask)
        vx_ch_whole = np.zeros_like(vx_ch)
        vir_ch = np.zeros_like(vx_ch)
        Wxx_ch = np.zeros_like(vx_ch)
        Wyy_ch = np.zeros_like(vx_ch)
        Wzz_ch = np.zeros_like(vx_ch)
        Wxy_ch = np.zeros_like(vx_ch)
        Wxz_ch = np.zeros_like(vx_ch)
        Wyz_ch = np.zeros_like(vx_ch)
        den_ch = np.zeros_like(vx_ch)
        jx_ch = np.zeros_like(vx_ch)
        mflowrate_ch = np.zeros_like(vx_ch)
        temp_ch = np.zeros_like(vx_ch)
        tempx_ch = np.zeros_like(vx_ch)
        tempy_ch = np.zeros_like(vx_ch)
        tempz_ch = np.zeros_like(vx_ch)
        temp_ch_solid = np.zeros_like(vx_ch)
        uCOMx = np.zeros_like(vx_ch)
        den_bulk_ch = np.zeros_like(vx_ch)

        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
            # Fluid -----------------------------------------
                    maskx_fluid = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_fluid[i, j, k], xx_fluid[i+1, j, k])['mask']
                    masky_fluid = utils.region(fluid_ycoords, fluid_ycoords,
                                            yy_fluid[i, j, k], yy_fluid[i, j+1, k])['mask']
                    maskz_fluid = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_fluid[i, j, k], zz_fluid[i, j, k+1])['mask']
                    maskxy_fluid = np.logical_and(maskx_fluid, masky_fluid)
                    mask_fluid = np.logical_and(maskxy_fluid, maskz_fluid)

                    # Count particles in the fluid cell
                    N_fluid_mask[:, i, j, k] = np.sum(mask_fluid, axis=1)
                    # Avoid having zero particles in the cell
                    N_fluid_mask_non_zero[:, i, j, k] = np.less(N_fluid_mask[:, i, j, k], 1)
                    Nzero_fluid = np.less(N_fluid_mask_non_zero[:, i, j, k], 1)
                    N_fluid_mask_non_zero[Nzero_fluid, i, j, k] = 1

            # Bulk -----------------------------------
                    maskx_bulk = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_bulk[i, j, k], xx_bulk[i+1, j, k])['mask']
                    masky_bulk = utils.region(fluid_ycoords, fluid_ycoords,
                                        yy_bulk[i, j, k], yy_bulk[i, j+1, k])['mask']
                    maskz_bulk = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_bulk[i, j, k], zz_bulk[i, 0, k+1])['mask']
                    maskxy_bulk = np.logical_and(maskx_bulk, masky_bulk)
                    mask_bulk = np.logical_and(maskxy_bulk, maskz_bulk)

                    # Count particles in the bulk cell
                    N_bulk_mask[:, i, j, k] = np.sum(mask_bulk, axis=1)

            # Stable --------------------------------------
                    maskx_stable = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_stable[i, j, k], xx_stable[i+1, j, k])['mask']
                    masky_stable = utils.region(fluid_ycoords, fluid_ycoords,
                                        yy_stable[i, j, k], yy_stable[i, j+1, k])['mask']
                    maskz_stable = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_stable[i, j, k], zz_stable[i, 0, k+1])['mask']
                    maskxy_stable = np.logical_and(maskx_stable, masky_stable)
                    mask_stable = np.logical_and(maskxy_stable, maskz_stable)

                    # Count particles in the stable cell
                    N_stable_mask[:, i, j, k] = np.sum(mask_stable, axis=1)
                    Nzero_stable = np.less(N_stable_mask[:, i, j, k], 1)
                    N_stable_mask[Nzero_stable, i, j, k] = 1

        # -----------------------------------------------------
        # Thermodynamic properties ----------------------------
        # -----------------------------------------------------

                    # Whole fluid domain -------------------------------------------

                    # Velocities in the  (Å/fs)
                    vx_ch_whole[:, i, j, k] = np.sum(fluid_vx * mask_fluid, axis=1) / N_fluid_mask[:, i, j, k]

                    # Density in the whole fluid (g/(mol.Å^3))
                    den_ch[:, i, j, k] = np.sum(mass_fluid * mask_fluid, axis=1) / vol_fluid[i, j, k]

                    # Density (g/(mol.Å^3))
                    den_bulk_ch[:, i, j, k] = np.sum(mass_fluid * mask_bulk, axis=1) / vol_bulk_cell[i, j, k]

                    # Mass flux in the whole fluid (g/(mol.Å^2.fs))
                    jx_ch[:, i, j, k] = vx_ch_whole[:, i, j, k] * den_ch[:, i, j, k]

                    # Mass flow rate in the whole fluid (g/(mol.fs))
                    mflowrate_ch[:, i, j, k] = np.sum(mass_fluid * fluid_vx * mask_fluid, axis=1) / (Lx / self.Nx)

                    # Temperature (K)
                    # COM velocity in the bin, shape (time,)
                    uCOM = np.sum(fluid_vx_t * mask_fluid, axis=1) / (N_fluid_mask_non_zero[:, i, j, k])
                    vCOM = np.sum(fluid_vy_t * mask_fluid, axis=1) / (N_fluid_mask_non_zero[:, i, j, k])
                    wCOM = np.sum(fluid_vz_t * mask_fluid, axis=1) / (N_fluid_mask_non_zero[:, i, j, k])
                    # Remove the streaming velocity from the lab frame velocity to get the thermal/peculiar velocity
                    # Shape: (Nf, time)
                    peculiar_vx = np.transpose(fluid_vx_t) - uCOM
                    peculiar_vy = np.transpose(fluid_vy_t) - vCOM
                    peculiar_vz = np.transpose(fluid_vz_t) - wCOM
                    peculiar_v = np.sqrt(peculiar_vx**2 + peculiar_vy**2 + peculiar_vz**2)
                    # Shape: (time, Nf)
                    peculiar_vx =  np.transpose(peculiar_vx) * mask_fluid
                    peculiar_vy =  np.transpose(peculiar_vy) * mask_fluid
                    peculiar_vz =  np.transpose(peculiar_vz) * mask_fluid
                    peculiar_v = np.transpose(peculiar_v) * mask_fluid

                    tempx_ch[:, i, j, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_vx**2 * A_per_fs_to_m_per_s**2, axis=1) / \
                                        ((N_fluid_mask[:, i, j, k] - 1) * sci.k)

                    tempy_ch[:, i, j, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_vy**2 * A_per_fs_to_m_per_s**2, axis=1) / \
                                        ((N_fluid_mask[:, i, j, k] - 1) * sci.k)

                    tempz_ch[:, i, j, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_vz**2 * A_per_fs_to_m_per_s**2, axis=1) / \
                                        ((N_fluid_mask[:, i, j, k] - 1) * sci.k)

                    temp_ch[:, i, j, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_v**2 * A_per_fs_to_m_per_s**2 , axis=1) / \
                                        ((3 * N_fluid_mask[:, i, j, k] - 3) * sci.k)

                    # Virial off-diagonal components (atm)
                    try:
                        Wxy_ch[:, i, j, k] = np.sum(virial[:, fluid_idx, 3] * mask_fluid, axis=1) / vol_fluid[i,j,k]
                        Wxz_ch[:, i, j, k] = np.sum(virial[:, fluid_idx, 4] * mask_fluid, axis=1) / vol_fluid[i,j,k]
                        Wyz_ch[:, i, j, k] = np.sum(virial[:, fluid_idx, 5] * mask_fluid, axis=1) / vol_fluid[i,j,k]
                    except IndexError:
                        pass

                    # Bulk ---------------------------------------------------------
                    # Virial diagonal components (atm)
                    Wxx_ch[:, i, j, k] = np.sum(virial[:, fluid_idx, 0] * mask_bulk, axis=1) / vol_bulk_cell[i,j,k]
                    Wyy_ch[:, i, j, k] = np.sum(virial[:, fluid_idx, 1] * mask_bulk, axis=1) / vol_bulk_cell[i,j,k]
                    Wzz_ch[:, i, j, k] = np.sum(virial[:, fluid_idx, 2] * mask_bulk, axis=1) / vol_bulk_cell[i,j,k]
                    vir_ch[:, i, j, k] = -(Wxx_ch[:, i, j, k] + Wyy_ch[:, i, j, k] + Wzz_ch[:, i, j, k]) / 3.

                    # Velocities in the stable region (Å/fs)
                    vx_ch[:, i, j, k] = np.sum(fluid_vx * mask_stable, axis=1) / N_stable_mask[:, i, j, k]


        return {'cell_lengths': cell_lengths_updated,
                'gap_height': gap_height,
                'bulkStartZ_time': bulkStartZ_time,
                'bulkEndZ_time': bulkEndZ_time,
                'com': comZ,
                'Nf': Nf,
                'Nm': Nm,
                'totVi': totVi,
                'del_totVi': del_totVi,
                'vx_ch_whole':vx_ch_whole,
                'den_ch': den_ch,
                'jx_ch' : jx_ch,
                'mflowrate_ch': mflowrate_ch,
                'tempx_ch': tempx_ch,
                'tempy_ch': tempy_ch,
                'tempz_ch': tempz_ch,
                'temp_ch': temp_ch,
                'Wxy_ch': Wxy_ch,
                'Wxz_ch': Wxz_ch,
                'Wyz_ch': Wyz_ch,
                'Wxx_ch': Wxx_ch,
                'Wyy_ch': Wyy_ch,
                'Wzz_ch': Wzz_ch,
                'vir_ch': vir_ch,
                'den_bulk_ch':den_bulk_ch,
                'vx_ch': vx_ch}
