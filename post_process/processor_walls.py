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
            (time,Nx,Nz): time series of each chunk along the x and the z-directions
                    (10 quantites: Vx, density, temp, pressure)
        """

        fluid_idx, solid_idx, solid_start, Nf, Nm = self.get_indices()
        Lx, Ly, Lz = self.get_dimensions()

        # Position, velocity, virial (Wi) array dimensions: (time, Natoms, dimension)
        # The unaveraged quantity is that which is dumped by LAMMPS every N timesteps
        # i.e. it is a snapshot of the system at this timestep.
        # The averaged quantities are the moving average of a few previous timesteps

        coords_data = self.data.variables["coordinates"]
        try:
            vels_data = self.data.variables["f_velocity"]     # averaged velocities
        except KeyError:
            vels_data = self.data.variables["velocities"]     # unaveraged velocities used for temp. calculation

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
        if np.max(types)==2: mass_map = {1: 39.948, 2: 196.96}  # argon and gold
        if np.max(types)==3: mass_map = {1: 15.03462, 2: 14.02667, 3: 196.96}   # CH3, CH2 and gold
        if np.max(types)==4 and self.fluid=='pentane': {1: 15.03462, 2: 14.02667, 3: 196.96, 4: 196.96} # CH3, CH2 and wetting and non-wetting gold
        if np.max(types)==4 and self.fluid=='squalane': {1: 13.019, 2: 14.02667, 3: 15.03462, 4: 196.96}  # CH, CH2, CH3 and gold

        for i in range(type_array.shape[0]):
            for j in range(type_array.shape[1]):
                mass[i, j] = mass_map.get(type_array[i, j], 0)

        # Fluid mass
        mass_fluid = mass[:, fluid_idx]      # (time, Nf)

        # ---------------------------------------------------------------------------------
        # Whole domain: arrays of shape (time,), (Nf,), (time,Ns), ------------------------
        # (time,NsurfU), (time,NsurfL) and (time, Nf) -------------------------------------
        # ---------------------------------------------------------------------------------

        # Cartesian Coordinates ------------------------------------------------
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
            surfU_vib_end = utils.cnonzero_min(surfU_zcoords)['local_min'] + \
                                0.5 * utils.extrema(surfL_zcoords)['local_max'] + avg_surfL_begin
        else: # Thermostat is applied on the walls away from the interface (2/3 of the wall is vibrating)
            surfU_vib_end = utils.cnonzero_min(surfU_zcoords)['local_min'] + \
                                0.667 * utils.extrema(surfL_zcoords)['local_max'] + avg_surfL_begin
        avg_surfU_vib_end = np.mean(comm.allgather(np.mean(surfU_vib_end)))

        surfU_vib = np.ma.masked_less(surfU_zcoords, avg_surfU_vib_end)
        surfU_vib_indices = np.where(surfU_vib[0].mask)[0]

        if rank == 0:
            logger.info(f'Number of vibraing atoms in the upper surface: {len(surfU_vib_indices)}')

        surfU_vib_xcoords, surfU_vib_zcoords = solid_xcoords[:,surfU_vib_indices], \
        solid_zcoords[:,surfU_vib_indices]

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
        Lz = avg_surfU_end - avg_surfL_begin
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

        # Wall Forces ----------------------------------------------------------
        try:
            forcesU = self.data.variables["f_fAtomU_avg"]
            forcesL = self.data.variables["f_fAtomL_avg"]

            # Shape: (time, Nsurf)
            forcesU_data = np.array(forcesU[self.start:self.end]).astype(np.float32)
            forcesL_data = np.array(forcesL[self.start:self.end]).astype(np.float32)

            surfU_fx,surfU_fy,surfU_fz = forcesU_data[:, solid_start:, 0][:,surfU_indices], \
                                         forcesU_data[:, solid_start:, 1][:,surfU_indices], \
                                         forcesU_data[:, solid_start:, 2][:,surfU_indices]

            surfL_fx,surfL_fy,surfL_fz = forcesL_data[:, solid_start:, 0][:,surfL_indices], \
                                         forcesL_data[:, solid_start:, 1][:,surfL_indices], \
                                         forcesL_data[:, solid_start:, 2][:,surfL_indices]
        except KeyError:
            surfU_fx, surfU_fy, surfU_fz = 0, 0, 0
            surfL_fx, surfL_fy, surfL_fz = 0, 0, 0

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

        # Mass flux (g/m2.ns) and flow rate (g/ns) in the pump region
        vels_pump = fluid_vx * pump_region / self.A_per_molecule        # (time, Nf)
        # Shape: (time,)
        mflux_pump = (np.sum(vels_pump, axis=1) * (sci.angstrom/fs_to_ns) * \
                     (self.mf/sci.N_A)) / (pump_vol * (sci.angstrom)**3)
        mflowrate_pump = (np.sum(vels_pump, axis=1) * (sci.angstrom/fs_to_ns) * \
                        (self.mf/sci.N_A)) / (pump_length * sci.angstrom)

        # Bulk -----------------------------------------------------------------
        bulkStartZ, bulkEndZ = (0.4 * Lz) + avg_surfL_begin , (0.6 * Lz) + avg_surfL_begin
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

        # Mass flux (g/m2.ns) and flow rate (g/ns) in the stable region
        vels_stable = fluid_vx * stable_region / self.A_per_molecule
        # Shape: (time,)
        mflux_stable = (np.sum(vels_stable, axis=1) * (sci.angstrom/fs_to_ns) * \
                       (self.mf/sci.N_A))/ (stable_vol * (sci.angstrom)**3)      # g/m2.ns
        mflowrate_stable = (np.sum(vels_stable, axis=1) * (sci.angstrom/fs_to_ns) * \
                        (self.mf/sci.N_A)) / (stable_length * sci.angstrom)       # g/ns

        fluxes = [mflux_pump, mflowrate_pump, mflux_stable, mflowrate_stable]

        # Heat flux from ev: total energy * velocity and sv: energy from virial * velocity (Kcal/mol . Å/fs)
        try:
            ev_data = self.data.variables["f_ev"]       # (time, Natoms, 3)
            centroid_vir_data = self.data.variables["f_sv"]     # (time, Natoms, 3)
            ev = np.array(ev_data[self.start:self.end]).astype(np.float32)
            centroid_vir = np.array(centroid_vir_data[self.start:self.end]).astype(np.float32)

            fluid_evx, fluid_evy, fluid_evz = ev[:, fluid_idx, 0], \
                                              ev[:, fluid_idx, 1], \
                                              ev[:, fluid_idx, 2]

            fluid_svx, fluid_svy, fluid_svz = centroid_vir[:, fluid_idx, 0] ,\
                                              centroid_vir[:, fluid_idx, 1] ,\
                                              centroid_vir[:, fluid_idx, 2]

            je_x = np.sum((fluid_evx + fluid_svx) * stable_region, axis=1)    # (time,)
            je_y = np.sum((fluid_evy + fluid_svy) * stable_region, axis=1)
            je_z = np.sum((fluid_evz + fluid_svz) * stable_region, axis=1)
        except KeyError:
            je_x, je_y, je_z = np.zeros([self.chunksize], dtype=np.float32),\
                               np.zeros([self.chunksize], dtype=np.float32),\
                               np.zeros([self.chunksize], dtype=np.float32)

        # A cube in the center of the fluid (5x5x5 Å^3) ------------------------
        # To measure velocity distribution and test for Local thermodynamic equilibrium (LTE)
        xcom, ycom, zcom = (cell_lengths_updated[0] + domain_min[0]) / 2., \
                           (cell_lengths_updated[1] + domain_min[1]) / 2., \
                           (cell_lengths_updated[2] + domain_min[2]) / 2.
        maskx_lte = utils.region(fluid_xcoords, fluid_xcoords, xcom-5, xcom+5)['mask']
        masky_lte = utils.region(fluid_ycoords, fluid_ycoords, ycom-5, ycom+5)['mask']
        maskz_lte = utils.region(fluid_zcoords, fluid_zcoords, zcom-5, zcom+5)['mask']
        mask_xy_lte = np.logical_and(maskx_lte, masky_lte)
        mask_lte = np.logical_and(mask_xy_lte, maskz_lte)

        # Shape: (time,)
        N_lte = np.sum(mask_lte, axis=1)    # No. of fluid atoms in the cube
        Nzero_lte = np.less(N_lte, 1)
        N_lte[Nzero_lte] = 1

        uCOM_lte = np.sum(fluid_vx_t * mask_lte, axis=1) / N_lte   # COM velocity in the cube
        vCOM_lte = np.sum(fluid_vy_t * mask_lte, axis=1) / N_lte
        wCOM_lte = np.sum(fluid_vz_t * mask_lte, axis=1) / N_lte

        # Remove the streaming velocity from the lab frame velocity to get the thermal/peculiar velocity
        # Shape: (Nf, time)
        peculiar_vx = np.transpose(fluid_vx_t) - uCOM_lte
        peculiar_vy = np.transpose(fluid_vy_t) - vCOM_lte
        peculiar_vz = np.transpose(fluid_vz_t) - wCOM_lte

        # Shape: (Nf,)
        fluid_vx_avg_lte = np.mean(np.transpose(peculiar_vx) * mask_lte, axis=0)
        fluid_vy_avg_lte = np.mean(np.transpose(peculiar_vy) * mask_lte, axis=0)
        fluid_vz_avg_lte = np.mean(np.transpose(peculiar_vz) * mask_lte, axis=0)

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
        # Cell Partition: arrays of shape (time, Nx, Nz) and (time, Nx)---------
        # ----------------------------------------------------------------------
        # initialize buffers to store the count 'N' and 'data_ch' of each chunk
        N_fluid_mask = np.zeros([self.chunksize, self.Nx, self.Nz], dtype=np.float32)
        N_fluid_mask_non_zero = np.zeros_like(N_fluid_mask)
        N_bulk_mask = np.zeros_like(N_fluid_mask)
        N_stable_mask = np.zeros_like(N_fluid_mask)
        N_Upper_vib_mask = np.zeros_like(N_fluid_mask)
        vx_ch =  np.zeros_like(N_fluid_mask)
        vx_ch_whole = np.zeros_like(N_fluid_mask)
        vir_ch = np.zeros_like(N_fluid_mask)
        Wxx_ch = np.zeros_like(N_fluid_mask)
        Wyy_ch = np.zeros_like(N_fluid_mask)
        Wzz_ch = np.zeros_like(N_fluid_mask)
        Wxy_ch = np.zeros_like(N_fluid_mask)
        Wxz_ch = np.zeros_like(N_fluid_mask)
        Wyz_ch = np.zeros_like(N_fluid_mask)
        den_ch = np.zeros_like(N_fluid_mask)
        jx_ch = np.zeros_like(N_fluid_mask)
        mflowrate_ch = np.zeros_like(N_fluid_mask)
        temp_ch = np.zeros_like(N_fluid_mask)
        tempx_ch = np.zeros_like(N_fluid_mask)
        tempy_ch = np.zeros_like(N_fluid_mask)
        tempz_ch = np.zeros_like(N_fluid_mask)
        temp_ch_solid = np.zeros_like(N_fluid_mask)

        N_Upper_mask = np.zeros([self.chunksize, self.Nx], dtype=np.float32)
        N_Lower_mask = np.zeros_like(N_Upper_mask)
        surfU_fx_ch = np.zeros_like(N_Lower_mask)
        surfU_fy_ch = np.zeros_like(surfU_fx_ch)
        surfU_fz_ch = np.zeros_like(surfU_fx_ch)
        surfL_fx_ch = np.zeros_like(surfU_fx_ch)
        surfL_fy_ch = np.zeros_like(surfU_fx_ch)
        surfL_fz_ch = np.zeros_like(surfU_fx_ch)
        den_bulk_ch = np.zeros_like(surfU_fx_ch)

        for i in range(self.Nx):
            for k in range(self.Nz):
        # Fluid partition-----------------------------------------
                maskx_fluid = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_fluid[i, 0, k], xx_fluid[i+1, 0, k])['mask']
                maskz_fluid = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_fluid[i, 0, k], zz_fluid[i, 0, k+1])['mask']
                mask_fluid = np.logical_and(maskx_fluid, maskz_fluid)

                # Count particles in the fluid cell
                N_fluid_mask[:, i, k] = np.sum(mask_fluid, axis=1)
                # Avoid having zero particles in the cell (avoid division by zero)
                N_fluid_mask_non_zero[:, i, k] = N_fluid_mask[:, i, k]
                Nzero_fluid = np.less(N_fluid_mask_non_zero[:, i, k], 1)
                N_fluid_mask_non_zero[Nzero_fluid, i, k] = 1

                # a = N_fluid_mask[:,i,k]
                # if rank==0:
                #     print(a[0])

        # Bulk partition-----------------------------------
                maskx_bulk = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_bulk[i, 0, k], xx_bulk[i+1, 0, k])['mask']
                maskz_bulk = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_bulk[i, 0, k], zz_bulk[i, 0, k+1])['mask']
                mask_bulk = np.logical_and(maskx_bulk, maskz_bulk)

                # Count particles in the bulk cell
                N_bulk_mask[:, i, k] = np.sum(mask_bulk, axis=1)

        # Stable partition--------------------------------------
                maskx_stable = utils.region(fluid_xcoords, fluid_xcoords,
                                    xx_stable[i, 0, k], xx_stable[i+1, 0, k])['mask']
                maskz_stable = utils.region(fluid_zcoords, fluid_zcoords,
                                    zz_stable[i, 0, k], zz_stable[i, 0, k+1])['mask']
                mask_stable = np.logical_and(maskx_stable, maskz_stable)

                # Count particles in the stable cell
                N_stable_mask[:, i, k] = np.sum(mask_stable, axis=1)
                Nzero_stable = np.less(N_stable_mask[:, i, k], 1)
                N_stable_mask[Nzero_stable, i, k] = 1

        # SurfU partition--------------------------------------------
                maskxU = utils.region(surfU_xcoords, surfU_xcoords,
                                        xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                N_Upper_mask[:, i] = np.sum(maskxU, axis=1)

        # SurfU Vibrating partition-----------------------------------
                maskxU_vib = utils.region(surfU_vib_xcoords, surfU_vib_xcoords,
                                        xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                N_Upper_vib_mask[:, i, k] = np.sum(maskxU_vib, axis=1)
                # To avoid warning with flat rigid walls
                Nzero_vib = np.less(N_Upper_vib_mask[:, i, k], 1)
                N_Upper_vib_mask[Nzero_vib, i, k] = 1

        # SurfL partition-----------------------------------
                maskxL = utils.region(surfL_xcoords, surfL_xcoords,
                                        xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                N_Lower_mask[:, i] = np.sum(maskxL, axis=1)

        # -----------------------------------------------------
        # Thermodynamic properties ----------------------------
        # -----------------------------------------------------

                # Whole fluid domain -------------------------------------------

                # Velocities in the  (Å/fs)
                vx_ch_whole[:, i, k] = np.sum(fluid_vx * mask_fluid, axis=1) / N_fluid_mask_non_zero[:, i, k]

                # Density in the whole fluid (g/(mol.Å^3))
                den_ch[:, i, k] = np.sum(mass_fluid * mask_fluid, axis=1) / vol_fluid[i,0,k]

                # Mass flux in the whole fluid (g/(mol.Å^2.fs))
                jx_ch[:, i, k] = vx_ch_whole[:, i, k] * den_ch[:, i, k]

                # Mass flow rate in the whole fluid (g/(mol.fs))
                mflowrate_ch[:, i, k] = np.sum(mass_fluid * fluid_vx * mask_fluid, axis=1) / (Lx / self.Nx)

                # Temperature (K)
                # COM velocity in the bin, shape (time,)
                uCOM = np.sum(fluid_vx_t * mask_fluid, axis=1) / (N_fluid_mask_non_zero[:, i, k])
                vCOM = np.sum(fluid_vy_t * mask_fluid, axis=1) / (N_fluid_mask_non_zero[:, i, k])
                wCOM = np.sum(fluid_vz_t * mask_fluid, axis=1) / (N_fluid_mask_non_zero[:, i, k])
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

                tempx_ch[:, i, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_vx**2 * A_per_fs_to_m_per_s**2, axis=1) / \
                                    ((N_fluid_mask[:, i, k] - 1) * sci.k)

                tempy_ch[:, i, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_vy**2 * A_per_fs_to_m_per_s**2, axis=1) / \
                                    ((N_fluid_mask[:, i, k] - 1) * sci.k)

                tempz_ch[:, i, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_vz**2 * A_per_fs_to_m_per_s**2, axis=1) / \
                                    ((N_fluid_mask[:, i, k] - 1) * sci.k)

                temp_ch[:, i, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_v**2 * A_per_fs_to_m_per_s**2 , axis=1) / \
                                    ((3 * N_fluid_mask[:, i, k] - 3) * sci.k)

                # if rank ==0:
                #     print(np.ma.masked_invalid(np.mean(temp_ch[:, i, k], axis=0)))
                #     print(N_fluid_mask[:, i, k])

                # Virial off-diagonal components (atm)
                try:
                    Wxy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 3] * mask_fluid, axis=1) / vol_fluid[i,0,k]
                    Wxz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 4] * mask_fluid, axis=1) / vol_fluid[i,0,k]
                    Wyz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 5] * mask_fluid, axis=1) / vol_fluid[i,0,k]
                except IndexError:
                    pass

                # Bulk ---------------------------------------------------------
                # Virial diagonal components (atm)
                Wxx_ch[:, i, k] = np.sum(virial[:, fluid_idx, 0] * mask_bulk, axis=1) / vol_bulk_cell[i,0,k]
                Wyy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 1] * mask_bulk, axis=1) / vol_bulk_cell[i,0,k]
                Wzz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 2] * mask_bulk, axis=1) / vol_bulk_cell[i,0,k]
                vir_ch[:, i, k] = -(Wxx_ch[:, i, k] + Wyy_ch[:, i, k] + Wzz_ch[:, i, k]) / 3.

                # Density (g/(mol.Å^3))
                den_bulk_ch[:, i] = np.sum(mass_fluid * mask_bulk, axis=1) / vol_bulk_cell[i, 0, k]

                # Velocities in the stable region (Å/fs)
                vx_ch[:, i, k] = np.sum(fluid_vx * mask_stable, axis=1) / N_stable_mask[:, i, k]

                # Whole solid --------------------------------------------------
                # Wall Forces (kcal/(mol.Å))
                surfU_fx_ch[:, i] = np.sum(surfU_fx * maskxU, axis=1)
                surfU_fy_ch[:, i] = np.sum(surfU_fy * maskxU, axis=1)
                surfU_fz_ch[:, i] = np.sum(surfU_fz * maskxU, axis=1)
                surfL_fx_ch[:, i] = np.sum(surfL_fx * maskxL, axis=1)
                surfL_fy_ch[:, i] = np.sum(surfL_fy * maskxL, axis=1)
                surfL_fz_ch[:, i] = np.sum(surfL_fz * maskxL, axis=1)

                # Upper surface ------------------------------------------------
                # Temperature (K)
                # Shape: (time,)
                uCOM_surfU_vib = np.sum(surfU_vx_vib * maskxU_vib, axis=1) / (N_Upper_vib_mask[:, i, k])
                vCOM_surfU_vib = np.sum(surfU_vy_vib * maskxU_vib, axis=1) / (N_Upper_vib_mask[:, i, k])
                wCOM_surfU_vib = np.sum(surfU_vz_vib * maskxU_vib, axis=1) / (N_Upper_vib_mask[:, i, k])
                # Shape: (Ns, time)
                peculiar_vx_surfU = np.transpose(surfU_vx_vib) - uCOM_surfU_vib
                peculiar_vy_surfU = np.transpose(surfU_vy_vib) - vCOM_surfU_vib
                peculiar_vz_surfU = np.transpose(surfU_vz_vib) - wCOM_surfU_vib
                peculiar_v_surfU_vib = np.sqrt(peculiar_vx_surfU**2 + peculiar_vy_surfU**2 + peculiar_vz_surfU**2)
                # Shape: (time, Ns)
                peculiar_v_surfU_vib = np.transpose(peculiar_v_surfU_vib) * maskxU_vib

                temp_ch_solid[:, i, k] =  np.sum(mass_surfU_vib * sci.gram / sci.N_A * \
                                        peculiar_v_surfU_vib**2 * A_per_fs_to_m_per_s**2 , axis=1) / \
                                        ((3 * N_Upper_vib_mask[:, i, k] - 3) * sci.k)  # Kelvin

        return {'cell_lengths': cell_lengths_updated,
                'gap_height': gap_height,
                'bulkStartZ_time': bulkStartZ_time,
                'bulkEndZ_time': bulkEndZ_time,
                'com': comZ,
                'Nf': Nf,
                'Nm': Nm,
                'totVi': totVi,
                'del_totVi': del_totVi,
                'je_x': je_x,
                'je_y': je_y,
                'je_z': je_z,
                'fluxes': fluxes,
                'fluid_vx_avg_lte': fluid_vx_avg_lte,
                'fluid_vy_avg_lte': fluid_vy_avg_lte,
                'fluid_vz_avg_lte': fluid_vz_avg_lte,
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
                'vx_ch': vx_ch,
                'surfU_fx_ch':surfU_fx_ch,
                'surfU_fy_ch':surfU_fy_ch,
                'surfU_fz_ch':surfU_fz_ch,
                'surfL_fx_ch':surfL_fx_ch,
                'surfL_fy_ch':surfL_fy_ch,
                'surfL_fz_ch':surfL_fz_ch,
                'temp_ch_solid': temp_ch_solid}
