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

    def __init__(self, data, start, end, Nx, Nz, mf, A_per_molecule, fluid, Ny=1, tessellate=0, TW_interface=1):

        self.data = data
        self.start, self.end = start, end
        self.chunksize = self.end - self.start
        self.Nx, self.Ny, self.Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)
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


    def get_chunks(self, stable_start, stable_end, pump_start, pump_end, nx, ny, nz, step):
        """
        Partitions the box into regions (solid and fluid) as well as chunks in
        those regions
        Parameters
        ----------
        stable_start: float
                Multiple of Lx, where the stable region (for sampling) starts
        stable_end: float
            Multiple of Lx, where the stable region (for sampling) ends
        pump_start: float
                Multiple of Lx, where the pump region (pertrubation field) starts
        pump_end: float
                Multiple of Lx, where the stable region (pertrubation field) ends
        Returns
        ----------
        Thermodynamic quantities (arrays) with different shapes. The 4 shapes are
        (time,)
            time series of the quantity
        (Nf,)
            time averaged quantity (here velocity) for each fluid particle
        (time,Nx)
            time series of the chunks along the x-direction
        (time,Nx,Nz)
            time series of the chunks along the x and the z-directions.
        """

        fluid_idx, solid_start, Nf, Nm = self.get_indices()
        Lx, Ly, Lz = self.get_dimensions()

        # Discrete Wavevectors
        nx = np.arange(0, nx, step)
        ny = np.arange(0, ny, step)
        nz = np.arange(0, nz, step)

        # Wavevectors
        kx = 2. * np.pi * nx / Lx
        ky = 2. * np.pi * ny / Lx   # For 1:1 aspect ratio

        # Array dimensions: (time, idx, dimesnion)
        try:
            coords_data = self.data.variables["f_position"]
        except KeyError:
            coords_data = self.data.variables["coordinates"]
        try:
            vels_data = self.data.variables["f_velocity"]
        except KeyError:
            vels_data = self.data.variables["velocities"]
        # The unaveraged quantity is that which is dumped by LAMMPS every N timesteps
        # i.e. it is a snapshot of the system at this timestep.
        # As opposed to averaged quantities which are the average of a few previous timesteps
        coords_data_unavgd = self.data.variables["coordinates"]   # Used for ACF calculation
        vels_data_unavgd = self.data.variables["velocities"]     # Used for temp. calculation

        # In some trajectories, virial is not computed (faster simulation)
        # Fluid Virial Pressure ----------------------------------
        try:
            voronoi_vol_data = self.data.variables["f_Vi_avg"]
            voronoi_vol = np.array(voronoi_vol_data[self.start:self.end]).astype(np.float32)
        except KeyError:
            if rank == 0:
                logger.info('Voronoi volume was not computed during LAMMPS Run!')
            pass

        try:
            virial_data = self.data.variables["f_Wi_avg"]
            virial = np.array(virial_data[self.start:self.end]).astype(np.float32)
        except KeyError:
            if rank == 0:
                logger.info('Virial was not computed during LAMMPS Run!')
            pass


        # For heat flux computation (total energy * velocity) and (virial * velocity) were computed
        try:
            ev_data = self.data.variables["f_ev"]
            centroid_vir_data = self.data.variables["f_sv"]
            ev = np.array(ev_data[self.start:self.end]).astype(np.float32)
            centroid_vir = np.array(centroid_vir_data[self.start:self.end]).astype(np.float32)

            # in kcal/mol * A/fs
            fluid_evx, fluid_evy, fluid_evz = ev[:, fluid_idx, 0], \
                                              ev[:, fluid_idx, 1], \
                                              ev[:, fluid_idx, 2]

            # in Kcal/mol * A/fs
            fluid_svx, fluid_svy, fluid_svz = centroid_vir[:, fluid_idx, 0] ,\
                                              centroid_vir[:, fluid_idx, 1] ,\
                                              centroid_vir[:, fluid_idx, 2]

        except KeyError:
            pass

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

        # Velocities ------------------------------------------------
        vels = np.array(vels_data[self.start:self.end]).astype(np.float32)

        # Velocity of each atom
        fluid_vx, fluid_vy, fluid_vz = vels[:, fluid_idx, 0], \
                                       vels[:, fluid_idx, 1], \
                                       vels[:, fluid_idx, 2]

        # For the velocity distribution
        fluid_vx_avg = np.mean(fluid_vx, axis=0)
        fluid_vy_avg = np.mean(fluid_vy, axis=0)
        fluid_vz_avg = np.mean(fluid_vz, axis=0)

        # Unaveraged velocities
        vels_t = np.array(vels_data_unavgd[self.start:self.end]).astype(np.float32)

        fluid_vx_t, fluid_vy_t, fluid_vz_t = vels_t[:, fluid_idx, 0], \
                                             vels_t[:, fluid_idx, 1], \
                                             vels_t[:, fluid_idx, 2]

        # Wall Stresses -------------------------------------------------
        if solid_start != None:
            # Forces -----------------------------------
            try:
                forcesU = self.data.variables["f_fAtomU_avg"]
                forcesL = self.data.variables["f_fAtomL_avg"]

                forcesU_data = np.array(forcesU[self.start:self.end]).astype(np.float32)
                forcesL_data = np.array(forcesL[self.start:self.end]).astype(np.float32)
                solid_forces = 1
            except KeyError:
                solid_forces = 0

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

            if solid_forces !=0:
                surfU_fx,surfU_fy,surfU_fz = forcesU_data[:, solid_start:, 0][:,surfU_indices], \
                                             forcesU_data[:, solid_start:, 1][:,surfU_indices], \
                                             forcesU_data[:, solid_start:, 2][:,surfU_indices]

                surfL_fx,surfL_fy,surfL_fz = forcesL_data[:, solid_start:, 0][:,surfL_indices], \
                                             forcesL_data[:, solid_start:, 1][:,surfL_indices], \
                                             forcesL_data[:, solid_start:, 2][:,surfL_indices]
            else:
                surfU_fx, surfU_fy, surfU_fz = 0, 0, 0
                surfL_fx, surfL_fy, surfL_fz = 0, 0, 0

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

            # Velocities --------------------------------
            surfU_vx_t, surfU_vy_t, surfU_vz_t = vels_t[:, solid_start:, 0][:,surfU_vib_indices], \
                                                 vels_t[:, solid_start:, 1][:,surfU_vib_indices], \
                                                 vels_t[:, solid_start:, 2][:,surfU_vib_indices]

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
        if solid_start != None:
            # Pump --------------------------------------------
            pumpStartX, pumpEndX = pump_start * Lx, pump_end * Lx
            pump_length, pump_region, pump_xcoords, pump_vol, pump_N = \
                itemgetter('interval', 'mask', 'data', 'vol', 'count')\
                (utils.region(fluid_xcoords, fluid_xcoords, pumpStartX, pumpEndX,
                                        ylength=Ly, length=avg_gap_height_conv))

            # Mass flux in the pump region
            # Velocity of each molecule
            vels_pump = fluid_vx * pump_region / self.A_per_molecule
            mflux_pump = (np.sum(vels_pump, axis=1) * (sci.angstrom/fs_to_ns) * \
                         (self.mf/sci.N_A)) / (pump_vol * (sci.angstrom)**3)      # g/m2.ns
            mflowrate_pump = (np.sum(vels_pump, axis=1) * (sci.angstrom/fs_to_ns) * \
                            (self.mf/sci.N_A)) / (pump_length * sci.angstrom)       # g/ns

            # Bulk --------------------------------------------
            # avg_surfL_end_conv = np.mean(comm.allgather(np.mean(surfL_end_conv)))
            bulkStartZ, bulkEndZ = (0.4 * Lz) + 1, (0.6 * Lz) + 1     # Boffset is 1 Angstrom
            bulk_height, bulk_region, bulk_zcoords, bulk_vol, bulk_N = \
                itemgetter('interval', 'mask', 'data', 'vol', 'count')\
                (utils.region(fluid_zcoords, fluid_zcoords, bulkStartZ, bulkEndZ,
                                                ylength=Ly, length=Lx))
            bulkStartZ_time = 0.4 * (surfU_end - surfL_begin) + 1
            bulkEndZ_time = 0.6 * (surfU_end - surfL_begin) + 1

            try: # If Vornoi tessellation was performed in LAMMPS
                # Voronoi volumes of atoms in the bulk region in each timestep
                Vi = voronoi_vol[:, fluid_idx] * bulk_region
                # Voronoi volume of the whole bulk region in each timestep
                totVi = np.sum(Vi, axis=1)
            except UnboundLocalError:
                totVi = 0
                pass

            # Perform Delaunay triangulation: (Could be useful for cavitation)
            if self.tessellate==1:
                bulk_xcoords = fluid_xcoords * bulk_region
                bulk_ycoords = fluid_ycoords * bulk_region
                bulk_zcoords = fluid_zcoords * bulk_region

                bulk_coords = np.zeros((self.chunksize, Nf, 3))
                bulk_coords[:,:,0], bulk_coords[:,:,1], bulk_coords[:,:,2] = \
                bulk_xcoords, bulk_ycoords, bulk_zcoords

                mytotVi = np.zeros([self.chunksize], dtype=np.float32)
                for i in range(self.chunksize):
                    mytotVi[i] = tessellation.delaunay_volumes(bulk_coords[i])
            else:
                mytotVi = 0

            # Stable ---------------------------------
            stableStartX, stableEndX = stable_start * Lx, stable_end * Lx
            stable_length, stable_region, stable_xcoords, stable_vol, stable_N = \
                itemgetter('interval', 'mask', 'data', 'vol', 'count')\
                (utils.region(fluid_xcoords, fluid_xcoords, stableStartX, stableEndX,
                                                ylength=Ly, length=avg_gap_height_div))

            # Mass flux in the stable region -------------------
            vels_stable = fluid_vx * stable_region / self.A_per_molecule
            mflux_stable = (np.sum(vels_stable, axis=1) * (sci.angstrom/fs_to_ns) * \
                           (self.mf/sci.N_A))/ (stable_vol * (sci.angstrom)**3)      # g/m2.ns
            mflowrate_stable = (np.sum(vels_stable, axis=1) * (sci.angstrom/fs_to_ns) * \
                            (self.mf/sci.N_A)) / (stable_length * sci.angstrom)       # g/ns

            fluxes = [mflux_pump, mflowrate_pump, mflux_stable, mflowrate_stable]

            # Heat flux --------------

            try:
                je_x = np.sum((fluid_evx*stable_region) + (fluid_svx*stable_region), axis=1)
                je_y = np.sum((fluid_evy*stable_region) + (fluid_svy*stable_region), axis=1)
                je_z = np.sum((fluid_evz*stable_region) + (fluid_svz*stable_region), axis=1)
            except UnboundLocalError:
                je_x, je_y, je_z = None, None, None

        else:
            bulkStartZ_time = None
            bulkEndZ_time = None
            fluxes = None
            try:
                # Voronoi volumes of atoms in the bulk region in each timestep
                Vi = voronoi_vol[:, fluid_idx]
                # Voronoi volume of the whole bulk region in each timestep
                totVi = np.sum(Vi, axis=1)
            except UnboundLocalError:
                totVi = 0
                pass

            # Perform Delaunay triangulation: (Could be useful for cavitation)
            if self.tessellate==1:
                bulk_xcoords = fluid_xcoords * bulk_region
                bulk_ycoords = fluid_ycoords * bulk_region
                bulk_zcoords = fluid_zcoords * bulk_region

                bulk_coords = np.zeros((self.chunksize, Nf, 3))
                bulk_coords[:,:,0], bulk_coords[:,:,1], bulk_coords[:,:,2] = \
                bulk_xcoords, bulk_ycoords, bulk_zcoords

                mytotVi = np.zeros([self.chunksize], dtype=np.float32)
                for i in range(self.chunksize):
                    mytotVi[i] = tessellation.delaunay_volumes(bulk_coords[i])
            else:
                mytotVi = 0

            # Heat flux --------------
            try:
                je_x = np.sum((fluid_evx + fluid_svx), axis=1)
                je_y = np.sum((fluid_evy + fluid_svy), axis=1)
                je_z = np.sum((fluid_evz + fluid_svz), axis=1)
            except UnboundLocalError:
                je_x, je_y, je_z = None, None, None

        # The Grid -------------------------------------------------------------
        # ----------------------------------------------------------------------
        dim = np.array([self.Nx, self.Ny, self.Nz])
        dim_reciproc = np.array([kx, ky, kz], dtype=object)

        # Whole cell --------------------------------------------
        # The minimum position has to be added to have equal Number of solid atoms in each chunk
        if solid_start != None:
            bounds = [np.arange(dim[i] + 1) / dim[i] * cell_lengths_updated[i] + fluid_min[i]
                            for i in range(3)]
            xx, yy, zz, vol_cell = utils.bounds(bounds[0], bounds[1], bounds[2])

            # Fluid ------------------------------------------------
            bounds_fluid = [np.arange(dim[i] + 1) / dim[i] * fluid_lengths[i] + fluid_min[i]
                            for i in range(3)]
            xx_fluid, yy_fluid, zz_fluid, vol_fluid = utils.bounds(bounds_fluid[0],
                                                      bounds_fluid[1], bounds_fluid[2])

            # Bulk --------------------------------------------
            bulkRangeZ = np.arange(dim[2] + 1) / dim[2] * bulk_height + bulkStartZ
            bounds_bulk = [bounds_fluid[0], bounds_fluid[1], bulkRangeZ]
            xx_bulk, yy_bulk, zz_bulk, vol_bulk_cell = utils.bounds(bounds_bulk[0],
                                                       bounds_bulk[1], bounds_bulk[2])

            # Stable --------------------------------------------
            stableRangeX = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX
            bounds_stable = [stableRangeX, bounds_fluid[1], bounds_fluid[2]]
            xx_stable, yy_stable, zz_stable, vol_stable_cell = utils.bounds(bounds_stable[0],
                                                                bounds_stable[1], bounds_stable[2])


            # Measure the velocity profile evolution along the stream -----------------------
            # Bounds R1
            R1Range = np.arange(dim[0] + 1) / dim[0] * 0.2 * Lx
            bounds_R1 = [R1Range, np.array([bounds[1]]), np.array([bounds_fluid[2]])]

            xx_R1, yy_R1, zz_R1 = np.meshgrid(bounds_R1[0], bounds_R1[1], bounds_R1[2])
            xx_R1 = np.transpose(xx_R1, (1, 0, 2))
            yy_R1 = np.transpose(yy_R1, (1, 0, 2))
            zz_R1 = np.transpose(zz_R1, (1, 0, 2))

            # Bounds R2
            R2Range = np.arange(dim[0] + 1) / dim[0] * 0.2 * Lx + bounds_R1[0][-1]
            bounds_R2 = [R2Range, np.array([bounds[1]]), np.array([bounds_fluid[2]])]

            xx_R2, yy_R2, zz_R2 = np.meshgrid(bounds_R2[0], bounds_R2[1], bounds_R2[2])
            xx_R2 = np.transpose(xx_R2, (1, 0, 2))
            yy_R2 = np.transpose(yy_R2, (1, 0, 2))
            zz_R2 = np.transpose(zz_R2, (1, 0, 2))

            # Bounds R3
            R3Range = np.arange(dim[0] + 1) / dim[0] * 0.2 * Lx + bounds_R2[0][-1]
            bounds_R3 = [R3Range, np.array([bounds[1]]), np.array([bounds_fluid[2]])]

            xx_R3, yy_R3, zz_R3 = np.meshgrid(bounds_R3[0], bounds_R3[1], bounds_R3[2])
            xx_R3 = np.transpose(xx_R3, (1, 0, 2))
            yy_R3 = np.transpose(yy_R3, (1, 0, 2))
            zz_R3 = np.transpose(zz_R3, (1, 0, 2))

            # Bounds R4
            R4Range = np.arange(dim[0] + 1) / dim[0] * 0.2 * Lx + bounds_R3[0][-1]
            bounds_R4 = [R4Range, np.array([bounds[1]]), np.array([bounds_fluid[2]])]

            xx_R4, yy_R4, zz_R4 = np.meshgrid(bounds_R4[0], bounds_R4[1], bounds_R4[2])
            xx_R4 = np.transpose(xx_R4, (1, 0, 2))
            yy_R4 = np.transpose(yy_R4, (1, 0, 2))
            zz_R4 = np.transpose(zz_R4, (1, 0, 2))

            # Bounds R5
            R5Range = np.arange(dim[0] + 1) / dim[0] * 0.2 * Lx + bounds_R4[0][-1]
            bounds_R5 = [R5Range, np.array([bounds[1]]), np.array([bounds_fluid[2]])]

            xx_R5, yy_R5, zz_R5 = np.meshgrid(bounds_R5[0], bounds_R5[1], bounds_R5[2])
            xx_R5 = np.transpose(xx_R5, (1, 0, 2))
            yy_R5 = np.transpose(yy_R5, (1, 0, 2))
            zz_R5 = np.transpose(zz_R5, (1, 0, 2))

            #---------------------------------------------------------------------

        else:
            # Bounds should change for example in NPT simulations
            # (NEEDED especially when the Box volume changes e.g. NPT)------------------------------------------------
            bounds = [np.arange(dim[i] + 1) / dim[i] * fluid_lengths[i] + fluid_min[i]
                            for i in range(3)]

            xx, yy, zz, vol_cell = utils.bounds(bounds[0], bounds[1], bounds[2])

        # -------------------------------------------------------
        # Cell Partition ---------------------------------------
        # -------------------------------------------------------
        # initialize buffers to store the count 'N' and 'data_ch' of each chunk
        N_fluid_mask = np.zeros([self.chunksize, self.Nx, self.Nz], dtype=np.float32)
        N_stable_mask = np.zeros_like(N_fluid_mask)
        N_bulk_mask = np.zeros_like(N_fluid_mask)
        N_Upper_vib_mask = np.zeros_like(N_fluid_mask)

        N_R1 = np.zeros_like(N_fluid_mask)
        N_R2 = np.zeros_like(N_fluid_mask)
        N_R3 = np.zeros_like(N_fluid_mask)
        N_R4 = np.zeros_like(N_fluid_mask)
        N_R5 = np.zeros_like(N_fluid_mask)

        vx_ch = np.zeros([self.chunksize, self.Nx, self.Nz], dtype=np.float32)

        vx_ch_whole = np.zeros_like(vx_ch)
        vx_R1 = np.zeros_like(vx_ch)
        vx_R2 = np.zeros_like(vx_ch)
        vx_R3 = np.zeros_like(vx_ch)
        vx_R4 = np.zeros_like(vx_ch)
        vx_R5 = np.zeros_like(vx_ch)

        vir_ch = np.zeros_like(vx_ch)
        Wxx_ch = np.zeros_like(vx_ch)
        Wyy_ch = np.zeros_like(vx_ch)
        Wzz_ch = np.zeros_like(vx_ch)

        Wxy_ch = np.zeros_like(vx_ch)
        Wxz_ch = np.zeros_like(vx_ch)
        Wyz_ch = np.zeros_like(vx_ch)

        den_ch = np.zeros_like(vx_ch)

        # rho_kx_ch = np.zeros([self.chunksize, len(kx)] , dtype=np.complex64)
        sf_x = np.zeros([self.chunksize, len(kx)] , dtype=np.complex64)
        sf_y = np.zeros([self.chunksize, len(ky)] , dtype=np.complex64)
        sf_x_solid = np.zeros([self.chunksize, len(kx)] , dtype=np.complex64)
        sf_y_solid = np.zeros([self.chunksize, len(ky)] , dtype=np.complex64)
        rho_k = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)
        sf = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)
        sf_solid = np.zeros([self.chunksize, len(kx), len(ky)] , dtype=np.complex64)

        jx_ch = np.zeros_like(vx_ch)
        mflowrate_ch = np.zeros_like(vx_ch)
        temp_ch = np.zeros_like(vx_ch)
        tempx_ch = np.zeros_like(vx_ch)
        tempy_ch = np.zeros_like(vx_ch)
        tempz_ch = np.zeros_like(vx_ch)
        temp_ch_solid = np.zeros_like(vx_ch)
        uCOMx = np.zeros_like(vx_ch)

        N_Upper_mask = np.zeros([self.chunksize, self.Nx], dtype=np.float32)
        N_Lower_mask = np.zeros_like(N_Upper_mask)

        surfU_fx_ch = np.zeros([self.chunksize, self.Nx], dtype=np.float32)
        surfU_fy_ch = np.zeros_like(surfU_fx_ch)
        surfU_fz_ch = np.zeros_like(surfU_fx_ch)
        surfL_fx_ch = np.zeros_like(surfU_fx_ch)
        surfL_fy_ch = np.zeros_like(surfU_fx_ch)
        surfL_fz_ch = np.zeros_like(surfU_fx_ch)
        den_bulk_ch = np.zeros_like(surfU_fx_ch)

        if solid_start != None:

            # velocity distribution in the channel center to test for LTE
            xcom, ycom, zcom = cell_lengths_updated[0]/2, cell_lengths_updated[1]/2, cell_lengths_updated[2]/2
            # LTE region: 5x5x5 Ang^3 in the middle of the box
            maskx_lte = utils.region(fluid_xcoords, fluid_xcoords, xcom-2.5, xcom+2.5)['mask']
            masky_lte = utils.region(fluid_ycoords, fluid_ycoords, ycom-2.5, ycom+2.5)['mask']
            maskz_lte = utils.region(fluid_zcoords, fluid_zcoords, zcom-2.5, zcom+2.5)['mask']
            mask_xy_lte = np.logical_and(maskx_lte, masky_lte)
            mask_lte = np.logical_and(mask_xy_lte, maskz_lte)
            N_lte = np.sum(mask_lte, axis=1) # no. of atoms in the lte box in every timestep
            Nzero_lte = np.less(N_lte, 1)
            N_lte[Nzero_lte] = 1

            # For the velocity distribution
            uCOMx_lte = np.sum(fluid_vx_t * mask_lte, axis=1) / N_lte # timeseries
            uCOMy_lte = np.sum(fluid_vy_t * mask_lte, axis=1) / N_lte # timeseries
            uCOMz_lte = np.sum(fluid_vz_t * mask_lte, axis=1) / N_lte # timeseries

            # Remove the streaming velocity from the lab frame velocity to get the thermal/peculiar velocity
            peculiar_vx = np.transpose(fluid_vx_t) - uCOMx_lte
            peculiar_vy = np.transpose(fluid_vy_t) - uCOMy_lte
            peculiar_vz = np.transpose(fluid_vz_t) - uCOMz_lte

            fluid_vx_avg_lte = np.mean(np.transpose(peculiar_vx)*mask_lte, axis=0) #np.mean(fluid_vx*mask_lte, axis=0)
            fluid_vy_avg_lte = np.mean(np.transpose(peculiar_vy)*mask_lte, axis=0)
            fluid_vz_avg_lte = np.mean(np.transpose(peculiar_vz)*mask_lte, axis=0)

            # Structure Factor calculation
            maskx_layer = utils.region(fluid_xcoords, fluid_xcoords, 0, Lx)['mask']
            masky_layer = utils.region(fluid_ycoords, fluid_ycoords, 0, Ly)['mask']
            # maskz_layer = utils.region(fluid_zcoords, fluid_zcoords, 0, 10)['mask']       # Rigid walls
            maskz_layer = utils.region(fluid_zcoords, fluid_zcoords, 14, 18)['mask']        # Vibrating walls

            maskx_layer_solid = utils.region(solid_xcoords, solid_xcoords, 0, Lx)['mask']
            masky_layer_solid = utils.region(solid_ycoords, solid_ycoords, 0, Ly)['mask']
            # maskz_layer_solid = utils.region(solid_zcoords, solid_zcoords, 7, 10)['mask']     # Rigid walls
            maskz_layer_solid = utils.region(solid_zcoords, solid_zcoords, 11, 14)['mask']       # Vibrating walls

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
                sf_x[:, i] = (np.sum(np.exp(1.j * kx[i] * fluid_xcoords_unavgd * mask_layer), axis=1))**2 / N_layer_mask
                sf_x_solid[:, i] = (np.sum(np.exp(1.j * kx[i] * solid_xcoords_unavgd * mask_layer_solid), axis=1))**2 / N_layer_mask_solid
            for k in range(len(ky)):
                sf_y[:, k] = (np.sum(np.exp(1.j * ky[k] * fluid_ycoords_unavgd * mask_layer), axis=1))**2 / N_layer_mask
                sf_y_solid[:, k] = (np.sum(np.exp(1.j * ky[k] * solid_ycoords_unavgd * mask_layer_solid), axis=1))**2 / N_layer_mask_solid

            for i in range(len(kx)):
                for k in range(len(ky)):
                    # Fourier components of the density
                    rho_k[:, i, k] = np.sum( np.exp(1.j * (kx[i]*fluid_xcoords_unavgd*mask_layer +
                                        ky[k]*fluid_ycoords_unavgd*mask_layer)) )
                    sf[:, i, k] =  (np.sum( np.exp(1.j * (kx[i]*fluid_xcoords_unavgd*mask_layer +
                                        ky[k]*fluid_ycoords_unavgd*mask_layer) ) , axis=1))**2 / N_layer_mask
                    sf_solid[:, i, k] =  (np.sum( np.exp(1.j * (kx[i]*solid_xcoords_unavgd*mask_layer_solid +
                                        ky[k]*solid_ycoords_unavgd*mask_layer_solid) ) , axis=1))**2 / N_layer_mask_solid

            for i in range(self.Nx):
                for k in range(self.Nz):
            # Fluid -----------------------------------------
                    maskx_fluid = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_fluid[i, 0, k], xx_fluid[i+1, 0, k])['mask']
                    maskz_fluid = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_fluid[i, 0, k], zz_fluid[i, 0, k+1])['mask']
                    mask_fluid = np.logical_and(maskx_fluid, maskz_fluid)

                    # Count particles in the fluid cell
                    N_fluid_mask[:, i, k] = np.sum(mask_fluid, axis=1)
                    # Avoid having zero particles in the cell
                    Nzero_fluid = np.less(N_fluid_mask[:, i, k], 1)
                    N_fluid_mask[Nzero_fluid, i, k] = 1

            # Stable --------------------------------------
                    maskx_stable = utils.region(fluid_xcoords, fluid_xcoords,
                                        xx_stable[i, 0, k], xx_stable[i+1, 0, k])['mask']
                    maskz_stable = utils.region(fluid_zcoords, fluid_zcoords,
                                        zz_stable[i, 0, k], zz_stable[i, 0, k+1])['mask']
                    mask_stable = np.logical_and(maskx_stable, maskz_stable)

                    # Count particles in the stable cell
                    N_stable_mask[:, i, k] = np.sum(mask_stable, axis=1)
                    Nzero_stable = np.less(N_stable_mask[:, i, k], 1)
                    N_stable_mask[Nzero_stable, i, k] = 1

            # Bulk -----------------------------------
                    maskx_bulk = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_bulk[i, 0, k], xx_bulk[i+1, 0, k])['mask']
                    maskz_bulk = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_bulk[i, 0, k], zz_bulk[i, 0, k+1])['mask']
                    mask_bulk = np.logical_and(maskx_bulk, maskz_bulk)

                    # Count particles in the bulk cell
                    N_bulk_mask[:, i, k] = np.sum(mask_bulk, axis=1)
                    Nzero_bulk = np.less(N_bulk_mask[:, i, k], 1)
                    N_bulk_mask[Nzero_bulk, i, k] = 1

            # SurfU -----------------------------------
                    maskxU = utils.region(surfU_xcoords, surfU_xcoords,
                                            xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                    N_Upper_mask[:, i] = np.sum(maskxU, axis=1)

            # Vibrating atoms
                    maskxU_vib = utils.region(surfU_vib_xcoords, surfU_vib_xcoords,
                                            xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                    N_Upper_vib_mask[:, i, k] = np.sum(maskxU_vib, axis=1)
                    # To avoid warning with flat rigid walls
                    Nzero_vib = np.less(N_Upper_vib_mask[:, i, k], 1)
                    N_Upper_vib_mask[Nzero_vib, i, k] = 1

            # SurfL -----------------------------------
                    maskxL = utils.region(surfL_xcoords, surfL_xcoords,
                                            xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                    N_Lower_mask[:, i] = np.sum(maskxL, axis=1)


            # Regions along the stream ------------------------------------------

                    # Region R1
                    maskx_R1 = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_R1[i, 0, k], xx_R1[i+1, 0, k])['mask']

                    maskz_R1 = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_R1[i, 0, k], zz_R1[i, 0, k+1])['mask']

                    mask_R1 = np.logical_and(maskx_R1, maskz_R1)

                    # Region R2
                    maskx_R2 = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_R2[i, 0, k], xx_R2[i+1, 0, k])['mask']

                    maskz_R2 = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_R2[i, 0, k], zz_R2[i, 0, k+1])['mask']

                    mask_R2 = np.logical_and(maskx_R2, maskz_R2)

                    # Region R3
                    maskx_R3 = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_R3[i, 0, k], xx_R3[i+1, 0, k])['mask']

                    maskz_R3 = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_R3[i, 0, k], zz_R3[i, 0, k+1])['mask']

                    mask_R3 = np.logical_and(maskx_R3, maskz_R3)

                    # Region R4
                    maskx_R4 = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_R4[i, 0, k], xx_R4[i+1, 0, k])['mask']

                    maskz_R4 = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_R4[i, 0, k], zz_R4[i, 0, k+1])['mask']

                    mask_R4 = np.logical_and(maskx_R4, maskz_R4)

                    # Region R5
                    maskx_R5 = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx_R5[i, 0, k], xx_R5[i+1, 0, k])['mask']

                    maskz_R5 = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz_R5[i, 0, k], zz_R5[i, 0, k+1])['mask']

                    mask_R5 = np.logical_and(maskx_R5, maskz_R5)

                    N_R1[:, i, k] = np.sum(mask_R1, axis=1)
                    Nzero_R1 = np.less(N_R1[:, i, k], 1)
                    N_R1[Nzero_R1, i, k] = 1

                    N_R2[:, i, k] = np.sum(mask_R2, axis=1)
                    Nzero_R2 = np.less(N_R2[:, i, k], 1)
                    N_R2[Nzero_R2, i, k] = 1

                    N_R3[:, i, k] = np.sum(mask_R3, axis=1)
                    Nzero_R3 = np.less(N_R3[:, i, k], 1)
                    N_R3[Nzero_R3, i, k] = 1

                    N_R4[:, i, k] = np.sum(mask_R4, axis=1)
                    Nzero_R4 = np.less(N_R4[:, i, k], 1)
                    N_R4[Nzero_R4, i, k] = 1

                    N_R5[:, i, k] = np.sum(mask_R5, axis=1)
                    Nzero_R5 = np.less(N_R5[:, i, k], 1)
                    N_R5[Nzero_R5, i, k] = 1


            # -----------------------------------------------------
            # Cell Averages ---------------------------------------
            # -----------------------------------------------------
                    # Velocities in the stable region
                    vx_ch[:, i, k] = np.sum(fluid_vx * mask_stable, axis=1) / N_stable_mask[:, i, k]
                    # Velocities in the whole domain
                    vx_ch_whole[:, i, k] = np.sum(fluid_vx * mask_fluid, axis=1) / N_fluid_mask[:, i, k]

                    vx_R1[:, i, k] = np.sum(fluid_vx * mask_R1, axis=1) / (N_R1[:, i, k])
                    vx_R2[:, i, k] = np.sum(fluid_vx * mask_R2, axis=1) / (N_R2[:, i, k])
                    vx_R3[:, i, k] = np.sum(fluid_vx * mask_R3, axis=1) / (N_R3[:, i, k])
                    vx_R4[:, i, k] = np.sum(fluid_vx * mask_R4, axis=1) / (N_R4[:, i, k])
                    vx_R5[:, i, k] = np.sum(fluid_vx * mask_R5, axis=1) / (N_R5[:, i, k])


                    # Density (whole fluid) ----------------------------
                    den_ch[:, i, k] = (N_fluid_mask[:, i, k] / self.A_per_molecule) / vol_fluid[i, 0, k]

                    # Mass flux (whole fluid) ----------------------------
                    jx_ch[:, i, k] = vx_ch_whole[:, i, k] * den_ch[:, i, k]

                    # Mass flow rate (whole fluid) -----------------------
                    mflowrate_ch[:, i, k] = (N_fluid_mask[:, i, k] / self.A_per_molecule) * vx_ch_whole[:, i, k]

                    # Temperature ----------------------------
                    # COM velocity in the bin
                    uCOMx = np.sum(fluid_vx_t * mask_fluid, axis=1) / (N_fluid_mask[:, i, k])
                    uCOMy = np.sum(fluid_vy_t * mask_fluid, axis=1) / (N_fluid_mask[:, i, k])
                    uCOMz = np.sum(fluid_vz_t * mask_fluid, axis=1) / (N_fluid_mask[:, i, k])

                    # In the upper surface
                    uCOMx_surfU = np.sum(surfU_vx_t * maskxU_vib, axis=1) / (N_Upper_vib_mask[:, i, k])
                    uCOMy_surfU = np.sum(surfU_vy_t * maskxU_vib, axis=1) / (N_Upper_vib_mask[:, i, k])
                    uCOMz_surfU = np.sum(surfU_vz_t * maskxU_vib, axis=1) / (N_Upper_vib_mask[:, i, k])

                    # Remove the streaming velocity from the lab frame velocity to get the thermal/peculiar velocity
                    peculiar_vx = np.transpose(fluid_vx_t) - uCOMx
                    peculiar_vy = np.transpose(fluid_vy_t) - uCOMy
                    peculiar_vz = np.transpose(fluid_vz_t) - uCOMz

                    # In the upper surface
                    peculiar_vx_surfU = np.transpose(surfU_vx_t) #- uCOMx_surfU
                    peculiar_vy_surfU = np.transpose(surfU_vy_t) #- uCOMy_surfU
                    peculiar_vz_surfU = np.transpose(surfU_vz_t) #- uCOMz_surfU

                    peculiar_v = np.sqrt(peculiar_vx**2 + peculiar_vy**2 + peculiar_vz**2)
                    peculiar_v = np.transpose(peculiar_v) * mask_fluid / self.A_per_molecule

                    peculiar_vxa =  np.transpose(peculiar_vx) * mask_fluid / self.A_per_molecule
                    tempx_ch[:, i, k] = ((self.mf * sci.gram / sci.N_A) * \
                                        np.sum(peculiar_vxa**2 , axis=1) * \
                                        A_per_fs_to_m_per_s**2)  / \
                                        (N_fluid_mask[:, i, k] * sci.k /self.A_per_molecule )  # Kelvin

                    peculiar_vya =  np.transpose(peculiar_vy) * mask_fluid / self.A_per_molecule
                    tempy_ch[:, i, k] = ((self.mf * sci.gram / sci.N_A) * \
                                        np.sum(peculiar_vya**2 , axis=1) * \
                                        A_per_fs_to_m_per_s**2)  / \
                                        (N_fluid_mask[:, i, k] * sci.k /self.A_per_molecule )  # Kelvin

                    peculiar_vza =  np.transpose(peculiar_vz) * mask_fluid / self.A_per_molecule
                    tempz_ch[:, i, k] = ((self.mf * sci.gram / sci.N_A) * \
                                        np.sum(peculiar_vza**2 , axis=1) * \
                                        A_per_fs_to_m_per_s**2)  / \
                                        (N_fluid_mask[:, i, k] * sci.k /self.A_per_molecule )  # Kelvin

                    peculiar_v_solid = np.sqrt(peculiar_vx_surfU**2 + peculiar_vy_surfU**2 + peculiar_vz_surfU**2)
                    peculiar_v_solid = np.transpose(peculiar_v_solid) * maskxU_vib

                    temp_ch[:, i, k] = ((self.mf * sci.gram / sci.N_A) * \
                                        np.sum(peculiar_v**2 , axis=1) * \
                                        A_per_fs_to_m_per_s**2)  / \
                                        (3 * N_fluid_mask[:, i, k] * sci.k /self.A_per_molecule)  # Kelvin

                    temp_ch_solid[:, i, k] = ((self.ms * sci.gram / sci.N_A) * \
                                        np.sum(peculiar_v_solid**2 , axis=1) * \
                                        A_per_fs_to_m_per_s**2)  / \
                                        (3 * N_Upper_vib_mask[:, i, k] * sci.k)  # Kelvin


                    # Virial pressure--------------------------------------
                    # Simulations with virial calculation
                    try:
                        Wxx_ch[:, i, k] = np.sum(virial[:, fluid_idx, 0] * mask_bulk, axis=1) / vol_bulk_cell[i,0,k]
                        Wyy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 1] * mask_bulk, axis=1) / vol_bulk_cell[i,0,k]
                        Wzz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 2] * mask_bulk, axis=1) / vol_bulk_cell[i,0,k]
                        vir_ch[:, i, k] = -(Wxx_ch[:, i, k] + Wyy_ch[:, i, k] + Wzz_ch[:, i, k]) / 3.
                    except UnboundLocalError:
                        pass

                    # Simulations with virial off-diagonal components calculation (calculated in the whole fluid)
                    try:
                        Wxy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 3] * mask_fluid, axis=1) / vol_fluid[i,0,k]
                        Wxz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 4] * mask_fluid, axis=1) / vol_fluid[i,0,k]
                        Wyz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 5] * mask_fluid, axis=1) / vol_fluid[i,0,k]

                    except (IndexError, UnboundLocalError) as e:
                        pass

                    # Wall Forces--------------------------------------
                    surfU_fx_ch[:, i] = np.sum(surfU_fx * maskxU, axis=1)
                    surfU_fy_ch[:, i] = np.sum(surfU_fy * maskxU, axis=1)
                    surfU_fz_ch[:, i] = np.sum(surfU_fz * maskxU, axis=1)
                    surfL_fx_ch[:, i] = np.sum(surfL_fx * maskxL, axis=1)
                    surfL_fy_ch[:, i] = np.sum(surfL_fy * maskxL, axis=1)
                    surfL_fz_ch[:, i] = np.sum(surfL_fz * maskxL, axis=1)

                    # Density (bulk) -----------------------------------
                    den_bulk_ch[:, i] = (N_bulk_mask[:, i, k] / self.A_per_molecule) / vol_bulk_cell[i, 0, k]

        else:
            # For the velocity distribution
            fluid_vx_avg_lte = np.mean(fluid_vx, axis=0)
            fluid_vy_avg_lte = np.mean(fluid_vy, axis=0)
            fluid_vz_avg_lte = np.mean(fluid_vz, axis=0)

            for i in range(self.Nx):
                for k in range(self.Nz):
                    maskx_fluid = utils.region(fluid_xcoords, fluid_xcoords,
                                            xx[i, 0, k], xx[i+1, 0, k])['mask']
                    maskz_fluid = utils.region(fluid_zcoords, fluid_zcoords,
                                            zz[i, 0, k], zz[i, 0, k+1])['mask']
                    mask_fluid = np.logical_and(maskx_fluid, maskz_fluid)

                    # Count particles in the fluid cell
                    N_fluid_mask[:, i, k] = np.sum(mask_fluid, axis=1)
                    # Avoid having zero particles in the cell
                    Nzero_fluid = np.less(N_fluid_mask[:, i, k], 1)
                    N_fluid_mask[Nzero_fluid, i, k] = 1

                    # Velocities of the fluid atoms
                    vx_ch[:, i, k] =  np.sum(fluid_vx * mask_fluid, axis=1) / N_fluid_mask[:, i, k]

                    # Density (whole fluid) ----------------------------
                    den_ch[:, i, k] = (N_fluid_mask[:, i, k] / self.A_per_molecule) / vol_cell[i, 0, k]

                    # Mass flux (whole fluid) ----------------------------
                    jx_ch[:, i, k] = vx_ch[:, i, k] * den_ch[:, i, k]

                    # Mass flow rate (whole fluid)
                    mflowrate_ch[:, i, k] = (N_fluid_mask[:, i, k] / self.A_per_molecule) * vx_ch[:, i, k]

                    # Temperature ----------------------------

                    peculiar_v = np.sqrt(fluid_vx_t**2 + fluid_vy_t**2 + fluid_vz_t**2) * mask_fluid / self.A_per_molecule

                    temp_ch[:, i, k] = ((self.mf * sci.gram / sci.N_A) * \
                                        np.sum(peculiar_v**2 , axis=1) * \
                                        A_per_fs_to_m_per_s**2)  / \
                                        (3 * sci.k * N_fluid_mask[:, i, k]/self.A_per_molecule)  # Kelvin

                    # Virial pressure--------------------------------------
                    try:
                        Wxx_ch[:, i, k] = np.sum(virial[:, fluid_idx, 0] * mask_fluid, axis=1) #/ vol_cell[i,0,k]
                        Wyy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 1] * mask_fluid, axis=1) #/ vol_cell[i,0,k]
                        Wzz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 2] * mask_fluid, axis=1) #/ vol_cell[i,0,k]
                        vir_ch[:, i, k] = -(Wxx_ch[:, i, k] + Wyy_ch[:, i, k] + Wzz_ch[:, i, k]) / 3.
                    except UnboundLocalError:
                        pass

                    # Simulations with virial off-diagonal components calculation
                    try:
                        Wxy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 3] * mask_fluid, axis=1) #/ vol_cell[i,0,k]
                        Wxz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 4] * mask_fluid, axis=1) #/ vol_cell[i,0,k]
                        Wyz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 5] * mask_fluid, axis=1) #/ vol_cell[i,0,k]

                    except (IndexError, UnboundLocalError) as e:
                        pass

                    # Density (bulk) -----------------------------------
                    den_bulk_ch[:, i] = (N_fluid_mask[:, i, k] / self.A_per_molecule) / vol_cell[i, 0, k]

        return {'cell_lengths': cell_lengths_updated,
                'kx': kx,
                'ky': ky,
                'kz': kz,
                'gap_heights': gap_heights,
                'bulkStartZ_time': bulkStartZ_time,
                'bulkEndZ_time': bulkEndZ_time,
                'com': comZ,
                'fluxes': fluxes,
                'totVi': totVi,
                'mytotVi': mytotVi,
                'fluid_vx_avg': fluid_vx_avg,
                'fluid_vy_avg': fluid_vy_avg,
                'fluid_vz_avg': fluid_vz_avg,
                'fluid_vx_avg_lte': fluid_vx_avg_lte,
                'fluid_vy_avg_lte': fluid_vy_avg_lte,
                'fluid_vz_avg_lte': fluid_vz_avg_lte,
                'vx_ch': vx_ch,
                'vx_ch_whole':vx_ch_whole,
                'vx_R1': vx_R1,
                'vx_R2': vx_R2,
                'vx_R3': vx_R3,
                'vx_R4': vx_R4,
                'vx_R5': vx_R5,
                'uCOMx': uCOMx,
                'den_ch': den_ch,
                'rho_k': rho_k,
                'sf': sf,
                'sf_solid': sf_solid,
                'sf_x': sf_x,
                'sf_x_solid': sf_x_solid,
                'sf_y': sf_y,
                'sf_y_solid': sf_y_solid,
                'jx_ch' : jx_ch,
                'mflowrate_ch': mflowrate_ch,
                'je_x': je_x,
                'je_y': je_y,
                'je_z': je_z,
                'vir_ch': vir_ch,
                'Wxx_ch': Wxx_ch,
                'Wyy_ch': Wyy_ch,
                'Wzz_ch': Wzz_ch,
                'Wxy_ch': Wxy_ch,
                'Wxz_ch': Wxz_ch,
                'Wyz_ch': Wyz_ch,
                'temp_ch': temp_ch,
                'tempx_ch': tempx_ch,
                'tempy_ch': tempy_ch,
                'tempz_ch': tempz_ch,
                'temp_ch_solid': temp_ch_solid,
                'surfU_fx_ch':surfU_fx_ch,
                'surfU_fy_ch':surfU_fy_ch,
                'surfU_fz_ch':surfU_fz_ch,
                'surfL_fx_ch':surfL_fx_ch,
                'surfL_fy_ch':surfL_fy_ch,
                'surfL_fz_ch':surfL_fz_ch,
                'den_bulk_ch':den_bulk_ch,
                'Nf': Nf,
                'Nm': Nm,
                'fluid_vol':fluid_vol}
