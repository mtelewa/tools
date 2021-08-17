#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, logging, warnings
import numpy as np
import scipy.constants as sci
import time as timer
from mpi4py import MPI
import netCDF4
import utils
from operator import itemgetter

# Warnings Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# np.set_printoptions(threshold=sys.maxsize)

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
fs_to_ns = 1e-6
A_per_fs_to_m_per_s = 1e5

# Set the Mass and no. of atoms per molecule for each fluid
mCH2, mCH3, mCH_avg = 12, 13, 12.5


t0 = timer.time()

class traj_to_grid:

    def __init__(self, data, start, end, Nx, Nz, mf, A_per_molecule, Ny=1):

        self.data = data
        self.start, self.end = start, end
        self.chunksize = self.end - self.start
        self.Nx, self.Ny, self.Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)
        self.mf, self.A_per_molecule = mf, A_per_molecule

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
        # Lennard-Jones
        if np.max(type)==2:
            fluid_idx.append(np.where(type == 1))
            fluid_idx = fluid_idx[0][0]
            solid_idx.append(np.where(type == 2))
        # Hydrocarbons
        if np.max(type)==3:
            fluid_idx.append(np.where([type == 1, type == 2]))
            fluid_idx = fluid_idx[0][1]
            solid_idx.append(np.where(type == 3))

        solid_idx = solid_idx[0][0]

        Nf, Nm, Ns = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/self.A_per_molecule, \
                                np.max(solid_idx)-np.max(fluid_idx)

        solid_start = np.min(solid_idx)

        if rank == 0:
            print('A box with {} fluid atoms ({} molecules) and {} solid atoms'.format(Nf,int(Nm),Ns))

        return fluid_idx, solid_start, Nf


    def get_chunks(self):
        """
        Partitions the box into regions (solid and fluid) as well as chunks in
        those regions
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

        d = traj_to_grid(self.data, self.start, self.end, self.Nx, self.Nz, self.mf, self.A_per_molecule)

        fluid_idx, solid_start, Nf = d.get_indices()
        Lx, Ly, Lz = d.get_dimensions()

        # Array dimensions: (time, idx, dimesnion)
        coords_data = self.data.variables["f_position"]
        vels_data = self.data.variables["f_velocity"]
        # The unaveraged quantity is that which is dumped by LAMMPS every N timesteps
        # i.e. it is a snapshot of the system at this timestep.
        # As opposed to averaged quantities which are the average of a few previous timesteps
        vels_data_unavgd = self.data.variables["velocities"]     # Used for temp. calculation

        # In some trajectories, virial is not computed (faster simulation)
        try:
            voronoi_vol_data = self.data.variables["f_Vi_avg"]
            virial_data = self.data.variables["f_Wi_avg"]
        except KeyError:
            if rank == 0:
                print('Virial was not computed during LAMMPS Run!')
            pass

        forcesU = self.data.variables["f_fAtomU_avg"]
        forcesL = self.data.variables["f_fAtomL_avg"]

        # for i in range(self.Nslices):
        # Coordinates ---------------------------------------------
        coords = np.array(coords_data[self.start:self.end]).astype(np.float32)

        fluid_xcoords,fluid_ycoords,fluid_zcoords = coords[:, fluid_idx, 0], \
                                                    coords[:, fluid_idx, 1], \
                                                    coords[:, fluid_idx, 2]

        # Velocities ------------------------------------------------
        vels = np.array(vels_data[self.start:self.end]).astype(np.float32)

        fluid_vx, fluid_vy, fluid_vz = vels[:, fluid_idx, 0], \
                                       vels[:, fluid_idx, 1], \
                                       vels[:, fluid_idx, 2]

        # For the velocity distribution
        fluid_vx_avg = np.mean(fluid_vx, axis=0)
        fluid_vy_avg = np.mean(fluid_vy, axis=0)

        fluid_v = np.sqrt(fluid_vx**2 + fluid_vy**2 + fluid_vz**2)

        # For the Temperature
        vels_t = np.array(vels_data_unavgd[self.start:self.end]).astype(np.float32)

        fluid_vx_t,fluid_vy_t,fluid_vz_t = vels_t[:, fluid_idx, 0], \
                                           vels_t[:, fluid_idx, 1], \
                                           vels_t[:, fluid_idx, 2]

        fluid_v_t = np.sqrt(fluid_vx_t**2 + fluid_vy_t**2 + fluid_vz_t**2) / self.A_per_molecule


        # Fluid Virial Pressure ----------------------------------
        try:
            voronoi_vol = np.array(voronoi_vol_data[self.start:self.end]).astype(np.float32)
            virial = np.array(virial_data[self.start:self.end]).astype(np.float32)
        except UnboundLocalError:
            pass

        # Wall Stresses -------------------------------------------------
        forcesU_data = np.array(forcesU[self.start:self.end]).astype(np.float32)
        forcesL_data = np.array(forcesL[self.start:self.end]).astype(np.float32)

        # Instanteneous extrema (array Needed for the gap height at each timestep)
        fluid_begin, fluid_end = utils.extrema(fluid_zcoords)['local_min'], \
                                 utils.extrema(fluid_zcoords)['local_max']

        fluid_zcoords_conv = utils.region(fluid_zcoords, fluid_xcoords, 0.49*Lx, 0.5*Lx)['data']
        fluid_begin_conv = utils.cnonzero_min(fluid_zcoords_conv)['local_min']
        fluid_end_conv = utils.extrema(fluid_zcoords_conv)['local_max']

        fluid_zcoords_div = utils.region(fluid_zcoords, fluid_xcoords, 0, 0.2*Lx)['data']
        fluid_begin_div = utils.cnonzero_min(fluid_zcoords_div)['local_min']
        fluid_end_div = utils.extrema(fluid_zcoords_div)['local_max']

        # Define the solid region ---------------------------------------------
        solid_xcoords, solid_zcoords = coords[:, solid_start:, 0], \
                                       coords[:, solid_start:, 2]

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


        surfU_fx,surfU_fy,surfU_fz = forcesU_data[:, solid_start:, 0][:,surfU_indices], \
                                     forcesU_data[:, solid_start:, 1][:,surfU_indices], \
                                     forcesU_data[:, solid_start:, 2][:,surfU_indices]

        surfL_fx,surfL_fy,surfL_fz = forcesL_data[:, solid_start:, 0][:,surfL_indices], \
                                     forcesL_data[:, solid_start:, 1][:,surfL_indices], \
                                     forcesL_data[:, solid_start:, 2][:,surfL_indices]


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


        # The converged and diverged regions fluid, surfU and surfL coordinates
        surfU_zcoords_div = utils.region(surfU_zcoords, surfU_xcoords, 0, 0.2*Lx)['data']
        surfL_zcoords_div = utils.region(surfL_zcoords, surfL_xcoords, 0, 0.2*Lx)['data']
        surfU_begin_div = utils.cnonzero_min(surfU_zcoords_div)['local_min']
        surfL_end_div = utils.extrema(surfL_zcoords_div)['local_max']

        # Narrow range because of outliers in the fluid coordinates
        surfU_zcoords_conv = utils.region(surfU_zcoords, surfU_xcoords, 0.49*Lx, 0.5*Lx)['data']
        surfL_zcoords_conv = utils.region(surfL_zcoords, surfL_xcoords, 0.49*Lx, 0.5*Lx)['data']
        surfU_begin_conv = utils.cnonzero_min(surfU_zcoords_conv)['local_min']
        surfL_end_conv = utils.extrema(surfL_zcoords_conv)['local_max']

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


        # REGIONS within the fluid------------------------------------------------
        #-------------------------------------------------------------------------
        # Pump --------------------------------------------
        pumpStartX, pumpEndX = 0.4 * Lx, 0.6 * Lx
        pump_length, pump_region, pump_xcoords, pump_vol, pump_N = \
            itemgetter('interval', 'mask', 'data', 'vol', 'count')\
            (utils.region(fluid_xcoords, fluid_xcoords, pumpStartX, pumpEndX,
                                    ylength=Ly, length=avg_gap_height_conv))

        # Mass flux in the pump region
        vels_pump = fluid_vx * pump_region / self.A_per_molecule
        mflux_pump = (np.sum(vels_pump, axis=1) * (sci.angstrom/fs_to_ns) * \
                     (self.mf/sci.N_A)) / (pump_vol * (sci.angstrom)**3)      # g/m2.ns
        mflowrate_pump = (np.sum(vels_pump, axis=1) * (sci.angstrom/fs_to_ns) * \
                        (self.mf/sci.N_A)) / (pump_length * sci.angstrom)       # g/ns

        # Bulk --------------------------------------------
        avg_surfL_end_conv = np.mean(comm.allgather(np.mean(surfL_end_conv)))
        bulkStartZ = (0.4 * avg_gap_height_conv) + avg_surfL_end_conv
        bulkEndZ = (0.6 * avg_gap_height_conv) + avg_surfL_end_conv
        bulk_height, bulk_region, bulk_zcoords, bulk_vol, bulk_N = \
            itemgetter('interval', 'mask', 'data', 'vol', 'count')\
            (utils.region(fluid_zcoords, fluid_zcoords, bulkStartZ, bulkEndZ,
                                            ylength=Ly, length=Lx))
        bulk_height_time = 0.2 * gap_height_conv

        try:
            # Voronoi volumes of atoms in the bulk region in each timestep
            Vi = voronoi_vol[:, fluid_idx] * bulk_region
            # Voronoi volume of the whole bulk region in each timestep
            totVi = np.sum(Vi, axis=1)
        except UnboundLocalError:
            totVi = 0
            pass

        # Stable ---------------------------------
        stableStartX, stableEndX = 0.4 * Lx, 0.8 * Lx
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


        # The Grid -------------------------------------------------------------
        # ----------------------------------------------------------------------
        dim = np.array([self.Nx, self.Ny, self.Nz])

        # Whole cell --------------------------------------------
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
        bounds_bulk = [bounds[0], bounds[1], bulkRangeZ]
        xx_bulk, yy_bulk, zz_bulk, vol_bulk_cell = utils.bounds(bounds_bulk[0],
                                                   bounds_bulk[1], bounds_bulk[2])

        # Stable --------------------------------------------
        stableRangeX = np.arange(dim[0] + 1) / dim[0] * stable_length + stableStartX
        bounds_stable = [stableRangeX, bounds[1], bounds[2]]
        xx_stable, yy_sytable, zz_stable, vol_stable_cell = utils.bounds(bounds_stable[0],
                                                            bounds_stable[1], bounds_stable[2])


        # -------------------------------------------------------
        # Cell Partition ---------------------------------------
        # -------------------------------------------------------
        # initialize buffers to store the count 'N' and 'data_ch' of each chunk
        N_fluid_mask = np.zeros([self.chunksize, self.Nx, self.Nz], dtype=np.float32)
        N_stable_mask = np.zeros_like(N_fluid_mask)
        N_bulk_mask = np.zeros_like(N_fluid_mask)

        vx_ch = np.zeros([self.chunksize, self.Nx, self.Nz], dtype=np.float32)
        vir_ch = np.zeros_like(vx_ch)
        den_ch = np.zeros_like(vx_ch)
        jx_ch = np.zeros_like(vx_ch)
        mflowrate_ch = np.zeros_like(vx_ch)
        temp_ch = np.zeros_like(vx_ch)

        N_Upper_mask = np.zeros([self.chunksize, self.Nx], dtype=np.float32)
        N_Lower_mask = np.zeros_like(N_Upper_mask)

        surfU_fx_ch = np.zeros([self.chunksize, self.Nx], dtype=np.float32)
        surfU_fy_ch = np.zeros_like(surfU_fx_ch)
        surfU_fz_ch = np.zeros_like(surfU_fx_ch)
        surfL_fx_ch = np.zeros_like(surfU_fx_ch)
        surfL_fy_ch = np.zeros_like(surfU_fx_ch)
        surfL_fz_ch = np.zeros_like(surfU_fx_ch)
        den_bulk_ch = np.zeros_like(surfU_fx_ch)

        for i in range(self.Nx):
            for k in range(self.Nz):
        # Fluid -----------------------------------------
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

        # SurfL -----------------------------------
                maskxL = utils.region(surfL_xcoords, surfL_xcoords,
                                        xx[i, 0, 0], xx[i+1, 0, 0])['mask']
                N_Lower_mask[:, i] = np.sum(maskxL, axis=1)

        # -----------------------------------------------------
        # Cell Averages ---------------------------------------
        # -----------------------------------------------------
                # Velocities in the stable region
                vx_ch[:, i, k] =  np.sum(fluid_vx * mask_stable, axis=1) / N_stable_mask[:, i, k]

                # Density (whole fluid) ----------------------------
                den_ch[:, i, k] = (N_fluid_mask[:, i, k] / self.A_per_molecule) / vol_fluid[i, 0, k]

                # Mass flux (whole fluid) ----------------------------
                jx_ch[:, i, k] = vx_ch[:, i, k] * den_ch[:, i, k]

                # # TODO: compute mflowrate in the chunks
                # mflowrate_ch[:, i, k] = vx_ch[:, i, k] *

                # Temperature ----------------------------
                temp_ch[:, i, k] = ((self.mf * sci.gram / sci.N_A) * np.sum((fluid_v_t**2) * mask_fluid, axis=1) * \
                                    A_per_fs_to_m_per_s**2)  / \
                                    (3 * N_fluid_mask[:, i, k] * sci.k / self.A_per_molecule)  # Kelvin

                # Virial pressure--------------------------------------
                try:
                    W1_ch = np.sum(virial[:, fluid_idx, 0] * mask_bulk, axis=1)
                    W2_ch = np.sum(virial[:, fluid_idx, 1] * mask_bulk, axis=1)
                    W3_ch = np.sum(virial[:, fluid_idx, 2] * mask_bulk, axis=1)
                    vir_ch[:, i, k] = -(W1_ch + W2_ch + W3_ch) #/ vol[i,0,k]
                except UnboundLocalError:
                    pass

                # Wall Forces--------------------------------------
                surfU_fx_ch[:, i] = np.sum(surfU_fx * maskxU, axis=1)
                surfU_fy_ch[:, i] = np.sum(surfU_fy * maskxU, axis=1)
                surfU_fz_ch[:, i] = np.sum(surfU_fz * maskxU, axis=1)
                surfL_fx_ch[:, i] = np.sum(surfL_fx * maskxL, axis=1)
                surfL_fy_ch[:, i] = np.sum(surfL_fy * maskxL, axis=1)
                surfL_fz_ch[:, i] = np.sum(surfL_fz * maskxL, axis=1)

                # Density (bulk) -----------------------------------
                den_bulk_ch[:, i] = (N_bulk_mask[:, i, k] / self.A_per_molecule) / vol_bulk_cell[i, 0, 0]


        return {'cell_lengths': cell_lengths_updated,
                'gap_heights': gap_heights,
                'bulk_height_time': bulk_height_time,
                'com': comZ,
                'fluxes': fluxes,
                'totVi': totVi,
                'fluid_vx_avg': fluid_vx_avg,
                'fluid_vy_avg': fluid_vy_avg,
                'vx_ch': vx_ch,
                'den_ch': den_ch,
                'jx_ch' : jx_ch,
                'vir_ch': vir_ch,
                'temp_ch': temp_ch,
                'surfU_fx_ch':surfU_fx_ch,
                'surfU_fy_ch':surfU_fy_ch,
                'surfU_fz_ch':surfU_fz_ch,
                'surfL_fx_ch':surfL_fx_ch,
                'surfL_fy_ch':surfL_fy_ch,
                'surfL_fz_ch':surfL_fz_ch,
                'den_bulk_ch':den_bulk_ch,
                'Nf': Nf}



# # The unaveraged velocities
# vels_t = np.array(vels_data_unavgd[start:end]).astype(np.float32)
#
# fluid_vx_t,fluid_vy_t,fluid_vz_t = vels_t[:, fluid_idx, 0], \
#                                    vels_t[:, fluid_idx, 1], \
#                                    vels_t[:, fluid_idx, 2]
#
# fluid_v_t = np.sqrt(fluid_vx_t**2 + fluid_vy_t**2 + fluid_vz_t**2) / self.A_per_molecule
#
# # Velocity in the fluid
# v_t_ch = np.zeros([chunksize, Nx, Nz], dtype=np.float32)
#
# # Unavgd velocities--------------------------------------
# if 'no-temp' not in sys.argv:
#     v_t_ch[:, i, k] = np.sum((fluid_v_t**2) * mask_fluid, axis=1) / \
#                              (3 * N_fluid_mask[:, i, k] / self.A_per_molecule)


# # Correction for the converging-diverging channel ---------------------------------------
#
#
# surfU_begin_cell = np.zeros([self.Nx], dtype=np.float32)
# surfL_end_cell = np.zeros_like(surfU_begin_cell)
# fluid_begin_cell = np.zeros_like(surfU_begin_cell)
# fluid_end_cell = np.zeros_like(surfU_begin_cell)
#
# for i in range(self.Nx):
#     cellzU = utils.region(surfU_zcoords, surfU_xcoords, xx[i, 0, 0], xx[i+1, 0, 0])['data']
#     cellzL = utils.region(surfL_zcoords, surfL_xcoords, xx[i, 0, 0], xx[i+1, 0, 0])['data']
#     cell_fluid = utils.region(fluid_zcoords, fluid_xcoords, xx[i, 0, 0], xx[i+1, 0, 0])['data']
#
#     surfU_begin_cell[i] = np.mean(comm.allgather(utils.cnonzero_min(cellzU)['local_min']))
#     surfL_end_cell[i] = np.mean(comm.allgather(utils.extrema(cellzL)['local_max']))
#     fluid_begin_cell[i] = np.mean(comm.allgather(utils.cnonzero_min(cell_fluid)['local_min']))
#     fluid_end_cell[i] = np.mean(comm.allgather(utils.extrema(cell_fluid)['local_max']))
#
#
# zlo = (fluid_begin_cell + surfL_end_cell) / 2.
# zhi = (fluid_end_cell + surfU_begin_cell) / 2.
# gap_height_cell = zhi - zlo
#
# # Update the z-bounds to get the right chunk volume
# if self.Nx > 1:
#     # # TODO: For now we take the last chunk and repeat it twice to
#     # get the dimensionality right but shouldn't use this later
#     zlo = np.append(zlo, zlo[-1])
#     zhi = np.append(zhi, zhi[-1])
#
#     zz_fluid = np.array([zlo, zhi])               # Array of min and max z in the region
#     zz_fluid = np.expand_dims(zz_fluid, axis=2)
#     zz_fluid = np.transpose(zz_fluid)                   # Region min and max are in one line
#     zz_fluid = np.concatenate((zz_fluid, zz_fluid), axis=0)    # Shape it like in the mesh
#     zz_fluid = np.transpose(zz_fluid, (1, 0, 2))
#
#     # Update the z-increment and the volume
#     dx = xx_fluid[1:, 1:, 1:] - xx_fluid[:-1, :-1, :-1]
#     dy = yy_fluid[1:, 1:, 1:] - yy_fluid[:-1, :-1, :-1]
#     dz = zz_fluid[1:, 1:, 1:] - zz_fluid[:-1, :-1, :-1]
#     vol_fluid = dx * dy * dz
#
#
