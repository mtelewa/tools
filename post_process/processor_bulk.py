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

    def __init__(self, data, start, end, Nx, Ny, Nz, A_per_molecule, tessellate):
        """
        Parameters:
        -----------
        data: str, NetCDF trajectory file
        start, end: Start and end of the sampled time in each processor
        Nx, Ny, Nz: int, Number of chunks in the x-, y- and z-direcions
        A_per_molecule: int, no. of atoms per molecule
        tessellate: int, boolean to perform Delaunay tessellation
        """

        self.data = data
        self.start, self.end = start, end
        self.chunksize = self.end - self.start
        self.Nx, self.Ny, self.Nz = comm.bcast(Nx, root=0), comm.bcast(Ny, root=0), comm.bcast(Nz, root=0)
        self.A_per_molecule = comm.bcast(A_per_molecule, root=0)
        self.tessellate = comm.bcast(tessellate, root=0)

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
        Define the fluid group in the first timestep from "type", shape (time, Natoms)
        Returns
        ----------
        fluid_idx: (Natoms,) array, indeces of fluid atoms
        Nf: int, no. of fluid atoms
        Nm: int, no. of fluid molecules
        """

        type = self.data.variables["type"]
        types = np.array(type[0, :]).astype(np.float32)

        fluid_idx = []      # Should be of shape: (Nf,)

        # Lennard-Jones
        if np.max(types)==1:
            fluid_idx.append(np.where(types == 1))
            fluid_idx = fluid_idx[0][0]
        # Hydrocarbons
        if np.max(types)==2:
            fluid_idx.append(np.where([types == 1, types == 2]))
            fluid_idx = fluid_idx[0][1]

        Nf, Nm = np.max(fluid_idx)+1, (np.max(fluid_idx)+1)/self.A_per_molecule

        if rank == 0:
            logger.info('A box with {} fluid atoms ({} molecules)'.format(Nf,int(Nm)))

        return fluid_idx, Nf, Nm


    def get_chunks(self):
        """
        Partition the box into spatial bins (chunks)
        Returns
        ----------
        Thermodynamic quantities (arrays) with different shapes. The 3 shapes are:
            (time,): time series of a quantity for the whole domain
                    (6 quantites: heat flux vector and volumes)
            (Nf,): time averaged quantity for each particle
                    (3 quantites: velocity vector)
            (time,Nx,Nz): time series of each chunk along the x and the z-directions
                    (10 quantites: Vx, density, temp, pressure)
        """

        fluid_idx, Nf, Nm = self.get_indices()
        Lx, Ly, Lz = self.get_dimensions()

        cell_lengths_array = [Lx, Ly, Lz]

        # Position, velocity, virial (Wi) array dimensions: (time, Natoms, dimension)
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
        # The averaged quantities are the moving average of a few previous timesteps
        coords_data_unavgd = self.data.variables["coordinates"]   # Used for ACF calculation
        vels_data_unavgd = self.data.variables["velocities"]     # Used for temp. calculation

        # Fluid Virial Pressure ------------------------------------------------
        try:
            virial_data = self.data.variables["f_Wi_avg"]
            virial = np.array(virial_data[self.start:self.end]).astype(np.float32)
        except KeyError:
            virial = np.zeros_like(vels_data[self.start:self.end])

        # Fluid mass -----------------------------------------------------------
        # Assign mass to type (Could be replaced by dumping mass from LAMMPS)
        type = self.data.variables["type"]
        type_array = np.array(type[self.start:self.end]).astype(np.float32)
        mass = np.zeros_like(type_array)
        # TODO: Assign masses for AA
        types = np.array(type_array[0, :]).astype(np.float32)
        if np.max(types)==1: mass_map = {1: 39.948}
        if np.max(types)==2: mass_map = {1: 15.03462, 2:14.02667}   # UA

        for i in range(type_array.shape[0]):
            for j in range(type_array.shape[1]):
                mass[i, j] = mass_map.get(type_array[i, j], 0)

        # Fluid mass
        mass_fluid = mass[:, fluid_idx]     # (time, Nf)

        # ----------------------------------------------------------------------
        # Whole domain: arrays of shape (time,), (Nf,) and (time, Nf)-----------
        # ----------------------------------------------------------------------

        # Voronoi tesellation volume, performed during simulation (Å^3)--------
        try:
            voronoi_vol_data = self.data.variables["f_Vi_avg"]  # (time, Natoms)
            voronoi_vol = np.array(voronoi_vol_data[self.start:self.end]).astype(np.float32)
            Vi = voronoi_vol[:, fluid_idx]  # (time, Nf)
            totVi = np.sum(Vi, axis=1)      # (time, )
        except KeyError:
            totVi = np.zeros([self.chunksize], dtype=np.float32)
            if rank == 0: logger.info('Voronoi volume was not computed during LAMMPS Run!')
            pass

        # Box volume (Å^3)-----------------------------------------------------------
        cell_lengths = self.data.variables["cell_lengths"]
        # Shape: (time,)
        fluid_vol = np.array(cell_lengths[self.start:self.end, 0]).astype(np.float32) * \
                    np.array(cell_lengths[self.start:self.end, 1]).astype(np.float32) * \
                    np.array(cell_lengths[self.start:self.end, 2]).astype(np.float32)

        # Heat flux from ev: total energy * velocity and sv: energy from virial * velocity (Kcal/mol . Å/fs) -----
        try:
            ev_data = self.data.variables["f_ev"]       # (time, Natoms, 3)
            centroid_vir_data = self.data.variables["f_sv"]   # (time, Natoms, 3)
            ev = np.array(ev_data[self.start:self.end]).astype(np.float32)
            centroid_vir = np.array(centroid_vir_data[self.start:self.end]).astype(np.float32)

            fluid_evx, fluid_evy, fluid_evz = ev[:, fluid_idx, 0], \
                                              ev[:, fluid_idx, 1], \
                                              ev[:, fluid_idx, 2]

            fluid_svx, fluid_svy, fluid_svz = centroid_vir[:, fluid_idx, 0] ,\
                                              centroid_vir[:, fluid_idx, 1] ,\
                                              centroid_vir[:, fluid_idx, 2]

            je_x = np.sum((fluid_evx + fluid_svx), axis=1) # (time,)
            je_y = np.sum((fluid_evy + fluid_svy), axis=1)
            je_z = np.sum((fluid_evz + fluid_svz), axis=1)
        except KeyError:
            je_x, je_y, je_z = np.zeros([self.chunksize], dtype=np.float32),\
                               np.zeros([self.chunksize], dtype=np.float32),\
                               np.zeros([self.chunksize], dtype=np.float32)

        # Cartesian Coordinates ------------------------------------------------
        coords = np.array(coords_data[self.start:self.end]).astype(np.float32)
        # Shape: (time, Nf)
        fluid_xcoords,fluid_ycoords,fluid_zcoords = coords[:, fluid_idx, 0], \
                                                    coords[:, fluid_idx, 1], \
                                                    coords[:, fluid_idx, 2]

        # Delaunay triangulation (performed in post-processing)-----------------
        if self.tessellate==1:
            fluid_coords = np.zeros((self.chunksize, Nf, 3))
            fluid_coords[:,:,0], fluid_coords[:,:,1], fluid_coords[:,:,2] = \
            fluid_xcoords, fluid_ycoords, fluid_zcoords

            del_totVi = np.zeros([self.chunksize], dtype=np.float32)
            for i in range(self.chunksize):
                del_totVi[i] = tessellation.delaunay_volumes(fluid_coords[i])
        else:
            del_totVi = np.zeros([self.chunksize], dtype=np.float32)

        # Unaveraged positions
        coords_unavgd = np.array(coords_data_unavgd[self.start:self.end]).astype(np.float32)

        fluid_xcoords_unavgd, fluid_ycoords_unavgd, fluid_zcoords_unavgd = coords_unavgd[:, fluid_idx, 0], \
                                                                           coords_unavgd[:, fluid_idx, 1], \
                                                                           coords_unavgd[:, fluid_idx, 2]

        # For box volume that changes with time (i.e. NPT ensemble)
        # Shape: (time,)
        fluid_min = [utils.extrema(fluid_xcoords)['global_min'],
                     utils.extrema(fluid_ycoords)['global_min'],
                     utils.extrema(fluid_zcoords)['global_min']]

        fluid_max = [utils.extrema(fluid_xcoords)['global_max'],
                     utils.extrema(fluid_ycoords)['global_max'],
                     utils.extrema(fluid_zcoords)['global_max']]

        # Fluid domain dimensions
        fluid_lengths = [fluid_max[0]- fluid_min[0], fluid_max[1]- fluid_min[1], fluid_max[2]- fluid_min[2]]

        # Velocities -----------------------------------------------------------
        vels = np.array(vels_data[self.start:self.end]).astype(np.float32)

        # Velocity of each atom
        # Shape: (time, Nf)
        fluid_vx, fluid_vy, fluid_vz = vels[:, fluid_idx, 0], \
                                       vels[:, fluid_idx, 1], \
                                       vels[:, fluid_idx, 2]

        # Unaveraged velocities
        vels_t = np.array(vels_data_unavgd[self.start:self.end]).astype(np.float32)

        fluid_vx_t, fluid_vy_t, fluid_vz_t = vels_t[:, fluid_idx, 0], \
                                             vels_t[:, fluid_idx, 1], \
                                             vels_t[:, fluid_idx, 2]

        # For the velocity distribution (Nf,)
        fluid_vx_avg = np.mean(fluid_vx, axis=0)
        fluid_vy_avg = np.mean(fluid_vy, axis=0)
        fluid_vz_avg = np.mean(fluid_vz, axis=0)

        # ----------------------------------------------------------------------
        # The Grid -------------------------------------------------------------
        # ----------------------------------------------------------------------
        dim = np.array([self.Nx, self.Ny, self.Nz])
        # Bounds should change for example in NPT simulations
        # (NEEDED especially when the Box volume changes e.g. NPT)
        bounds = [np.arange(dim[i] + 1) / dim[i] * fluid_lengths[i] + fluid_min[i] for i in range(3)]

        xx, yy, zz, vol_cell = utils.bounds(bounds[0], bounds[1], bounds[2])

        # ----------------------------------------------------------------------
        # Cell Partition: arrays of shape (time, Nx, Nz)------------------------
        # ----------------------------------------------------------------------
        # initialize buffers to store the count 'N' and 'data_ch' of each chunk
        N_fluid_mask = np.zeros([self.chunksize, self.Nx, self.Nz], dtype=np.float32)
        vx_ch = np.zeros_like(N_fluid_mask)
        den_ch = np.zeros_like(vx_ch)
        temp_ch = np.zeros_like(vx_ch)
        Wxx_ch = np.zeros_like(vx_ch)
        Wyy_ch = np.zeros_like(vx_ch)
        Wzz_ch = np.zeros_like(vx_ch)
        Wxy_ch = np.zeros_like(vx_ch)
        Wxz_ch = np.zeros_like(vx_ch)
        Wyz_ch = np.zeros_like(vx_ch)
        vir_ch = np.zeros_like(vx_ch)

        for i in range(self.Nx):
            for k in range(self.Nz):
        # Fluid partition ----------------------------------------
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

        # Thermodynamic properties ----------------------------------
                # Velocities (Å/fs) --------------------------------------------
                vx_ch[:, i, k] =  np.sum(fluid_vx * mask_fluid, axis=1) / N_fluid_mask[:, i, k]

                # Density (g/(mol.Å^3)) ----------------------------------------
                den_ch[:, i, k] = np.sum(mass_fluid * mask_fluid, axis=1) / vol_cell[i, 0, k]

                # Temperature (K) ----------------------------------------------
                peculiar_v = np.sqrt(fluid_vx_t**2 + fluid_vy_t**2 + fluid_vz_t**2) * mask_fluid
                temp_ch[:, i, k] = np.sum(mass_fluid * sci.gram / sci.N_A * peculiar_v**2 * A_per_fs_to_m_per_s**2 , axis=1) / \
                                    (3 * N_fluid_mask[:, i, k] * sci.k)

                # Virial pressure (atm.Å^3)-------------------------------------
                Wxx_ch[:, i, k] = np.sum(virial[:, fluid_idx, 0] * mask_fluid, axis=1)
                Wyy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 1] * mask_fluid, axis=1)
                Wzz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 2] * mask_fluid, axis=1)
                vir_ch[:, i, k] = -(Wxx_ch[:, i, k] + Wyy_ch[:, i, k] + Wzz_ch[:, i, k]) / 3.

                # Virial off-diagonal components (atm.Å^3)----------------------
                try:
                    Wxy_ch[:, i, k] = np.sum(virial[:, fluid_idx, 3] * mask_fluid, axis=1)
                    Wxz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 4] * mask_fluid, axis=1)
                    Wyz_ch[:, i, k] = np.sum(virial[:, fluid_idx, 5] * mask_fluid, axis=1)
                except IndexError:
                    pass

        return {'cell_lengths': cell_lengths_array,
                'Nf': Nf,
                'Nm': Nm,
                'totVi': totVi,
                'del_totVi': del_totVi,
                'fluid_vol':fluid_vol,
                'je_x': je_x,
                'je_y': je_y,
                'je_z': je_z,
                'fluid_vx_avg': fluid_vx_avg,
                'fluid_vy_avg': fluid_vy_avg,
                'fluid_vz_avg': fluid_vz_avg,
                'vx_ch': vx_ch,
                'den_ch': den_ch,
                'temp_ch': temp_ch,
                'Wxx_ch': Wxx_ch,
                'Wyy_ch': Wyy_ch,
                'Wzz_ch': Wzz_ch,
                'Wxy_ch': Wxy_ch,
                'Wxz_ch': Wxz_ch,
                'Wyz_ch': Wyz_ch,
                'vir_ch': vir_ch}
