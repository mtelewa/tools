#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier

def modify_geometry(datafile, slope, intercept, cut_begin):
    """
    Import the NetCDF trajectory and add the construct surfacr mesh modifier

    Parameters:
    datafile: str, LAMMPS data file to modify
    slope: float, slope of the cut plane
    intercept: float, shift the cut plane along the z-direction
    cut_begin: float, x-coordinate to use in the boolean expression for removing atoms
    """

    pipeline = import_file(datafile, multiple_frames = False)

    # Run the pipleine
    data = pipeline.compute()

    # Number of atoms in the trajectory
    # Select type(s)
    # pipeline.modifiers.append(SelectTypeModifier(types = {1,2}))
    natoms = data.particles.count
    print(f'Number of atoms: {natoms}')

    # Select atoms from the lower wall above the slope
    pipeline.modifiers.append(ExpressionSelectionModifier(
        expression = f'Position.X>{cut_begin}*CellSize.X && Position.Z<0.5*CellSize.Z \
                      && Position.Z>({slope}*Position.X)+{intercept}'))

    # Delete the selected atoms
    pipeline.modifiers.append(DeleteSelectedModifier())

    # Select atoms from the upper wall below the slope # the 3 used here is one unit length in the z-direction for gold (sqrt(3)*4.08)
    pipeline.modifiers.append(ExpressionSelectionModifier(
        expression = f'Position.X>{cut_begin}*CellSize.X && Position.Z>0.5*CellSize.Z \
                      && Position.Z<({-slope}*Position.X)+{-intercept-3}+CellSize.Z'))

    pipeline.modifiers.append(DeleteSelectedModifier())

    return pipeline



if __name__ == "__main__":
    # For iter-14
    # pipeline = modify_geometry(sys.argv[1], 0.186, -64, 0.6)
    # For iter-16
    pipeline = modify_geometry(sys.argv[1], 0.884, -475, 0.9)

    export_file(pipeline, "output.data", "lammps/data", atom_style="molecular")
