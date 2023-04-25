# Tools for MD analysis

This repository contains a collection of tools for molecular dynamics (MD) simulations. The tools are designed to be used with [LAMMPS MD software package](https://www.lammps.org/) and [Moltemplate](https://moltemplate.org/).

## Installation

To use the tools in this repository, you should clone the repository to your local machine:
``git clone git@github.com:mtelewa/tools.git``

Continue installation instructions ..

## Usage

The tools in this repository are organized into several subdirectories, each containing a set of related scripts. To use the tools, navigate to the appropriate subdirectory and run the desired script. The subdirectories are:

* cluster: Simulation and post-processing submission sctipts for bwUniCluster (UC2) and bwForCluster (NEMO) including Singularity scripts
* compute\_thermo: Compute thermodynamic variables from the grids created from post-processing
* init\_molecules: Initialize molecular domains with LAMMPS and Moltemplate
* plot: Plotting scripts
* post\_process: Post-processing scripts that convert LAMMPS NetCDF trajectory to a grid and store it in NetCDF files
* snippets: Miscallenous scripts to run on local and remote machines

## Contributing

Contributions to this repository are welcome. If you find a bug or have a suggestion for a new tool, please open an issue on the repository's issue tracker. If you would like to contribute code, please fork the repository and submit a pull request with your changes.
