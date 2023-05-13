#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, subprocess
from ruamel.yaml import YAML
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import dtoolcore
from dtool_create.dataset import _get_readme_template
from pathlib import Path

class manipulate_ds:

    def __init__(self, ds):
        """
        Parameters:
        -----------
        ds: str
            Source dataset name
        """
        self.dataset_name = ds
        # global yaml setting
        self.yaml = YAML()
        self.yaml.explicit_start = True
        self.yaml.indent(mapping=4, sequence=4, offset=2)

        # Base path for the simulation datset
        self.base_uri = Path.cwd()
        # Simulation dataset path
        self.sim_uri = Path.cwd().joinpath(self.dataset_name)
        # Sim dataset out path
        self.sim_out_uri = os.path.join(self.sim_uri,'data','out')
        # Get the simulation dataset uuid
        try:    # For ProtoDataSets
            self.sim_dataset = dtoolcore.ProtoDataSet.from_uri(self.dataset_name)
        # TODO: Handle the exceptions
        except dtoolcore.DtoolCoreTypeError: # If it is not a dataset
            pass
        except TypeError:  # For frozen datasets
            self.sim_dataset = dtoolcore.DataSet.from_uri(self.dataset_name)

    def create_dataset(self, **kwargs):
        """
        Creates a parent dataset
        """

        ds = dtoolcore.DataSetCreator(self.dataset_name, str(self.base_uri), creator_username='mtelewa')
        metadata = self.update_readme(**kwargs)
        template = os.path.join(self.sim_uri, 'README.yml')

        # write the readme of the dataset
        with open(template, "w") as f:
            self.yaml.dump(metadata, f)


    def create_post(self, freeze=None, copy=None, **kwargs):
        """
        Creates a post-processing dataset derived from a simulation step
        """

        # Post-proc dataset path
        post_uri = str(self.sim_uri)+'-post'
        # Post-proc dataset name
        post_name = self.dataset_name +'-post'
        # Post-proc dataset readme template
        post_template = os.path.join(post_uri, 'README.yml')

        metadata = self.update_readme(**kwargs)
        metadata['derived_from'][0]['uuid'] = self.sim_dataset.uuid

        # Create the derived dataset if not already existing
        if not os.path.isdir(post_name):
            dds = dtoolcore.create_derived_proto_dataset(post_name, str(self.base_uri),
                        self.sim_dataset, creator_username='mtelewa')
        else:
            dds = dtoolcore.ProtoDataSet.from_uri(post_uri)

        # Copy the files to the post-proc dataset and remove them from the simulation dataset
        for root, dirs, files in os.walk(self.sim_out_uri):
            for i in files:
                if 'x' in i or 'vs' in i:
                    dds.put_item(os.path.join(self.sim_out_uri,i), i)
                    os.remove(os.path.join(self.sim_out_uri,i))

        for root, dirs, files in os.walk(self.sim_uri):
            for i in files:
                if 'log.lammps' in i:
                    dds.put_item(os.path.join(self.sim_uri,'data',i), i)

        # write the readme of the post-proc dataset
        with open(post_template, "w") as f:
            self.yaml.dump(metadata, f)

        if freeze:
            print(f'Freezing Simulation dataset: {self.sim_uri} ----')
            self.sim_dataset.freeze()
        if copy:
            print(f'Copying dataset: {self.sim_uri} to S3 ----')
            dtoolcore.copy(self.sim_dataset.uri, 's3://frct-simdata',
                    config_path=os.path.join(os.path.expanduser('~'),'.config/dtool/dtool.json'))
            # subprocess.call([f"rsync -av {self.sim_dataset.uri} lsdf:{input('lsdf_dir:')}"])

    def create_derived(self, derived_name, **kwargs):
        """
        Creates a dataset derived from a previous simulation step
        """

        derived_uri = os.path.join(self.base_uri, derived_name)

        # Create the derived dataset if not already existing
        dds = dtoolcore.create_derived_proto_dataset(derived_name, str(self.base_uri),
                    self.sim_dataset, creator_username='mtelewa')

        # # derived dataset readme template
        derived_template = os.path.join(derived_uri, 'README.yml')
        # template = os.path.join(self.sim_uri, 'README.yml')

        metadata = self.update_readme(**kwargs)
        if '-post' not in self.dataset_name:    # Take the UUID of the dataset itself not the post-processed one
            metadata['derived_from'][0]['uuid'] = self.sim_dataset.uuid

        # write the readme of the post-proc dataset
        with open(derived_template, "w") as f:
            self.yaml.dump(metadata, f)

    def put_tag(self, tag, **kwargs):
        """
        Adds a tag to the dataset
        """
        dtoolcore.DataSet.from_uri(self.dataset_name).put_tag(tag)

    def update_readme(self, **kwargs):

        # Check if readme of the dataset is empty
        ds_readme = _get_readme_template(os.path.join(self.sim_uri, 'README.yml'))

        if self.yaml.load(ds_readme) == None:
            # Load Readme template
            read_from = os.path.join(os.path.expanduser('~'), '.dtool', 'custom_dtool_readme.yml')
        else:
            # Load the simulation dataset readme
            read_from = os.path.join(self.sim_uri, 'README.yml')

        readme_template = _get_readme_template(read_from)
        # load readme template and update
        metadata = self.yaml.load(readme_template)

        metadata['creation_date'] = datetime.today()
        metadata['expiration_date'] = metadata['creation_date'] + relativedelta(years=10)

        if 'project' in kwargs:
            metadata['project'] = kwargs['project']
        if 'description' in kwargs:
            metadata['description'] = kwargs['description']
        if 'uuid' in kwargs:
            metadata['derived_from'][0]['uuid'] = kwargs['uuid']
        if 'bc' in kwargs:
            metadata['boundary_conditions'] = kwargs['bc']
        if 'fluid' in kwargs:
            metadata['Materials'][0]['fluid'] = kwargs['fluid']
        if 'walls' in kwargs:
            metadata['Materials'][1]['walls'] = kwargs['walls']
        if 'FF' in kwargs:
            metadata['Force_fields/Potentials'][0]['force field fluid'] = kwargs['FF']
        if 'pot_wall' in kwargs:
            metadata['Force_fields/Potentials'][1]['potential wall'] = kwargs['pot_wall']
        if 'eps_s' in kwargs:
            metadata['Force_fields/Potentials'][2]['eps_s'] = kwargs['eps_s']
        if 'dt' in kwargs:
            metadata['Force_fields/Potentials'][3]['dt'] = kwargs['dt']
        if 'method' in kwargs:
            metadata['Method'] = kwargs['method']
        if 'thermostat' in kwargs:
            metadata['Thermostat'] = kwargs['thermostat']
        if 'barostat' in kwargs:
            metadata['Barostat'] = kwargs['barostat']
        if 'lx' in kwargs:
            metadata['Geometry'][0]['lx'] = kwargs['lx']
        if 'h' in kwargs:
            metadata['Geometry'][1]['h'] = kwargs['h']
        if 'Nf' in kwargs:
            metadata['Geometry'][2]['Nf'] = kwargs['Nf']
        if 'temperature' in kwargs:
            metadata['State Vars'][0]['temperature'] = kwargs['temperature']
        if 'density' in kwargs:
            metadata['State Vars'][1]['density'] = kwargs['density']
        if 'press' in kwargs:
            metadata['State Vars'][2]['press'] = kwargs['press']
        if 'delta_p' in kwargs:
            metadata['State Vars'][3]['delta_p'] = kwargs['delta_p']
        if 'mflowrate' in kwargs:
            metadata['State Vars'][4]['mflowrate'] = kwargs['mflowrate']
        if 'pump_size' in kwargs:
            metadata['State Vars'][5]['pump_size'] = kwargs['pump_size']
        if 'u_shear' in kwargs:
            metadata['State Vars'][5]['u_shear'] = kwargs['u_shear']

        # write the readme of the dataset
        write_to = os.path.join(self.sim_uri, 'README.yml')
        # if write:
        with open(write_to, "w") as f:
            self.yaml.dump(metadata, f)

        return metadata
#
