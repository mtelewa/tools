#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from ruamel.yaml import YAML
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import dtoolcore
from pathlib import Path
from dtool_create.dataset import _get_readme_template

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
        try:
            self.sim_dataset = dtoolcore.ProtoDataSet.from_uri(self.dataset_name)
        except TypeError:
            self.sim_dataset = dtoolcore.DataSet.from_uri(self.dataset_name)

    def write_readme(self, **kwargs):

        # Simulation dataset readme template
        template = os.path.join(os.path.expanduser('~'), '.dtool', 'custom_dtool_readme.yml')
        # load readme template and update
        readme_template = _get_readme_template(template)
        metadata = self.yaml.load(readme_template)

        metadata['expiration_date'] = metadata['creation_date'] + relativedelta(years=10)

        if kwargs:
            metadata['Independent Vars'] = kwargs

        # write the readme of the post-proc dataset
        with open(template, "w") as f:
            self.yaml.dump(metadata, f)

        return metadata


    def update_readme(self, **kwargs):

        # Simulation dataset readme template
        template = os.path.join(self.sim_uri, 'README.yml')
        # load readme template and update
        readme_template = _get_readme_template(template)
        metadata = self.yaml.load(readme_template)

        metadata['expiration_date'] = metadata['creation_date'] + relativedelta(years=10)

        if 'uuid' in kwargs:
            metadata['derived_from'] = kwargs
        if 'description' in kwargs:
            metadata['description'] = kwargs['description']
        else:
            metadata['Independent Vars'] = kwargs

        # write the readme of the post-proc dataset
        with open(template, "w") as f:
            self.yaml.dump(metadata, f)

        return metadata

    def create_post(self, freeze=None, copy=None):

        # Post-proc dataset path
        post_uri = str(self.sim_uri)+'-post'
        # Post-proc dataset name
        post_name = self.dataset_name +'-post'
        # Post-proc dataset readme template
        post_template = os.path.join(post_uri, 'README.yml')


        template = os.path.join(self.sim_uri, 'README.yml')
        # # load readme template and update
        readme_template = _get_readme_template(template)
        metadata = self.yaml.load(readme_template)

        metadata['creation_date'] = datetime.today()
        metadata['expiration_date'] = metadata['creation_date'] + relativedelta(years=10)
        metadata['derived_from']['uuid'] = self.sim_dataset.uuid

        # Create the derived dataset if not already existing
        if not os.path.isdir(post_name):
            dds = dtoolcore.create_derived_proto_dataset(post_name, str(self.base_uri),
                        self.sim_dataset)
        else:
            dds = dtoolcore.ProtoDataSet.from_uri(post_uri)

        # Copy the files to the post-proc dataset
        for root, dirs, files in os.walk(self.sim_out_uri):
            for i in files:
                if 'x' in i:
                    dds.put_item(os.path.join(self.sim_out_uri,i), i)

        # write the readme of the post-proc dataset
        with open(post_template, "w") as f:
            self.yaml.dump(metadata, f)

        if freeze:
            print(f'Freezing Simulation dataset: {self.sim_uri} ----')
            self.sim_dataset.freeze()
        if copy:
            print(f'Copying dataset: {dds} ----')
            dtoolcore.copy(dds.uri, '/work3')

        return dds


    def create_derived(self, derived_uri, **kwargs):

        # # derived dataset readme template
        derived_template = os.path.join(derived_uri, 'README.yml')

        template = os.path.join(self.sim_uri, 'README.yml')
        # load readme template and update
        readme_template = _get_readme_template(template)
        metadata = self.yaml.load(readme_template)

        metadata['creation_date'] = datetime.today()
        metadata['expiration_date'] = metadata['creation_date'] + relativedelta(years=10)
        metadata['derived_from']['uuid'] = self.sim_dataset.uuid

        if kwargs:
            metadata['Independent Vars'].update(kwargs)

        # write the readme of the post-proc dataset
        with open(derived_template, "w") as f:
            self.yaml.dump(metadata, f)



if __name__ == "__main__":

    man=manipulate_ds(sys.argv[1])

    if 'update_readme' in sys.argv:
        man.update_readme(**dict(arg.split('=') for arg in sys.argv[3:]))
    elif 'create_post' in sys.argv:
        man.create_post()
    elif 'create_derived' in sys.argv:
        man.create_derived(sys.argv[2],**dict(arg.split('=') for arg in sys.argv[4:]))
    elif 'write_readme' in sys.argv:
        man.write_readme()


        # subprocess.call(['rm '], shell=True)
