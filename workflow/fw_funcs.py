#!/usr/bin/env python

import fabric
import os

host = 'uc2.scc.kit.edu'
user = 'lr1762'
key_file = os.path.expanduser('~/.ssh/id_rsa')
workspace = '/pfs/work7/workspace/scratch/lr1762-simulations-0'

# Test connection with Fabric
connection = fabric.connection.Connection(host, user=user, connect_kwargs=
                                {"key_filename": key_file})

def create_local_ds(dir):
    # Create the local datasets
    subprocess.call(['if [ ! -d "{0}" ]; \
                        echo yes ;\
                        then cp -r equilib {0}; \
                        else cp -r equilib/* {0}; \
                    fi'.format(dir)], shell=True)

def create_remote_ds(dir):
    connection.run('source $HOME/fireworks/bin/activate; \
                            if [ ! -d "{0}/{1}" ]; \
                                then cd {0} ; \
                                dtool create {1} ; \
                                cd {1} ; \
                                cd data ; \
                                mkdir out blocks; \
                            fi'.format(workspace, dir))
