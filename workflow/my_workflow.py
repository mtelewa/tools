#!/usr/bin/env python

from fireworks import Firework, FWorker, LaunchPad, ScriptTask, TemplateWriterTask, FileTransferTask, PyTask, Workflow
from fireworks.utilities.filepad import FilePad
import fireworks.core.rocket_launcher as rocket_launcher
import fireworks.queue.queue_launcher as queue_launcher # launch_rocket_to_queue, rapidfire
import fireworks.queue.queue_adapter as queue_adapter
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
import fw_funcs
# Extract info from the keychain
import keyring
#Check which python interpreter
# print(sys.executable)
import os, glob, sys, datetime, subprocess, itertools
# For remote transfer
import paramiko
# For launching on the queue
# from fabric.api import run, env  >> deprecated in fabric 2+
import fabric


# SSH key file
key_file = os.path.expanduser('~/.ssh/id_rsa')

# host = 'login1.nemo.uni-freiburg.de'
host = 'uc2.scc.kit.edu'

# user = 'ka_lr1762'
user = 'lr1762'

# workspace = '/work/ws/nemo/ka_lr1762-my_workspace-0/'
workspace = '/pfs/work7/workspace/scratch/lr1762-simulations-0'

# Current Working directory
prefix = os.getcwd()

# FireWorks config directory
local_fws = os.path.expanduser('~/.fireworks')

# Test connection with Fabric
connection = fabric.connection.Connection(host, user=user, connect_kwargs=
                                {"key_filename": key_file})

# set up the LaunchPad and reset it
lp = LaunchPad.auto_load()

# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()

project_id = 'equilibrate'



# Initialize ----------------------------------------

parametric_dimension_labels = ['nUnitsX', 'nUnitsY', 'density']

parametric_dimensions = [ {
    'nUnitsX':             [6, 12, 18],
    'nUnitsY':             [10],
    'density':             [0.7],
    'fluid'  :             ['pentane']
    }
]

parameter_sets = list(
    itertools.chain(*[
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels, s)) for s in parameter_sets ]

fw_list = []

# Create a dummy root fw

for i in range(len(parameter_sets)):
    ft = ScriptTask.from_str('echo "Start the Workflow"')
    root_fw = Firework([ft],
        name = 'Load Files',
        spec = {'_category': 'cmsquad35',
                'metadata': {'project': project_id,
                            'datetime': datetime.datetime.now()}
                })

fw_list.append(root_fw)



for i in range(len(parameter_sets)):

    # Create the local datasets
    # ds_local = PyTask(func='fw_funcs.create_local_ds', args=['equilib-{}'.format(i)])

    subprocess.call(['if [ ! -d "equilib-{0}" ]; \
                        echo yes ;\
                        then cp -r equilib equilib-{0}; \
                        else cp -r equilib/* equilib-{0}; \
                    fi'.format(i)], shell=True)


    # Create the remote datasets
    ds_remote = PyTask(func='fw_funcs.create_remote_ds', args=['equilib-{}'.format(i)])

    firework_create_ds = Firework([ds_remote],
                             name = 'Create Dataset',
                             spec = {'_category' : 'cmsquad35',
                                     '_dupefinder': DupeFinderExact()},
                             parents = [root_fw])


    fw_list.append(firework_create_ds)

    # Local Machine directories
    local_equilib = os.path.join(prefix,'equilib-{}'.format(i),'data')
    local_blocks = os.path.join(local_equilib, 'blocks')
    local_moltemp = os.path.join(local_equilib,'moltemp')

    # Remote workspace directories
    workspace_equilib = os.path.join(workspace,'equilib-{}'.format(i),'data')
    workspace_blocks = os.path.join(workspace_equilib, 'blocks')

    # Initialization FW----------------

    build = ScriptTask.from_str('cd {} ; init_moltemp_walls.py {} {} 1 50 {} {}'.
            format(local_moltemp, parameter_sets[i][0], parameter_sets[i][1], parameter_sets[i][2], parameter_sets[i][3]))
    init = ScriptTask.from_str('cd {} ; ./setup.sh'.format(local_moltemp))#, {'stdout_file': 'a.out'})

    firework_init = Firework([build, init],
                             name = 'Initialize',
                             spec = {'_category' : 'cmsquad35',
                                     '_dupefinder': DupeFinderExact()},
                             parents = [firework_create_ds])

    fw_list.append(firework_init)

    # firework_init.to_file('init.yaml')

    # blocks_files = sorted(glob.glob(os.path.join(local_blocks,'*')))
    # infiles = [os.path.join(local_equilib, f) for f in os.listdir(local_equilib)
    #                                     if os.path.isfile(os.path.join(local_equilib, f))]

    # Transfer to the cluster ----------------------------------
    remote_transfer = FileTransferTask({'files': sorted(glob.glob(os.path.join(local_blocks,'*'))),
                                        'dest': workspace_blocks,
                                        'mode': 'rtransfer',
                                        'server': host,
                                        'user': user})

    remote_transfer2 = FileTransferTask({'files': [os.path.join(local_equilib, f)
                                                for f in os.listdir(local_equilib)
                                                if os.path.isfile(os.path.join(local_equilib, f))],
                                        'dest': workspace_equilib,
                                        'mode': 'rtransfer', 'server': host,
                                        'user': user})

    firework_transfer = Firework([remote_transfer, remote_transfer2],
                                 name = 'Transfer',
                                 spec = {'_category': 'cmsquad35',
                                         '_dupefinder': DupeFinderExact()},
                                 parents = [firework_init])

    fw_list.append(firework_transfer)
    # firework_transfer.to_file('transfer.yaml')

    # Equilibrate ----------------------------------------------

    equilibrate = ScriptTask.from_str('mpirun --bind-to core --map-by core -report-bindings lmp_mpi -in $(pwd)/equilib.LAMMPS')

    firework_equilibrate = Firework(equilibrate,
                                    name = 'Equilibrate',
                                    spec = {'_category': '{}'.format(host),
                                            '_dupefinder': DupeFinderExact()},
                                    parents = [firework_transfer])

    fw_list.append(firework_equilibrate)
    # firework_equilibrate.to_file('equilibrate.yaml')


    # Post_process ----------------------------------------------

    postproc = ScriptTask.from_str('bash $HOME/post_proc.sh equilib pentane')

    firework_postproc = Firework(postproc,
                                    name = 'Post-process',
                                    spec = {'_category': '{}'.format(host),
                                            '_dupefinder': DupeFinderExact()},
                                    parents = [firework_equilibrate])

    fw_list.append(firework_postproc)
    # firework_equilibrate.to_file('equilibrate.yaml')

# Launch the Workflow ---------------------------------------
wf = Workflow(fw_list,
    name = 'test_wf',
    metadata = {
        'project': 'equilibrate',
        'fluid'  : 'pentane'
        'datetime': datetime.datetime.now(),
        'type':    'benchmark'
    })

#Store the workflow to the db
lp.add_wf(wf)

#Write out the Workflow to a flat file
wf.to_file('wf.yaml')


# Launch the fireworks on the local machine
rocket_launcher.rapidfire(lp, FWorker('%s/fworker_cms.yaml'.format(local_fws)))

# Remote fireworks will be FIZZLED so rerun them
subprocess.call(['source $HOME/fireworks/bin/activate; \
                 lpad rerun_fws -s FIZZLED'], shell=True)

# Launch on the queue the FWs that were FIZZLED
for i in range(len(parameter_sets)):
    connection.run("cd {};\
                   source $HOME/fireworks/bin/activate ;\
                   qlaunch singleshot".format(os.path.join(workspace,'equilib-{}'.format(i),'data')))





exit()






# launch workflow locally (rlaunch) and remotely (qlaunch)
# rocket_launcher.launch_rocket(lp, FWorker('%s/fworker_cms.yaml'.format(local_fws)))

# queue_launcher.launch_rocket_to_queue(lp, FWorker('/home/mohamed/.fireworks/fworker_uc2.yaml'),
#             queue_adapter.QueueAdapterBase({'_qadapter':'/home/mohamed/.fireworks/qadapter_uc2.yaml'}))



# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}

print(fp.filepad.count_documents(query))

# connection.run('shell_source  ; qlaunch singleshot')


# datafiles = sorted(glob.glob(os.path.join(data_prefix,'*')))
# files = { os.path.basename(f): f for f in datafiles }

# def transfer(src,dest):
#     # create the Firework
#
#     firetask1 = FileTransferTask({'files': [{'src': '%s/out/h.txt' %src, 'dest': '%s' %dest},
#                                             {'src': '%s/out/comZ.txt' %src, 'dest': '%s' %dest}],'mode': 'copy'})
#     firetask2 = ScriptTask.from_str('lmp_mpi < %s/load.LAMMPS' %src)
#     # firetask3 = ScriptTask.from_str('get_com.py %g' %skip)
#     # firetask4 = ScriptTask.from_str('')
#
#     firework = Firework([firetask1,firetask2])
#
#     #Write out the Workflow to a flat file
#     fw_yaml = firework.to_file("my_firework.yaml")
#
#     # store workflow and launch it locally
#     lp.add_wf(firework)
#     rapid_fire(lp,FWorker())


# transfer('/home/mohamed/tools/workflow/03-load', '/home/mohamed/tools/workflow/04-flow/b')
