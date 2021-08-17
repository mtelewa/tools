#!/usr/bin/env python
# coding: utf-8

# # Run LAMMPS benchmark via Fireworks

# Johannes Hörmann, Feb 2020
#
# Based on
#
# * https://git.scc.kit.edu/jk7683/grk2450-fireworks-tutorial/blob/master/demos/7_filepad/doc/FilePad_slides.pdf
# * https://git.scc.kit.edu/jk7683/grk2450-fireworks-tutorial/blob/master/demos/7_filepad/PythonGoldCluster.ipynb

# ## Initialization

# ### Imports

# to display plots directly inline within this notebook
#get_ipython().run_line_magic('matplotlib', 'inline')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# we might need these for
# * enforced memory clean-up (gc)
# * bash-like globbing of files on the local file system (glob)
# * basic stream-from-string input (io)
# * outer products of lists (itertools)
# * extracting interseting information from log files by regular expressions (re)
# * os, sys ...
import datetime, gc, glob, io, itertools, os, re, sys, ruamel.yaml
os.environ["OMP_NUM_THREADS"] = "1"

# import ase, ase.io, ase.visualize # read and visualize LAMMPS trajectories
from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import scipy as scp # here for handling rotations as matrices and quaternions
import scipy.spatial
import scipy.optimize as opt
from pprint import pprint

# FireWorks functionality
import pymongo
from fireworks import Firework, LaunchPad, ScriptTask, Workflow
# new here: tasks using FireWorks FilePad
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, GetFilesByQueryTask
# from jlhfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
# direct FilePad access, similar to the familiar LaunchPad:
from fireworks.utilities.filepad import FilePad

prefix = os.getcwd()
# to start with
infile_prefix = os.path.join(prefix,'infiles')
data_prefix = os.path.join(prefix,'data')

print(data_prefix)

# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()

# some unique identifier for our study
# project_id = 'forhlr2-lmp-bench-2020-01-21'
project_id = 'juwels-lmp-bench-2020-02-05-b'

# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}

# use underlying MongoDB functionality to check total number of documents matching query

fp.filepad.count_documents(query)

# if necessary, remove all files matching query
# fp.delete_file_by_query(query)

# ### Managing input files

infiles = sorted(glob.glob(os.path.join(infile_prefix,'in.*')))

print(infiles)

files = { os.path.basename(f): f for f in infiles }

print(files)

# metadata common to all these files
metadata = {
    'project': project_id,
    'type': 'input'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.type': 'input'
}

# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)

# we fixed simple identifiers of this form
print(identifier)

# on a lower level, each object has a unique "GridFS id":
pprint(fp_files) # underlying GridFS id and readable identifiers

# if necessary, remove all files matching query
#fp.delete_file_by_query(query)


# ### Managing data files

datafiles = sorted(glob.glob(os.path.join(data_prefix,'*')))

files = { os.path.basename(f): f for f in datafiles }

pprint(files)

bench_data_mapping = {
    'in.intel.sds': {
        'datafile.lammps': 'interface_AU_111_150Ang_cube_SDS_646_hemicylinders_equilibration_dpd_201907121627_default.lammps',
        'coeff.input':     'SDS_in_H2O_on_AU_coeff_hybrid_lj_charmmfsw_coul_long.input',
        'Au-Grochola-JCP05-units-real.eam.alloy': 'Au-Grochola-JCP05-units-real.eam.alloy'
    },
    'in.intel.shear': {
        'temp10_centered.data': 'temp10_centered.data'
    }
}

# file name : file path dict
files = {
    f: os.path.join(data_prefix,f) for b, d in bench_data_mapping.items() for n, f in d.items()
}


for p in files.values():
    assert len(glob.glob(p)) > 0, "Data file does not exist"

# metadata common to all these files
metadata = {
    'project': project_id,
    'type': 'data'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )

# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.type': 'data'
}

# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)

# we fixed simple identifiers of this form
print(identifier)

# on a lower level, each object has a unique "GridFS id":
pprint(fp_files) # underlying GridFS id and readable identifiers

# if necessary, remove all files matching query
#fp.delete_file_by_query(query)


# ### Machine-specific settings

hpc_max_specs = {
    'forhlr2': {
        'fw_queue_category':   'forhlr2_queue',
        'fw_noqueue_category': 'forhlr2_noqueue',
        'queue':'develop',
        'physical_cores_per_node': 20,
        'logical_cores_per_node':  40,
        'nodes': 4,
        'walltime':  '00:60:00'
    },
    'juwels_devel': {
        'fw_queue_category':   'juwels_queue',
        'fw_noqueue_category': 'juwels_noqueue',
        'queue':'devel',
        'physical_cores_per_node': 48,
        'logical_cores_per_node':  96,
        'nodes': 8,
        'walltime':  '00:30:00'
    },
    'juwels': {
        'fw_queue_category':   'juwels_queue',
        'fw_noqueue_category': 'juwels_noqueue',
        'queue':'batch',
        'physical_cores_per_node': 48,
        'logical_cores_per_node':  96,
        'nodes': 1024,
        'walltime':  '00:30:00'
    }
}


std_exports = {
    'forhlr2': {
        'OMP_NUM_THREADS': 1,
        'KMP_AFFINITY':    "'verbose,compact,1,0'",
        'I_MPI_PIN_DOMAIN':'core'
    },
    'juwels': {
        'OMP_NUM_THREADS': 1,
        'KMP_AFFINITY':    "'verbose,compact,1,0'",
        'I_MPI_PIN_DOMAIN':'core'
    }
}


# d = 1 to use 'diff ad' for PPPM in some benchmarks
lmp_std_args=["-screen","none","-v","d",1]


# ### Benchmark modes

def get_mode_dict(nthreads_per_task=2):
    threads = nthreads_per_task
    rthreads = threads - 1
    # OPT
    mode_dict = {
        "lmp_std": { "args": lmp_std_args },

        "lmp_opt": { "args": ["-sf","opt",*lmp_std_args] },

    # OMP
        "lmp_omp": { "args": ["-sf","omp","-pk","omp",threads,*lmp_std_args] },

    # USER-INTEL
        "lmp_intel_single": { "args": ["-sf","intel","-pk","intel",0,"omp",threads,"mode","single",*lmp_std_args] },
        "lmp_intel_double": { "args": ["-sf","intel","-pk","intel",0,"omp",threads,"mode","double",*lmp_std_args] },

    # USER-INTEL with LRT (long range thread)
    # export KMP_AFFINITY=none
        "lmp_intel_single_lrt": {
            "args": ["-sf","intel","-pk","intel",0,"omp",rthreads,"mode","single","lrt","yes",*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },
        "lmp_intel_double_lrt": {
            "args": ["-sf","intel","-pk","intel",0,"omp",rthreads,"mode","double","lrt","yes",*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },

    # HYBRID INTEL-OPT
        "lmp_hybrid_intel_single_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","single",*lmp_std_args] },
        "lmp_hybrid_intel_double_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","double",*lmp_std_args] },
        "lmp_hybrid_intel_single_lrt_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","single","lrt","yes",*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },
        "lmp_hybrid_intel_double_lrt_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","double","lrt","yes",*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },

    # HYBRID INTEL-OPT NEWTON OFF
        "lmp_hybrid_intel_single_noff_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","single","-v","N","off",*lmp_std_args] },
        "lmp_hybrid_intel_double_noff_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","double","-v","N","off",*lmp_std_args] },
        "lmp_hybrid_intel_single_lrt_noff_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","single","lrt","yes","-v","N","off",*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },
        "lmp_hybrid_intel_double_lrt_noff_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","double","lrt","yes","-v","N","off",*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },

    # HYBRID INTEL-OPT, NUMA MAPPING ON
        "lmp_hybrid_intel_single_numa_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","single","-v","m",1,*lmp_std_args] },
        "lmp_hybrid_intel_double_numa_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","double","-v","m",1,*lmp_std_args] },
        "lmp_hybrid_intel_single_lrt_numa_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","single","lrt","yes","-v","m",1,*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },
        "lmp_hybrid_intel_double_lrt_numa_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","double","lrt","yes","-v","m",1,*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },

    # HYBRID INTEL-OPT, PPPM COLLECTIVES
        "lmp_hybrid_intel_single_collectives_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","single","-v","c",1,*lmp_std_args] },
        "lmp_hybrid_intel_double_collectives_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","double","-v","c",1,*lmp_std_args] },
        "lmp_hybrid_intel_single_lrt_collectives_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","single","lrt","yes","-v","c",1,*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },
        "lmp_hybrid_intel_double_lrt_collectives_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","double","lrt","yes","-v","c",1,*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },

    # HYBRID INTEL-OPT, NO DIFF AD
        "lmp_hybrid_intel_single_nodiffad_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","single","-v","d",0,*lmp_std_args] },
        "lmp_hybrid_intel_double_nodiffad_opt": { "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",threads,"mode","double","-v","d",0,*lmp_std_args] },
        "lmp_hybrid_intel_single_lrt_nodiffad_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","single","lrt","yes","-v","d",0,*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },
        "lmp_hybrid_intel_double_lrt_nodiffad_opt": {
            "args": ["-sf","hybrid","intel","opt","-pk","intel",0,"omp",rthreads,"mode","double","lrt","yes","-v","d",0,*lmp_std_args],
            "exports": { "KMP_AFFINITY": "none", "OMP_NUM_THREADS": rthreads }
        },

    # HYBRID INTEL-OMP, failed previously with
    #   No /omp style for force computation currently active
    #   ERROR: Requested neighbor pair method does not exist (../neighbor.cpp:740)

    # LMP_HYBRID_INTEL_SINGLE_OMP_ARGS="-sf hybrid intel omp -pk intel 0 omp ${threads} mode single -pk omp ${threads} ${LMP_STD_ARGS}"
    # LMP_HYBRID_INTEL_DOUBLE_OMP_ARGS="-sf hybrid intel omp -pk intel 0 omp ${threads} mode double -pk omp ${threads} ${LMP_STD_ARGS}"

    # KOKKOS
    # export OMP_PROC_BIND=spread
    # export OMP_PLACES=threads
        "lmp_kokkos_std": {
            "args": ["-k","on","t",threads,"-sf","kk",*lmp_std_args],
            "exports": { "OMP_PROC_BIND": "spread", "OMP_PLACES": "threads" }
        },
        "lmp_kokkos_neigh": {
            "args": ["-k","on","t",threads,"-sf","kk","-pk","kokkos","newton","on","neigh","half",*lmp_std_args],
            "exports": { "OMP_PROC_BIND": "spread", "OMP_PLACES": "threads" }
        },
        "lmp_kokkos_no_comm": {
            "args": ["-k","on","t",threads,"-sf","kk","-pk","kokkos","newton","on","neigh","half","comm","no",*lmp_std_args],
            "exports": { "OMP_PROC_BIND": "spread", "OMP_PLACES": "threads" }
        },
    }
    return mode_dict


# ### Benchmark parameter sets

parametric_dimension_labels = ['nthreads_per_task', 'nodes', 'smt', 'mode', 'system']

# Some sample sets below, pick one (and modify)

parametric_dimensions = {
    'nthreads_per_task': [1,2],
    'nodes':               [1,2,4],
    'smt':                 [True,False],
    'mode':                list(get_mode_dict().keys()),
    'system':              [
        'lj',
        'rhodo',
        'sw',
        'water',
        'eam',
        'sds', # jlh
        'shear' # jm
    ] }

parametric_dimensions = [ {
    'nthreads_per_task':   [1,2,4,8],
    'nodes':               [1],
    'smt':                 [False],
    'mode':                ['lmp_omp'],
    'system':              ['lj','sds', 'shear']
    },{
    'nthreads_per_task':   [1,2,4,8,16],
    'nodes':               [1],
    'smt':                 [True],
    'mode':                ['lmp_omp'],
    'system':              ['lj','sds', 'shear']
    },{
    'nthreads_per_task':   [1],
    'nodes':               [1,2,3,4,5,6,7,8],
    'smt':                 [True],
    'mode':                ['lmp_std'],
    'system':              ['lj','sds', 'shear']
    }
]

parametric_dimensions = [ {
    'nthreads_per_task':   [1],
    'nodes':               [12,16,20,24,28,32],
    'smt':                 [True],
    'mode':                ['lmp_std'],
    'system':              ['sds']
    }
]


parameter_sets = list(
    itertools.chain(*[
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

len(parameter_sets)

parameter_sets[0]

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# ### Check input files in filepad

# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.type': 'input'
}


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)

match_aggregation = {
        "$match": query
    }

# group by parameters of interest, i.e. cluster_size and velocity:
group_aggregation = {
    "$group": {
        "_id": {
            "name": "$metadata.name"
        }
    }
}

aggregation_pipeline = [ match_aggregation, group_aggregation ]

cursor = fp.filepad.aggregate(aggregation_pipeline)

unique_input_file_names = [ c["_id"]["name"] for c in cursor ]

machine = 'juwels'

wf_name = 'LAMMPS benchmark {machine:}, {id:}'.format(machine=machine,id=project_id)

ft = ScriptTask.from_str('echo "Dummy root"')
root_fw = Firework([ft],
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        'metadata': {
          'project': project_id,
          'datetime': datetime.datetime.now(),
          'step':    'dummy_root'
        }
    }
)

fw_list = [root_fw]
fw_dict = {}

# create one input fetch task for each benchmark system
input_name_template = 'in.intel.{}'
datafile_label_template = 'datafile_{:d}'
fw_name_template = 'fetch inputs, system: {system:s}'


for s in pd.DataFrame(parameter_dict_sets)["system"].unique():
    input_name = input_name_template.format(s)

    fts = [
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    input_name
            },
            limit = 1,
            new_file_names = ['bench.input']
            )
        ]

    files_out = {'input_file': 'bench.input'}

    if input_name in bench_data_mapping:
        additional_data_files = benchmark_datafile_mapping[input_name]
        for i, (file_name, fp_metadata_name) in enumerate(additional_data_files.items()):
            fts.append(
                GetFilesByQueryTask(
                    query = {
                        'metadata->project': project_id,
                        'metadata->name':    fp_metadata_name
                    },
                    limit = 1,
                    new_file_names = [ file_name ]
                )
            )
            files_out.update({datafile_label_template.format(i): file_name})


    fw = Firework(fts,
        name = fw_name_template.format(system=s),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': datetime.datetime.now(),
                'step':    'bench_fetch_input_file',
                'system':  s

            }
        },
        parents = [root_fw])
    fw_list.append(fw)
    fw_dict.update({s:fw})

fw_name_template = 'system: {system:s}, mode: {mode:s}, nodes: {nodes:d}, nthreads_per_task: {nthreads_per_task:d}, smt: {smt:}'
for d in parameter_dict_sets:

    # data file labels and names
    input_name = input_name_template.format(d["system"])
    files_in = {'input_file': 'bench.input'}
    if input_name in bench_data_mapping:
        additional_data_files = benchmark_datafile_mapping[input_name]
        for i, file_name in enumerate(additional_data_files.keys()):
            files_in.update({datafile_label_template.format(i): file_name})

    # mode
    mode_dict  = get_mode_dict(nthreads_per_task=d["nthreads_per_task"])

    exports = std_exports[machine].copy()
    if "exports" in mode_dict[d["mode"]]:
        exports.update( mode_dict[d["mode"]]["exports"] )
    exports.update( {'OMP_NUM_THREADS':d['nthreads_per_task']} )
    export_str = ','.join([ "{}={}".format(k,v) for k,v in exports.items() ])

    # SMT: (simultaneous multithreading, synonymous to "hyperthreading" on intel)
    thread_multiplicity = 2 if d['smt'] else 1

    ntasks_per_node = hpc_max_specs[machine]['physical_cores_per_node']//d['nthreads_per_task'] * thread_multiplicity

    ft_bench = CmdTask(
        cmd='lmp',
        opt=['-in','bench.input',*(mode_dict[d["mode"]]["args"])],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True)

    fw_bench = Firework([ft_bench],
        name = ', '.join(('run lmp bench', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_queue_category'],
            '_queueadapter': {
                'queue':           hpc_max_specs[machine]['queue'],
                'walltime' :       hpc_max_specs[machine]['walltime'],
                'ntasks_per_node': ntasks_per_node,
                'ntasks':          ntasks_per_node * d['nodes'],
                'nodes':           d['nodes'],
                'export':          export_str
            },
            '_files_in': files_in,
            '_files_out': {'log_file': 'log.lammps'},
            'metadata': {
                'project': project_id,
                'datetime': datetime.datetime.now(),
                'step':    'bench_run',
                 **d
            }
        },
        parents = [ fw_dict[d["system"]]] )

    # store log in db
    fw_list.append(fw_bench)

    ft_transfer = AddFilesTask( {
        'compress':True ,
        'paths': "log.lammps",
        'metadata': {
            'project': project_id,
            'datetime': datetime.datetime.now(),
            'type':    'log',
             **d }
        } )

    fw_transfer = Firework([ft_transfer],
        name = ', '.join(('transfer lmp bench log', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': {'log_file': 'log.lammps'},
            'metadata': {
                'project': project_id,
                'datetime': datetime.datetime.now(),
                'step':    'log_transfer',
                 **d
            }
        },
        parents = [ fw_bench ] )

    fw_list.append(fw_transfer)

wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': datetime.datetime.now(),
        'type':    'benchmark'
    })


# In[2017]:


# we can write a static yaml file and have a look at, then submit it
# on the command line with "lpad add wf.yaml"...
wf.to_file('wf.yaml')

# or directly submit from here
fw_ids = lp.add_wf(wf)


# ## Query results

parameter_names = ['system', 'mode', 'smt', 'nthreads_per_task', 'nodes']
parameter_dict = { p: [] for p in parameter_names }


query = {
    "metadata.project": project_id,
}

# use underlying MongoDB functionality
fp.filepad.count_documents(query)


query = {
    "metadata.project": project_id,
    "metadata.type":    'log',
    "metadata.datetime": {"$exists": True}
}


# use underlying MongoDB functionality
fp.filepad.count_documents(query)


# ### Extract performance data from logs and attach to db

# Regular expression to extract performance data from LAMMPS logs:
regex_pattern_file = os.path.join(prefix,'lmp_performance.regex')

with open(regex_pattern_file,'r') as f:
    regex_pattern = f.read()
    # use re.DEBUG for detailed logging here
    # regex = re.compile(regex_pattern,flags=(re.MULTILINE | re.DEBUG) )
    regex = re.compile(regex_pattern,flags=(re.MULTILINE) )

sort_aggregation = {
        "$sort": {
            "metadata.datetime": pymongo.DESCENDING,
            **{ 'metadata.{}'.format(p) : pymongo.ASCENDING for p in parameter_names }
        }
    }


match_aggregation = {
        "$match": query
    }


# group by parameters of interest, i.e. cluster_size and velocity:
group_aggregation = {
    "$group": {
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}

aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]

cursor = fp.filepad.aggregate(aggregation_pipeline)

failed_logs = {}

# run through distinct parameter sets and parse latest logs
for i, c in enumerate(cursor):
    content, metadata = fp.get_file_by_id(c["latest"])
    data_raw = {}
    try:
        m = regex.match(content.decode("utf-8", "strict"))
        data_raw[i] = m.groupdict()
        print('.',end='')
    except:
        logger.warning("Error parsing log no. {:d}: {:s}!".format(i,c["latest"]))
        failed_logs.update({c["latest"]: (content,metadata)})
        continue

    try:
        fp.filepad.update_one(
            filter = {"gfs_id": c["latest"]},
            update = {
                "$set": { "metadata.performance_data": m.groupdict() } } )
    except:
        logger.warning("Error updating metadata on log no. {:d}: {:s}!".format(i,c["latest"]))
        continue

    print('.',end='')
print('')


# ### Evaluate performance data

query = {
    "metadata.project": project_id,
    "metadata.type":    'log',
    "metadata.datetime": {"$exists": True},
    "metadata.performance_data": {"$exists": True}
}


fp.filepad.count_documents(query)

match_aggregation = {
        "$match": query
    }

sort_aggregation = {
        "$sort": {
            "metadata.datetime": pymongo.DESCENDING,
            **{ 'metadata.{}'.format(p) : pymongo.ASCENDING for p in parameter_names }
        }
    }


# group by parameters of interest, i.e. cluster_size and velocity:
group_aggregation = {
    "$group": {
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        #"degeneracy": {"$sum": 1}, # number matching data sets
        "performance_data": {"$first": "$metadata.performance_data"} # unique gridfs id of file
    }
}

aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]

cursor = fp.filepad.aggregate(aggregation_pipeline)

data_rows = []
for c in cursor:
    row = c["_id"]
    row.update(c["performance_data"])
    # get types in data row:
    # data_types = { k: type(v) for k,v in row.items() }
    # try to convert strings to floats
    casted_row = {}
    for k, v in row.items():
        if type(v) is str:
            try:
                v = float(v)
            except:
                pass
        casted_row.update({k:v})
    data_rows.append(casted_row)

data_types = { k: type(v) for k,v in data_rows[0].items() }

perf_df = pd.DataFrame(data_rows)

perf_mi_df = perf_df.set_index(parameter_names) # MultiIndex for display purposes

lj_dat = perf_mi_df.loc['lj','lmp_std'].sort_values('timesteps_per_second',ascending=False)
lj_dat

sds_dat = perf_mi_df.loc['sds','lmp_std'].sort_values('timesteps_per_second',ascending=False)

shear_dat = perf_mi_df.loc['shear','lmp_std'].sort_values('timesteps_per_second',ascending=False)

query = {
    "metadata.project": project_id,
    "metadata.type":    'log',
    "metadata.system":  'lj',
    "metadata.mode":    'lmp_std',
    "metadata.nthreads_per_task":    1,
    "metadata.datetime": {"$exists": True},
    "metadata.performance_data": {"$exists": True}
}

fp.filepad.count_documents(query)

files = fp.get_file_by_query(query)


# print(files[0][0].decode())

# print(files[1][0].decode())
