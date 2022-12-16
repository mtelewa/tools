#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import argparse
import man_ds
import itertools

def get_parser():
    parser = argparse.ArgumentParser(
    description='Manipulate datasets: Create parent (simulation), derived(simulation) or post-processing datasets \
     / Update dataset README / Freeze and copy to S3 storage system.')

    #Positional arguments
    #--------------------
    parser.add_argument('action', metavar='action', action='store', type=str,
                    help='What to do with the dataset (create_ds/ create_post/ create_derived/ update_readme) ')
    parser.add_argument('--freeze', metavar='freeze', action='store', type=str,
                    help='Freeze the dataset')
    parser.add_argument('--copy', metavar='copy', action='store', type=str,
                    help='Copy the datset to the S3 system')
    # parser.add_argument('--write', metavar='write', action='store', type=str,
    #                 help='Write the updated README')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        # you can pass any arguments to add_argument
        if arg.startswith(("-ds")): parser.add_argument(arg.split('=')[0], type=str, help='datasets')
        if arg.startswith(("-der")): parser.add_argument(arg.split('=')[0], type=str, help='derived datasets')
        if arg.startswith(("-var")): parser.add_argument(arg.split('=')[0], type=str, help='yaml nodes')
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    datasets, nodes, deriveds = [], [], []
    for key, value in vars(args).items():
        if key.startswith('ds') and value!='all': datasets.append(value)
        if key.startswith('ds') and value=='all':
            for i in os.listdir(os.getcwd()):
                if os.path.isdir(i):
                    datasets.append(i)
        if key.startswith('var'): nodes.append(value)
        if key.startswith('der'): deriveds.append(value)

    # Create the dict of yaml nodes
    k, v = [], []
    for var in nodes:
        k.append(var.split(':')[0])
        v.append(var.split(':')[1])

    variables = dict(zip(k,v))

    # iterate over the parent and derived datasets, default is one parent with many children
    for dataset, derived in itertools.zip_longest(datasets, deriveds, fillvalue=datasets[0]):
        ds = man_ds.manipulate_ds(dataset)
        if args.action=='create_ds': ds.create_dataset(**dict(variables))
        if args.action=='tag_ds': ds.put_tag(**dict(variables))
        if args.action=='create_post' and args.freeze and not args.copy: ds.create_post(freeze='y', **dict(variables))
        if args.action=='create_post' and args.copy and not args.freeze: ds.create_post(copy='y', **dict(variables))
        if args.action=='create_post' and args.freeze and args.copy: ds.create_post(freeze='y',copy='y', **dict(variables))
        if args.action=='create_post' and not args.freeze and not args.copy: ds.create_post(**dict(variables))
        if args.action=='create_derived': ds.create_derived(derived, **dict(variables))
        if args.action=='update_readme': ds.update_readme(**dict(variables))
        # if args.action=='update_readme' and args.write: ds.update_readme(write='y', **dict(variables))


#
