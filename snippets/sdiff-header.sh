#!/bin/bash

fileOne=${1}
fileTwo=${2}
numLines=${3:-"1"}

# If filenames are not provided, throw an error
if [ -z "$fileOne" ] || [ -z "$fileTwo" ];
then
	echo "ERROR: Insert filename"
	exit
fi

# show only differences (-s) with display width (-w) of 200 columns
sdiff -s -w 200 <(head -n ${numLines} ${fileOne}) <(head -n ${numLines} ${fileTwo})
