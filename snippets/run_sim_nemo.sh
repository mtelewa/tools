#! /bin/sh

# SUB="norton"
for i in $(ls -d */);do
	#if [[ $i == *"$SUB"* ]] ; then
	cd $i/data
		msub submit_parallel.moab
	cd ../../
	#fi
done
