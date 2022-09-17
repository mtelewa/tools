#! /bin/sh

# SUB="norton"
for i in $(ls -d */);do
	#if [[ $i == *"$SUB"* ]] ; then
	cd $i/data
		sbatch submit_sim.sh
	cd ../../
	#fi
done
