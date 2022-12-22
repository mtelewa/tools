#! /bin/sh

# SUB="norton"
for i in $(ls -d */);do
	#if [[ $i == *"$SUB"* ]] ; then
	cd $i/data/out
	  rm cluster.* flow_*
		cp $HOME/tools/cluster/submit_postproc.sh .
		sbatch submit_postproc.sh -i flow -N 144 -f pentane -s 0.4 -e 0.8 -p 0.0 -x 0.2
	cd ../../../
	#fi
done
