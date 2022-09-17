#! /bin/sh

# SUB="norton"
for i in $(ls -d */);do
	#if [[ $i == *"$SUB"* ]] ; then
	cd $i/data/out
  	rm cluster.* vs*
		post_proc.sh -i nvt -N 144 -f pentane -s 0.4 -e 0.8 -p 0.0 -x 0.2
	cd ../../../
	#fi
done
