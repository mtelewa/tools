#! /bin/sh

# Take the command from the terminal

SUB="norton"
for i in $(ls -d */);do
	cd $i/data/out
        rm cluster.* vs*
	post_proc.sh -i nvt -N 144 -f pentane -s 0.4 -e 0.8 -p 0.0 -x 0.2
	cd ../../../
	#if [[ $i == *"$SUB"* ]] ; then
		#echo $i
		#cd $i/data/out
	        #rm flow_1*
		#post_proc.sh flow pentane	
		#cd ../../../
	#fi
done
