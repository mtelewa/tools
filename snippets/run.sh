#! /bin/sh

# Take the command from the terminal

SUB="norton"
for i in $(ls -d */);do
	echo $i
	if [[ $i == *"$SUB"* ]] ; then
		echo $i
		cd $i/data/out
	        rm flow_1*
		post_proc.sh flow pentane	
		cd ../../../
	fi
done
