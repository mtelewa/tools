#! /bin/sh

# Take the command from the terminal

SUB="norton"
for i in $(ls -d */);do
	cd $i/data/out
	cp $HOME/tools/submit_scripts/submit_postroc.moab .
	msub submit_postroc.moab -v SCRIPT_FLAGS='-i flow -N 144 -f pentane -s 0.4 -e 0.8 -p 0.0 -x 0.2'
	#get_variables.py 5000 pentane mflux flow_144x1.nc flow_1x144.nc
	cd ../../../
	#if [[ $i == *"$SUB"* ]] ; then
		#echo $i
		#cd $i/data/out
	        #rm flow_1*
		#post_proc.sh flow pentane	
		#cd ../../../
	#fi
done
