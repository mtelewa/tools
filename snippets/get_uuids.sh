#! /bin/sh

# SUB="norton"
for i in $(ls -d *);do
	if [ -d $i ];
	then
	dtool uuid $i
	fi
done
