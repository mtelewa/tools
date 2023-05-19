#! /bin/sh

# SUB="norton"
for i in $(ls -d *);do
	dtool uuid $i
done
