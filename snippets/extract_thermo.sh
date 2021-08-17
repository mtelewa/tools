#! /bin/sh

cat log.lammps | sed -n "/Step/,/Loop time/p" | head -n-1 > thermo.out

# if a system argument (data start) is given, the thermo out will keep data only starting from the line given
if [[ -n $1 ]]
then
  count=`cat thermo.out | wc -l`
  sed -n "$1, $count p" 'thermo.out' > 'thermo2.out'
  mv thermo2.out thermo.out
fi
