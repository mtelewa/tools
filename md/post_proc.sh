#! /bin/sh

usage="$(basename "$0") [-h -f] [-N] -- program to calculate the answer to life, the universe and everything

where:
    -h  show this help text
    -f  trajectory file
    -N  set the No. of chunks (default: 144)"
 
#Nchunks=144
#infile="file"

while getopts ':hfN:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    f) infile=$OPTARG
       ;;
    N) Nchunks=$OPTARG
       ;;   
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

echo $infile

mpirun -np 8 postproc.py $1.nc 144 1 1000 $2 ${3:-1}
cdo mergetime $1_144x1_*.nc $1_144x1.nc ; rm $1_144x1_*

mpirun -np 8 postproc.py $1.nc 1 144 1000 $2 ${3:-1}
cdo mergetime $1_1x144_*.nc $1_1x144.nc ; rm $1_1x144_*

#mpirun -np 8 mpi_netcdf_postproc.py $1.nc 36 1 1000 $2 ${3:-1}
#cdo merge $1_36x1_*.nc $1_36x1.nc ; rm $1_36x1_*
