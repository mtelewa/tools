#! /bin/sh

usage="$(basename "$0") [-h] [-i -N] -- run the netcdf post-processor

where:
    -h   show this help text
    -i   trajectory file
    -N   set the No. of chunks
    -f   Fluid
    -s   stable start
    -e   stable end
    -p   pump start
    -x   pump end"

# Flags ----------------
# This is a while loop that uses the getopts function and a so-called optstring
# to iterate through the arguments.
# The while loop walks through the optstring, which contains the flags that are
# used to pass arguments and assigns the argument value provided for that flag to
# the variable option.
# The case statement then assigns the value of the variable option to a global
# variable that is used after all the arguments have been read.

#The colons in the optstring mean that values are required for the corresponding flags.
#In the above example of u:d:p:f:, all flags are followed by a colon.
#This means all flags need a value. If, for example,
#the d and f flags were not expected to have a value, u:dp:f would be the optstring.

while getopts ":h:i:N:f:s:e:p:x:" option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    i) infile="$OPTARG"
    ;;
    N) Nchunks="$OPTARG"
    ;;
    f) fluid="$OPTARG"
    ;;
    s) stable_start="$OPTARG"
    ;;
    e) stable_end="$OPTARG"
    ;;
    p) pump_start="$OPTARG"
    ;;
    x) pump_end="$OPTARG"
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

# Write the arguments to a text file
args=("$@") ;
echo "Flags: ${args[@]}" > flags.txt

# call the shift command at the end of the processing loop to remove options that
# have already been handled from $@
shift $((OPTIND - 1))

mpirun -np 8 proc.py $infile.nc $Nchunks 1 1000 $fluid $stable_start $stable_end $pump_start $pump_end
mpirun -np 8 proc.py $infile.nc 1 $Nchunks 1000 $fluid $stable_start $stable_end $pump_start $pump_end

if [ ! -f ${infile}_${Nchunks}x1_001.nc ]; then
  mv ${infile}_${Nchunks}x1_000.nc ${infile}_${Nchunks}x1.nc
  mv ${infile}_1x${Nchunks}_000.nc ${infile}_1x${Nchunks}.nc
else
  cdo mergetime ${infile}_${Nchunks}x1_*.nc ${infile}_${Nchunks}x1.nc ; rm ${infile}_${Nchunks}x1_*
  cdo mergetime ${infile}_1x${Nchunks}_*.nc ${infile}_1x${Nchunks}.nc ; rm ${infile}_1x${Nchunks}_*
fi

