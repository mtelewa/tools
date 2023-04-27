#! /bin/sh

usage="$(basename "$0") [-h] [-i -N] -- run the netcdf post-processor

where:
    -h   show this help text
    -i   trajectory file
    -X   set the No. of chunks in x-direction
    -Y   set the No. of chunks in y-direction
    -Z   set the No. of chunks in z-direction
    -f   Fluid
    -s   stable start
    -e   stable end
    -p   pump start
    -x   pump end
    -t   tessellate
    -T   TW_interface"

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

while getopts ":h:i:X:Y:Z:f:s:e:p:x:t:T:" option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    i) infile="$OPTARG"
    ;;
    X) NchunksX="$OPTARG"
    ;;
    Y) NchunksY="$OPTARG"
    ;;
    Z) NchunksZ="$OPTARG"
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
    t) tessellate="$OPTARG"
    ;;
    T) TW_interface="$OPTARG"
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

# Default value for tessellate is 0
if [ -n "$tessellate" ]; then
  tessellate=1
else
  tessellate=0
fi

# Default value for TW_interface is 1
if [ -n "$TW_interface" ]; then
  TW_interface=0
else
  TW_interface=1
fi

mpirun -np 8 proc_4d.py $infile.nc $NchunksX $NchunksY $NchunksZ 1000 $fluid \
                    $stable_start $stable_end $pump_start $pump_end --tessellate $tessellate --TW_interface $TW_interface

if [ ! -f ${infile}_${NchunksX}x${NchunksY}x${NchunksZ}_001.nc ]; then
  mv ${infile}_${Nchunks}x${NchunksY}x${NchunksZ}_000.nc ${infile}_${NchunksX}x${NchunksY}x${NchunksZ}.nc
else
  cdo mergetime ${infile}_${NchunksX}x${NchunksY}x${NchunksZ}_*.nc ${infile}_${NchunksX}x${NchunksY}x${NchunksZ}.nc
  rm ${infile}_${NchunksX}x${NchunksY}x${NchunksZ}_*
fi
