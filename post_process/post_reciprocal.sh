#! /bin/sh

usage="$(basename "$0") [-h] [-i -N] -- run the netcdf post-processor

where:
    -h   show this help text
    -i   trajectory file
    -X   set the No. of longitudnal wavevectors
    -Y   set the No. of transverse wavevectors
    -f   Fluid
    -s   fluid start
    -e   fluid end
    -p   solid start
    -x   solid end
    -z   nz
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

while getopts ":h:i:X:Y:f:s:e:p:x:z:T:" option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    i) infile="$OPTARG"
    ;;
    X) longWaveVectors="$OPTARG"
    ;;
    Y) transWaveVectors="$OPTARG"
    ;;
    f) fluid="$OPTARG"
    ;;
    s) fluid_start="$OPTARG"
    ;;
    e) fluid_end="$OPTARG"
    ;;
    p) solid_start="$OPTARG"
    ;;
    x) solid_end="$OPTARG"
    ;;
    z) nz="$OPTARG"
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

# Default value for Ny is 1
if [ -n "$nz" ]; then
  nz="$nz"
else
  nz=1
fi

# Default value for TW_interface is 1
if [ -n "$TW_interface" ]; then
  TW_interface=0
else
  TW_interface=1
fi

mpirun -np 8 proc_reciprocal.py $infile.nc $longWaveVectors ${transWaveVectors} 1000 \
        $fluid $fluid_start $fluid_end $solid_start $solid_end --nz $nz --TW_interface $TW_interface

if [ ! -f ${infile}_${longWaveVectors}x${transWaveVectors}_001.nc ]; then
  mv ${infile}_${longWaveVectors}x${transWaveVectors}_000.nc ${infile}_${longWaveVectors}x${transWaveVectors}.nc
else
  cdo mergetime ${infile}_${longWaveVectors}x${transWaveVectors}_*.nc ${infile}_${longWaveVectors}x${transWaveVectors}.nc
  rm ${infile}_${longWaveVectors}x${transWaveVectors}_*
fi
