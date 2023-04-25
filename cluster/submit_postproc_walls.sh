#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=2:00:00
#SBATCH --partition=single
#SBATCH --output=cluster.out
#SBATCH --error=cluster.err
#SBATCH --job-name=Proc
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mohamed.hassan@kit.edu

export KMP_AFFINITY=compact,1,0

module load devel/python/3.8.6_intel_19.1

source $HOME/venv/bin/activate

module load compiler/intel/19.1
module load mpi/openmpi/4.0

printenv

## check if $SCRIPT_FLAGS is "set"
if [ -n "${SCRIPT_FLAGS}" ] ; then
   ## but if positional parameters are already present
   ## we are going to ignore $SCRIPT_FLAGS
   if [ -z "${*}"  ] ; then
      set -- ${SCRIPT_FLAGS}
   fi
fi

while getopts ":h:i:N:f:s:e:p:x:y:t:T:" option; do
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
    y) Ny="$OPTARG"
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

args=("$@")
echo ${args[@]} > $(pwd)/flags.txt

# call the shift command at the end of the processing loop to remove options that
# have already been handled from $@
shift $((OPTIND - 1))

# Default value for Ny is 1
if [ -n "$Ny" ]; then
  Ny="$Ny"
else
  Ny=1
fi

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

cd $(pwd)

mpirun --bind-to core --map-by core -report-bindings proc_nc.py $infile.nc $Nchunks 1 1000 $fluid $stable_start $stable_end \
        $pump_start $pump_end --Ny $Ny --tessellate $tessellate --TW_interface $TW_interface
mpirun --bind-to core --map-by core -report-bindings proc_nc.py $infile.nc 1 $Nchunks 1000 $fluid $stable_start $stable_end \
        $pump_start $pump_end --Ny $Ny --tessellate $tessellate --TW_interface $TW_interface

if [ ! -f ${infile}_${Nchunks}x1_001.nc ]; then
  mv ${infile}_${Nchunks}x1_000.nc ${infile}_${Nchunks}x1.nc
  mv ${infile}_1x${Nchunks}_000.nc ${infile}_1x${Nchunks}.nc
else
  cdo mergetime ${infile}_${Nchunks}x1_*.nc ${infile}_${Nchunks}x1.nc ; rm ${infile}_${Nchunks}x1_*
  cdo mergetime ${infile}_1x${Nchunks}_*.nc ${infile}_1x${Nchunks}.nc ; rm ${infile}_1x${Nchunks}_*
fi
