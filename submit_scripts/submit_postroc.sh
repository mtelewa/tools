#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH --time=72:00:00
#SBATCH --partition=multiple
#SBATCH --output=cluster.out
#SBATCH --error=cluster.err
#SBATCH --job-name=Proc
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mohamed.hassan@kit.edu

export KMP_AFFINITY=compact,1,0

source $HOME/fireworks/bin/activate

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
# call the shift command at the end of the processing loop to remove options that
# have already been handled from $@
shift $((OPTIND - 1))


cd ${MOAB_SUBMITDIR}

mpirun --bind-to core --map-by core -report-bindings proc.py $infile.nc $Nchunks 1 1000 $fluid $stable_start $stable_end $pump_start $pump_end
mpirun --bind-to core --map-by core -report-bindings proc.py $infile.nc 1 $Nchunks 1000 $fluid $stable_start $stable_end $pump_start $pump_end

if [ ! -f ${infile}_${Nchunks}x1_001.nc ]; then
  mv ${infile}_${Nchunks}x1_000.nc ${infile}_${Nchunks}x1.nc
  mv ${infile}_1x${Nchunks}_000.nc ${infile}_1x${Nchunks}.nc
else
  cdo mergetime ${infile}_${Nchunks}x1_*.nc ${infile}_${Nchunks}x1.nc ; rm ${infile}_${Nchunks}x1_*
  cdo mergetime ${infile}_1x${Nchunks}_*.nc ${infile}_1x${Nchunks}.nc ; rm ${infile}_1x${Nchunks}_*
fi

