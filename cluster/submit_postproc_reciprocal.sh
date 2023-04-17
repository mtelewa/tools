#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
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

while getopts ":h:i:X:Y:f:s:e:p:x:" option; do
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

args=("$@")
echo ${args[@]} > $(pwd)/flags.txt

# call the shift command at the end of the processing loop to remove options that
# have already been handled from $@
shift $((OPTIND - 1))

cd $(pwd)

mpirun --bind-to core --map-by core -report-bindings proc_reciprocal.py $infile.nc $longWaveVectors ${transWaveVectors} 1000 $fluid $fluid_start $fluid_end $solid_start $solid_end

if [ ! -f ${infile}_${longWaveVectors}x${transWaveVectors}_001.nc ]; then
  mv ${infile}_${longWaveVectors}x${transWaveVectors}_000.nc ${infile}_${longWaveVectors}x${transWaveVectors}.nc
else
  cdo mergetime ${infile}_${longWaveVectors}x${transWaveVectors}_*.nc ${infile}_${longWaveVectors}x${transWaveVectors}.nc
  rm ${infile}_${longWaveVectors}x${transWaveVectors}_*
fi
