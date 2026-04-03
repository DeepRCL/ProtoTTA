#!/bin/bash

#SBATCH --time=00:60:00
#SBATCH --account=st-sdutta10-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=kan-head-data-extract
#SBATCH -e kan-head-data-extract-%j.log
#SBATCH -o kan-head-data-extract-%j.log
#SBATCH --mail-user=rudransh@ece.ubc.ca
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
JOB_NAME="kan-head-data-extract"
VENV_PATH="/scratch/st-sdutta10-1/rudransh/kan-head/checkpoint_2/.venv/bin/activate"

log() {
    local level="$1"
    shift
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[SlurmJobScript]-[$level]-[$timestamp]- $@"
}

log INFO "starting job: ${JOB_NAME}"
log INFO "loading modules"

module load gcc
module load cuda
module load intel-oneapi-compilers/2023.1.0
module load python/3.11.6

log INFO "initial working directory: $(pwd)"
source $VENV_PATH

tar -xvzf /scratch/st-sdutta10-1/rudransh/kan-head/checkpoint_2/proto_baseline/datasets/CUB_200_2011.tgz -C /scratch/st-sdutta10-1/rudransh/kan-head/checkpoint_2/proto_baseline/datasets

log INFO "finished job: ${JOB_NAME}"