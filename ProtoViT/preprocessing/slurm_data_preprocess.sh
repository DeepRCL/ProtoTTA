#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --account=st-sdutta10-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=kan-head-data-preprocess
#SBATCH -e kan-head-data-preprocess-%j.log
#SBATCH -o kan-head-data-preprocess-%j.log
#SBATCH --mail-user=rudransh@ece.ubc.ca
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
JOB_NAME="kan-head-data-preprocess"
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

python crop_images.py --dataset_dir ../datasets/CUB_200_2011 --output_dir ../datasets/cub200_cropped
python split_dataset.py --dataset_dir ../datasets/CUB_200_2011 --cropped_dir ../datasets/cub200_cropped/cropped --output_dir ../datasets/cub200_cropped
python image_aug_v2.py --input_dir ../datasets/cub200_cropped/train_cropped --output_dir ../datasets/cub200_cropped/train_cropped_augmented

log INFO "finished job: ${JOB_NAME}"