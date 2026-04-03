#!/bin/bash
# =============================================================================
# Train ProtoPFormer (deit_small_patch16_224) on Stanford Dogs
#
# NOTE: num_workers=4 and pin_memory=False are set to avoid /dev/shm exhaustion
# (this login node has only 64 MB of shared memory).
#
# Usage:
#   sh scripts/train_dogs_small.sh [num_gpus] [batch_size] [resume_ckpt]
#
# Examples:
#   sh scripts/train_dogs_small.sh          # 1 GPU, batch 64, auto-resume
#   sh scripts/train_dogs_small.sh 1 128    # 1 GPU, batch 128, auto-resume
#   sh scripts/train_dogs_small.sh 2 128    # 2 GPUs, batch 128
#   sh scripts/train_dogs_small.sh 1 128 output_cosine/.../epoch-best.pth
# =============================================================================

set -e
export PYTHONPATH=./:$PYTHONPATH

# ---------- Always use the protovit conda env, regardless of active env ----------
CONDA_PYTHON=/home/mahdi.abootorabi/miniconda3/envs/protovit/bin/python
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: protovit conda env not found at $CONDA_PYTHON"
    exit 1
fi
export PATH="$(dirname $CONDA_PYTHON):$PATH"

# ---------- Arguments (with defaults) ----------
num_gpus=${1:-1}
batch_size=${2:-64}
resume_ckpt=${3:-""}   # Optional: explicit checkpoint path to resume from

# ---------- Model ----------
model=deit_small_patch16_224

# For deit_small_patch16_224, hidden dim = 384 (NOT 192 like deit_tiny)
dim=384

# ---------- Dataset ----------
data_set=Dogs
prototype_num=1200        # 120 classes x 10 local prototypes/class
# Data path: parent of stanford_dogs/
data_path=$(realpath "$(dirname "$0")/../datasets")

# ---------- Reproducibility ----------
seed=1028
use_port=2675

# ---------- Learning Rates ----------
warmup_lr=1e-4
warmup_epochs=5
features_lr=5e-5          # Lower LR for pretrained ViT backbone (fine-tune carefully)
add_on_layers_lr=3e-3
prototype_vectors_lr=3e-3

# ---------- Optimizer & Scheduler ----------
opt=adamw
sched=cosine
decay_epochs=10
decay_rate=0.1
weight_decay=0.05
epochs=200
input_size=224

# ---------- DataLoader (shm-safe) ----------
# /dev/shm on this node is only 64 MB; pin_memory uses shm heavily.
# Using num_workers=4 + no-pin-mem avoids Bus error / shm exhaustion.
num_workers=4

# ---------- ProtoPFormer Specific ----------
use_global=True
use_ppc_loss=True           # PPC loss (prototype compactness & separation)
last_reserve_num=81         # Keep top-81 attended tokens in last layer
global_coe=0.5              # 50% global (CLS-token) + 50% local (patch) branches
ppc_cov_thresh=1.           # Covariance threshold for PPC loss
ppc_mean_thresh=2.          # Mean threshold for PPC loss
global_proto_per_class=5    # Global (CLS) prototypes per class
ppc_cov_coe=0.1             # Weight for PPC_sigma loss
ppc_mean_coe=0.5            # Weight for PPC_mu loss

# DeiT-Small uses the last transformer block (index 11) for token selection
reserve_layer_idx=11

ft=protopformer

# ---------- Output ----------
output_dir=output_cosine/$data_set/$model/${seed}-adamw-${weight_decay}-${epochs}-${ft}

# ---------- Auto-resume ----------
# If no explicit checkpoint given, check if epoch-best.pth already exists
# and resume from it automatically.
if [ -z "$resume_ckpt" ]; then
    default_ckpt="$output_dir/checkpoints/epoch-best.pth"
    if [ -f "$default_ckpt" ]; then
        resume_ckpt="$default_ckpt"
    fi
fi

echo "=============================================="
echo " ProtoPFormer Training — Stanford Dogs"
echo "=============================================="
echo "  Backbone     : $model  (dim=$dim)"
echo "  GPUs         : $num_gpus"
echo "  Batch size   : $batch_size"
echo "  Epochs       : $epochs"
echo "  Prototypes   : $prototype_num local + $(( 120 * global_proto_per_class )) global"
echo "  Data path    : $data_path"
echo "  Output dir   : $output_dir"
if [ -n "$resume_ckpt" ]; then
echo "  Resuming from: $resume_ckpt"
else
echo "  Resume       : none (fresh start)"
fi
echo "=============================================="
echo ""

mkdir -p "$output_dir/checkpoints"

# Shared training arguments (POSIX sh-compatible, no arrays)
TRAIN_ARGS="
    --base_architecture=$model
    --data_set=$data_set
    --data_path=$data_path
    --input_size=$input_size
    --output_dir=$output_dir
    --model=$model
    --batch_size=$batch_size
    --seed=$seed
    --opt=$opt
    --sched=$sched
    --warmup-epochs=$warmup_epochs
    --warmup-lr=$warmup_lr
    --decay-epochs=$decay_epochs
    --decay-rate=$decay_rate
    --weight_decay=$weight_decay
    --epochs=$epochs
    --finetune=$ft
    --features_lr=$features_lr
    --add_on_layers_lr=$add_on_layers_lr
    --prototype_vectors_lr=$prototype_vectors_lr
    --prototype_shape $prototype_num $dim 1 1
    --reserve_layers $reserve_layer_idx
    --reserve_token_nums $last_reserve_num
    --use_global=$use_global
    --use_ppc_loss=$use_ppc_loss
    --ppc_cov_thresh=$ppc_cov_thresh
    --ppc_mean_thresh=$ppc_mean_thresh
    --global_coe=$global_coe
    --global_proto_per_class=$global_proto_per_class
    --ppc_cov_coe=$ppc_cov_coe
    --ppc_mean_coe=$ppc_mean_coe
    --num_workers=$num_workers
    --no-pin-mem
"

# Append resume flag if a checkpoint exists
if [ -n "$resume_ckpt" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume $resume_ckpt"
fi

if [ "$num_gpus" -eq 1 ]; then
    # Single GPU: plain python — no distributed launcher.
    # On SLURM nodes, SLURM_PROCID is set and tricks init_distributed_mode
    # into trying to init NCCL with env:// which needs MASTER_ADDR.
    # Unset it so the code takes the clean non-distributed path.
    unset SLURM_PROCID
    unset RANK
    unset WORLD_SIZE
    unset LOCAL_RANK
    echo "==> Single-GPU mode: running plain python (no distributed)"
    $CONDA_PYTHON main.py $TRAIN_ARGS
else
    # Multi-GPU: dynamic free port to avoid EADDRINUSE collisions.
    use_port=$($CONDA_PYTHON -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    echo "==> Multi-GPU mode: $num_gpus GPUs, port=$use_port"
    $CONDA_PYTHON -m torch.distributed.launch \
        --nproc_per_node=$num_gpus \
        --master_port=$use_port \
        --use_env \
        main.py $TRAIN_ARGS
fi

echo ""
echo "==> Training complete!"
echo "==> Best checkpoint: $output_dir/checkpoints/epoch-best.pth"
