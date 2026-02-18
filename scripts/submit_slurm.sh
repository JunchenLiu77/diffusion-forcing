#!/bin/bash

NUM_GPUS=$1
EXP_NAME=$2
# tuple the rest of the arguments
ARGS=${@:4}

if [ -z "$NUM_GPUS" ] || [ -z "$EXP_NAME" ]; then
    echo "Usage: bash scripts/submit_slurm.sh <NUM_GPUS> <EXP_NAME> [EXTRA_ARGS...]"
    echo "  NUM_GPUS:   Number of GPUs per node"
    echo "  EXP_NAME:   Experiment name (used for logging and as +name= argument)"
    echo "  EXTRA_ARGS: Additional arguments passed to the training command"
    echo ""
    echo "Example: bash scripts/submit_slurm.sh 8 ar_baseline"
    echo "Example: bash scripts/submit_slurm.sh 8 df_pyramid '' algorithm.scheduling_matrix=pyramid"
    exit 1
fi

if [ -z "$ARGS" ]; then
    ARGS=''
fi

CONTAINER_IMAGE=/lustre/fsw/portfolios/av/projects/av_alpamayo_cosmos/users/qiwu/containers/qiwu-fast-infer-v6.sqsh
WORKDIR=/lustre/fsw/portfolios/av/projects/av_alpamayo_cosmos/users/junchenl/diffusion-forcing
SRUN_TIMEOUT=13200  # 3 hours and 40 minutes in seconds

PRE_COMMAND=(
    # "export HF_TOKEN=<your_hf_token>"
    # "export WANDB_API_KEY=<your_wandb_api_key>"
)
COMMAND=(
    "uv run torchrun --nproc_per_node=${NUM_GPUS} --standalone -m main +name=${EXP_NAME} dataset=video_minecraft $ARGS"
)

# Build pre-command string (may be empty)
PRE_COMMAND_STRING=""
if [ ${#PRE_COMMAND[@]} -gt 0 ]; then
    PRE_COMMAND_STRING="${PRE_COMMAND[*]} && "
fi

# The base training command (without resume; resume is injected at runtime).
BASE_COMMAND="${COMMAND[*]}"

echo "Base command: "
echo "--------------------------------"
echo "${PRE_COMMAND_STRING}${BASE_COMMAND}"
echo "--------------------------------"

# Ensure logs directory exists
mkdir -p ${WORKDIR}/outputs/${EXP_NAME}

# Create a temporary job script.
# NOTE: resume arg is resolved at *job start* time (not submission time) so that
# scontrol requeue automatically picks up the wandb run ID saved by the first run.
cat << EOF > ${WORKDIR}/outputs/${EXP_NAME}/run.sh
#!/bin/bash
#SBATCH --account=av_alpamayo_cosmos
#SBATCH --gpus-per-node=$NUM_GPUS
#SBATCH --partition=pool0_av
#SBATCH --time=4:00:00
#SBATCH --output=${WORKDIR}/outputs/${EXP_NAME}/slurm.out
#SBATCH --error=${WORKDIR}/outputs/${EXP_NAME}/slurm.err

# Inject resume arg if a wandb run ID was saved by a previous run.
WANDB_ID_FILE="${WORKDIR}/outputs/${EXP_NAME}/wandb_id"
RESUME_ARG=""
if [ -f "\$WANDB_ID_FILE" ]; then
    WANDB_RUN_ID=\$(cat "\$WANDB_ID_FILE")
    RESUME_ARG="resume=\${WANDB_RUN_ID}"
    echo "Resuming wandb run: \${WANDB_RUN_ID}"
fi

FULL_COMMAND="${PRE_COMMAND_STRING}${BASE_COMMAND} \$RESUME_ARG && echo 'job done'"

timeout $SRUN_TIMEOUT srun \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$HOME:/root,/lustre:/lustre \
--container-workdir=$WORKDIR \
bash -c "\$FULL_COMMAND"

# Check if timeout occurred
if [ \$? -eq 124 ]; then
    echo "Command timed out after $SRUN_TIMEOUT seconds, Requeueing job \$SLURM_JOB_ID"
    scontrol requeue \$SLURM_JOB_ID
fi
EOF

# Submit the job
sbatch ${WORKDIR}/outputs/${EXP_NAME}/run.sh
