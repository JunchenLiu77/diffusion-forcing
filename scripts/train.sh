# single node training script
uv run python -m main +name=ar_baseline algorithm=df_video dataset=video_minecraft

# submit a slurm job, auto-resubmit every 4 hours
sh scripts/submit_slurm.sh 8 ar_baseline

# test the training script on 2 GPUs using interactive node
OMP_NUM_THREADS=$(($(nproc) / 2)) uv run torchrun --nproc_per_node=2 --standalone -m main +name=ar_baseline_test dataset=video_minecraft

# test resuming
OMP_NUM_THREADS=$(($(nproc) / 2)) uv run torchrun --nproc_per_node=2 --standalone -m main +name=ar_baseline_test resume=woxlo4fj dataset=video_minecraft