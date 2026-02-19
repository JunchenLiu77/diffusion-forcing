# single node training script
uv run python -m main +name=ar_baseline algorithm=df_video dataset=video_minecraft

# submit a slurm job, auto-resubmit every 4 hours
sh scripts/submit_slurm.sh 8 ar_baseline

# test the training script on 2 GPUs using interactive node
uv run torchrun --nproc_per_node=2 --standalone -m main +name=ar_baseline_test dataset=video_minecraft