import os

# True for the main process when running single-GPU or multi-GPU via torchrun/SLURM.
is_rank_zero = int(os.environ.get("LOCAL_RANK", 0)) == 0
