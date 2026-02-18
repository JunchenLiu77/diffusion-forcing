uv run python -m main +name=ar_baseline algorithm=df_video dataset=video_minecraft

uv run python -m main +name=fs_baseline algorithm=df_video dataset=video_minecraft \
    algorithm.scheduling_matrix=full_sequence \
    algorithm.chunk_size=-1 \
    algorithm.causal=False

uv run python -m main +name=df_pyramid algorithm=df_video dataset=video_minecraft \
    algorithm.scheduling_matrix=pyramid

sh scripts/submit_slurm.sh 8 ar_baseline