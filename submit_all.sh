#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks=5
#SBATCH --gpus-per-node 0
#SBATCH --cpus-per-task 12
#SBATCH --job-name=eval_arena
#SBATCH --output=LOGS/eval_arena.%j.out
#SBATCH --error=LOGS/eval_arena.%j.err

# set -e

NAME=${1:-"main"}
OUTPATH=OUTPUT
echo "Argument 1: $NAME"

srun --ntasks=1 --cpus-per-task=8 --exclusive \
    python -u run_arena.py data="data/vllm_evals/lowk_temp1.jsonl" \
    out_dir=${OUTPATH}/${NAME}/lowk_temp1 \
    max_diff=0.2 recompute=True &

srun --ntasks=1 --cpus-per-task=8 --exclusive \
    python -u run_arena.py data="data/vllm_evals/lowk_temp0.7.jsonl" \
    out_dir=${OUTPATH}//${NAME}/lowk_temp0.7 \
    max_diff=0.2 recompute=True &

srun --ntasks=1 --cpus-per-task=8 --exclusive \
    python -u run_arena.py data="data/vllm_evals/highk_temp0.7.jsonl" \
    out_dir=${OUTPATH}/${NAME}/highk_temp0.7 \
    max_diff=0.2 recompute=True &

srun --ntasks=1 --cpus-per-task=8 --exclusive \
    python -u run_arena.py data="data/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/main \
    max_diff=0.2 recompute=True include_var_components=False &

srun --ntasks=1 --cpus-per-task=8 --exclusive \
    python -u run_arena.py data="data/train_curve/*.jsonl" \
    out_dir=${OUTPATH}/${NAME}/train_curve \
    max_diff=0.2 recompute=True &

wait