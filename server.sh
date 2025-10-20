#!/bin/bash
#SBATCH --job-name=llm_server
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH -G 2
#SBATCH -C "A100-80GB|H100|H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH -N 1
echo "launching LLM Server"

hostname

module load cuda
module load uv

# Make sure CUDA can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1

export SERVER_HOSTNAME=$(hostname)

HOSTNAME_FILE=$(pwd)"/hostname.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME"

source .venv/bin/activate

python -m uvicorn server:app --host $SERVER_HOSTNAME --port 8137 --workers 1

deactivate
