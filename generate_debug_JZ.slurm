#!/bin/bash
#
# Debug run: generate one image with GAN_into_5Gc/generate.py on Jean Zay
#

#SBATCH -A lfb@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --job-name=sg2_generate_dbg
#SBATCH --output=GENE-%j.out

set -eo pipefail

# Some module/conda hook scripts reference unset variables.
# Avoid failing the job because of `set -u` during `module load`.
set +u
module purge
# Reference environment: PyTorch 2.0 (Jean Zay module below).
module load pytorch-gpu/py3/2.0.0
set -u
export PYTHONUNBUFFERED=1

# Paths
CODEDIR="$WORK/ENV_JEAN_ZAY/GAN_into_5Gc_final/stylegan2-ada-pytorch"
NETWORK="$CODEDIR/vanilla_weights/CelebA_128x128.pkl"
OUTDIR="$CODEDIR/out"

# Make StyleGAN2-ADA code (legacy.py, dnnlib/, ...) importable by CODEDIR/generate.py
export PYTHONPATH="$CODEDIR:${PYTHONPATH:-}"

mkdir -p "$OUTDIR"

cd "$CODEDIR"

echo "Network : $NETWORK"
echo "Outdir  : $OUTDIR"
echo "Running : python generate.py --network ... --outdir ... --seeds 88 --num-images 1"

python generate.py \
  --network "$NETWORK" \
  --outdir "$OUTDIR" \
  --seeds 88 \
  --num-images 1
