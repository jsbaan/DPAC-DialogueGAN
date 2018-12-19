#!/bin/sh
#SBATCH -N 1
#SBATCH -t 18:00:00
#SBATCH -p gpu_shared

module load Python/3.6.3-foss-2017b
module load cuDNN
module load OpenMPI/2.1.1-GCC-6.4.0-2.28 NCCL
export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian$

logdir="$TMPDIR"/$(date +"%Y-%m-%d_%H:%M:%S")/
mkdir $logdir
touch $logdir/log

pip install --user word2vec
pip install --user matplotlib
python -c "exec(\"import nltk\nnltk.download('punkt')\")"

cd source
python main.py | tee "$logdir"/log

cp -r $logdir ~
