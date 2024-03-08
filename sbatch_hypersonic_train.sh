#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J hypersonic_train_diff_rk4
#SBATCH -o hypersonic_train_diff_rk4.out
#SBATCH -e hypersonic_train_diff_rk4.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem=64G
#SBATCH --constraint=a100_80gb


module purge
module load apptainer tensorflow/2.13.0
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif parc_v2_demo_hypersonic.py --train 1.6 1.7 1.8 2.1 2.2 2.3 2.4 2.6 2.7 2.8 2.9 3.1 3.2 3.3 3.4 3.6 3.7 3.8 3.9 4.1 4.2 4.3 4.4 \
                                                                                    --val 1.1 1.2 1.3 1.4 1.5 2.0 2.5 3.0 3.5 4.0 4.5 4.6 4.7 4.8 4.9 5.0 \
                                                                                  --mode differentiator_training --seq_len 2 --nsteps 40\
                                                                                  --solver rk4 \
                                                                                  --save_diff parc2_5_sym_diff_rk4_2.h5 \
                                                                                  --epoch 500 --lr 1e-5
