#!/bin/bash
#SBATCH -J test
#SBATCH -p CS177h
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=zhujl@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 360:00
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# python /public/home/cs177h/zhujl/project/MLP.py
# python /public/home/cs177h/zhujl/project/verify.py
python /public/home/cs177h/zhujl/project/src/transformer.py