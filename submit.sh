#!/bin/bash

#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p sugon                # 分区参数                                                       
#SBATCH -w sugon-gpu-2          # 分区参数

nvidia-smi topo --matrix
#nvprof --print-gpu-trace ./hyper
#nvprof --print-api-trace ./hyper
#srun nvprof --device-buffer-size 16 ./hyper
srun ./Hyper
