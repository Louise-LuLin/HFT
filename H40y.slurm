#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=16
#SBATCH --mem=12000MB
#SBATCH --ntasks-per-node=10
#SBATCH --partition=parallel
cd /home/ll5fy/TUIR/HFT
source="YelpNew"
crossV=5
for k in 40 45 50
do
    /home/ll5fy/TUIR/HFT/train -dir /home/ll5fy/TUIR/dataset -source $source -crossV $crossV -K $k > /home/ll5fy/TUIR/HFT/output/"$source"_"$model"_"$k".output
done
