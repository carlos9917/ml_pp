#!/usr/bin/env bash
#SBATCH --error=glatpp.%j.err
#SBATCH --output=glatpp.%j.out
#SBATCH --job-name=glat_pp
#SBATCH --qos=nf
#SBATCH --mem-per-cpu=64000

# Some examples
module load R/4.2.2
Rscript kirkwood_method_r01.R
