#!/bin/bash
  
#BSUB -J feature_detect
#BSUB -nnodes 1
#BSUB -P csc340
#BSUB -W 00:05
#BSUB -q debug

cd build/

## this works fine
jsrun -n 2 -g 2 statistical_feature_detection/stat_feature_detect
