#BSUB -P csc340
#BSUB -W 0:10
#BSUB -nnodes 16
#BSUB -J fcc_16_thread
#BSUB -o fcc16.%J
#BSUB -e fcc16.%J
#BSUB -q debug

jsrun -n 32 -a 1 -c 16 -r 2 -EOMP_NUM_THREADS=16 -dpacked -brs /ccs/home/dutta/scratch/MFIX_Catalyst/fcc/mfix_exec_link inputs
