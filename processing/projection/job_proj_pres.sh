#!/bin/bash
#SBATCH -J projp		        ### job name
#SBATCH --nodes=2
##SBATCH --ntasks=0
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBTACH --threads-per-core=1
#SBATCH --mem=118000        ### using nodes with 128Go
#SBATCH --constraint=HSW24
#SBATCH --time=02:30:00
#SBATCH -e outjob_proj_p.e%j
#SBATCH -o outjob_proj_p.o%j
#SBATCH --exclusive
#SBATCH --mail-user=noe.lahaye@inria.fr # Receive e-mail from slurm
#SBATCH --mail-type=END # Type of e-mail from slurm; other options are: Error, Info.

# to launch: sbatch "name of this script"
set -e

# activate conda environment
#module load /opt/software/alfred/spack-dev/modules/tools/linux-rhel7-x86_64/miniconda3/4.7.12.1-gcc-4.8.5
eval "$(conda shell.bash hook)"
conda activate /scratch/cnt0024/ige2071/nlahaye/conda/conda38
nmpi=$SLURM_NTASKS #$(( $SLURM_NTASKS + 1 )) # this is (dangerous) cheating

#### one method for initializing a dask cluster. 
#   I did not manage to get it work on occigen
#   something like
# srun -n $nmpi dask-mpi --threads= --memory-limit= --no-nanny --scheduler_file=scheduler.json
# This writes a scheduler.json file into your home directory
# You can then connect with the following Python code
# >>> from dask.distributed import Client
# >>> client = Client(scheduler_file='~/scheduler.json')
#scheduler_file="$HOME/working_on/processing/scheduler.json"

echo "now doing it" `date` "JOB ID:" $SLURM_JOBID

i_day=$(seq 3 4)
prog_name=proj_pres_ty-loop.py # proj_pres_ty-loop_local.py #
prog_work=${prog_name%".py"}.$SLURM_JOBID.py

cp $prog_name $prog_work
echo "running" $prog_work", using" $SLURM_NNODES" nodes, "$nmpi" tasks, "$SLURM_CPUS_PER_TASK" cpu/task"

srun -n $nmpi python $prog_work $i_day

echo "finished" `date`

