#!/bin/bash
##### THIS SCRIPT IS A TEMPLATE, TO BE "PREPROCESSED" BY PYTHON SCRIPT LAUNCH_TASKS.PY
#SBATCH -J proj-VAR		        ### job name
#SBATCH --nodes=NBODES
#SBATCH --ntasks-per-node=TASKPNODE
#SBATCH --cpus-per-task=CPUPTASK
#SBTACH --threads-per-core=1
#SBATCH --mem=118000        ### using nodes with 128Go
#SBATCH --constraint=HSW24
#SBATCH --time=TIME
#SBATCH -e outjob_proj_VAR.e%j
#SBATCH -o outjob_proj_VAR.o%j
#SBATCH --exclusive
#SBATCH --mail-user=noe.lahaye@inria.fr # Receive e-mail from slurm
#SBATCH --mail-type=END # Type of e-mail from slurm; other options are: Error, Info.

# to launch: sbatch "name of this script"
set -e

# activate conda environment
#module load /opt/software/alfred/spack-dev/modules/tools/linux-rhel7-x86_64/miniconda3/4.7.12.1-gcc-4.8.5
eval "$(conda shell.bash hook)"
conda activate /scratch/cnt0024/ige2071/nlahaye/conda/conda38
nmpi=$SLURM_NTASKS 

echo "now doing it" `date` "JOB ID:" $SLURM_JOBID

i_day=I_DAY 
if [ VAR == "p" ]; then
    var=""
    prog_name=proj_pres_ty-loop.py
else
    var=VAR
    prog_name=proj_uv_ty-loop.py
prog_work=${prog_name%".py"}.$SLURM_JOBID.py

cp $prog_name $prog_work
echo "running" $prog_work "for VAR, using" $SLURM_NNODES" nodes, "$nmpi" tasks, "$SLURM_CPUS_PER_TASK" cpu/task"

srun -n $nmpi python $prog_work $var $i_day
echo "finished" `date`
