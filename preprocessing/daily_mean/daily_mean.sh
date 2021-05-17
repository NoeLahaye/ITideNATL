#!/bin/bash
#SBATCH --nodes=60
#SBATCH --ntasks=120
#SBATCH --constraint=BDW28
#SBATCH -J TAVE
#SBATCH -e tave.e%j
#SBATCH -o tave.o%j
#SBATCH --time=04:00:00
#SBATCH --exclusive

# to launch: sbatch daily_mean.sh

# base processing: --nodes=60 --ntasks=120
# if number of files to process is odd, just delete one processed file

# nodes=60 (120 days): peaks at 28Go/s write and 20Go/s read

ulimit -s unlimited

# activate conda environment, see .bashrc
#conda_activate # not working .. weird add to .profile?
source /scratch/cnt0024/ige2071/aponte/conda/occigen/bin/activate

# start timer
date
start_time="$(date -u +%s)"

# log useful information
which python # check we are using correct python environment
echo "Number of tasks = $SLURM_NTASKS"

# generate task.conf
python generate_tasks.py  $SLURM_NTASKS

# run processing
srun --cpus-per-task 14 -m cyclic  -K1 -o log_%j-%2t.out -e log_%j-%2t.err --multi-prog ./task.conf
# useful link: https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/

# end timer
date
end_time="$(date -u +%s)"
seconds=$(($end_time-$start_time))
minutes=$(($seconds / 60))
echo "Total of $minutes minutes elapsed for job"

