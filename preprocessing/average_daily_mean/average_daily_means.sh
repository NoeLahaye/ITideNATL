#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=118000
#SBATCH --constraint=HSW24
#SBATCH -J TAVE
#SBATCH -e aved.e%j
#SBATCH -o aved.o%j
#SBATCH --time=02:00:00
#SBATCH --exclusive

# to launch: sbatch daily_mean.sh

ulimit -s unlimited

# activate conda environment, see .bashrc
#conda_activate # not working .. weird add to .profile?
source /scratch/cnt0024/ige2071/aponte/conda/occigen/bin/activate

# start timer
start_time="$(date -u +%s)"

# log useful information
which python # check we are using correct python environment
#echo "Number of tasks = $SLURM_NTASKS"

# generate task.conf
#srun --mpi=pmi2 -K1 -n $SLURM_NTASKS ./mon_executable param1 param2

#python daily_mean_generate_tasks.py  $SLURM_NTASKS

# run processing
#srun --cpus-per-task 28  -K1 -o log_%j-%2t.out -e log_%j-%2t.err python average_daily_means.py
#srun --cpus-per-task 28  -K1 -o log_%j-%2t.out -e log_%j-%2t.err python average_daily_means.py
python average_daily_means.py
# useful link: https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/

# end timer
end_time="$(date -u +%s)"
seconds=$(($end_time-$start_time))
minutes=$(($seconds / 60))
echo "Total of $minutes minutes elapsed for job"
