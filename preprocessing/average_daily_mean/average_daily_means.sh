#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=118000
#SBATCH --constraint=HSW24
#SBATCH -J TAVE
#SBATCH -e aved.e%j
#SBATCH -o aved.o%j
#SBATCH --time=03:00:00
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

# run processing
python average_daily_means.py

# end timer
end_time="$(date -u +%s)"
seconds=$(($end_time-$start_time))
minutes=$(($seconds / 60))
echo "Total of $minutes minutes elapsed for job"
