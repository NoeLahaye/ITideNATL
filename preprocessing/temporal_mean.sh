#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
##SBATCH --ntasks-per-node=2
##SBATCH --threads-per-core=1
##SBATCH --mem-per-cpu=5G
#SBATCH --constraint=BDW28
#SBATCH -J TAVE
#SBATCH -e tave.e%j
#SBATCH -o tave.o%j
#SBATCH --time=04:00:00
#SBATCH --exclusive

ulimit -s unlimited

# activate conda environment, see .bashrc
#conda_activate # not working .. weird add to .profile?
source /scratch/cnt0024/ige2071/aponte/conda/occigen/bin/activate

# path to data:
# /store/CT1/hmg2840/lbrodeau/eNATL60/
#   eNATL60-BLB002 experiment (WITHOUT explicit tidal motion)
#   eNATL60-BLBT02 experiment (WITH explicit tidal motion)

root_data_dir='/store/CT1/hmg2840/lbrodeau/eNATL60/'
run='eNATL60-BLB002-S'
# what is `eNATL60-BLB002X-S` `eNATL60-BLB002X-R`?

out_dir='/scratch/cnt0024/ige2071/aponte/tmean/'

# start timer
start_time="$(date -u +%s)"

# generate task.conf
which python
echo $ntasks
python generate_tasks.py $root_data_dir$run $out_dir temporal_mean.py gridT

# run processing
tasks=$( cat task.conf | wc -l )
#srun -n $tasks --mpi=pmi2 -m cyclic  -K1 -o log_%j-%2t.out -e log_%j-%2t.err --multi-prog ./task.conf
#srun --ntasks $tasks -m cyclic  -K1 -o log_%j-%2t.out -e log_%j-%2t.err --multi-prog ./task.conf
srun --cpus-per-task 14 -m cyclic  -K1 -o log_%j-%2t.out -e log_%j-%2t.err --multi-prog ./task.conf
# useful link: https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/


# end timer
end_time="$(date -u +%s)"
seconds=$(($end_time-$start_time))
minutes=$(($seconds / 60))
echo "Total of $minutes minutes elapsed for job"


