#!/bin/bash
#SBATCH --nodes=18
##SBATCH --ntasks=487
#SBATCH --ntasks-per-node=28
#SBATCH --threads-per-core=1
#SBATCH --constraint=BDW28
#SBATCH -J NC4
#SBATCH -e znc4.e%j
#SBATCH -o znc4.o%j
#SBATCH --time=1:00:00
#SBATCH --exclusive

ulimit -s unlimited

# activate conda environment, see .bashrc
conda_activate

# path to data:
# /store/CT1/hmg2840/lbrodeau/eNATL60/
#   eNATL60-BLB002 experiment (WITHOUT explicit tidal motion)
#   eNATL60-BLBT02 experiment (WITH explicit tidal motion)

root_data_dir='/store/CT1/hmg2840/lbrodeau/eNATL60/'
run='eNATL60-BLB002-S'
# what is `eNATL60-BLB002X-S` `eNATL60-BLB002X-R`?

out_dir='/scratch/cnt0024/ige2071/aponte/tmean/'

python generate_tasks.py $root_data_dir$run $out_dir temporal_mean.py gridT

cd /scratch/cnt0024/hmg2840/molines/CHALLENGE_CINES/SSH
# Compute daily means
mkdir -p DAILY
rm -f task.conf

n=0
for f in *.nc ; do
  g=$( echo $f | sed -e 's/\.1h/\.1d/')
  g=${g%.nc}
  echo $n"-"$n "cdfmoy -l $f -nc4 -o DAILY/$g" >> task.conf
  n=$(( n + 1 ))
done

srun --mpi=pmi2 -m cyclic  -K1 --multi-prog ./task.conf

cd DAILY
rm -f *SSH2.nc
# compute monthly mean
mkdir -p MONTHLY
rm -f task.conf
n=0
for y in {2009..2010} ; do
   for m in {01..12} ; do
     ls *y${y}m${m}*nc > /dev/null 2>&1
     if [ $? = 0 ] ; then
       lst=''
       for f in *y${y}m${m}*nc ; do
         lst="$lst $f"
       done
       echo  $n"-"$n "cdfmoy -l $lst -nc4 -o MONTHLY/eNATL60-BLBT02X_y${y}m${m}.1m_SSH " >> task.conf
       n=$(( n + 1 ))
     fi
   done
done

tasks=$( cat task.conf | wc -l )

srun -n $tasks --mpi=pmi2 -m cyclic  -K1 --multi-prog ./task.conf

cd MONTHLY
rm -f task.conf
n=0
rm -f *SSH2.nc
# computed weighed average of the montlhy mean
mkdir -p ALL_MEAN

cdfmoy_weighted -l *SSH.nc -nc4 -o  ALL_MEAN/eNATL60-BLBT02X_MEAN_SSH.nc
