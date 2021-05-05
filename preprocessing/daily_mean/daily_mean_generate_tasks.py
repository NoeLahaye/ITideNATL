# Usage:
# python generate_tasks.py path_in output_dir pyscript

# python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH pyscript file_in

# degug
#python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH temporal_mean.py gridT


import os, sys
from glob import glob

import numpy as np
import pandas as pd

#root_data_dir="/store/CT1/hmg2840/lbrodeau/eNATL60/"
root_data_dir="/work/CT1/hmg2840/lbrodeau/eNATL60/"

# output directory
#output_dir="/scratch/cnt0024/ige2071/aponte/tmean/"
output_dir="/work/CT1/ige2071/SHARED/mean/"

variable="gridT"

# best if batch_size matches task number in daily_mean.sh (ntasks parameter)
#batch_size = 30

# which batch we consider
#n_batch = 0

#tstart="2009-06-30"
#tend="2009-06-30"

# global start end:
#2009-06-30 00:00:00
#2010-10-29 00:00:00

# path to data:
#   eNATL60-BLB002 experiment (WITHOUT explicit tidal motion)
#   eNATL60-BLBT02 experiment (WITH explicit tidal motion)

def _get_raw_files(run, variable):
    """ Return raw netcdf files

    Parameters
    ----------
    run: str, list
        string corresponding to the run or list of strings
    variable:
        variable to consider, e.g. ("gridT", "gridS", etc)
    """

    # multiple runs may be passed at once
    if isinstance(run, list):
        files = []
        for r in run:
            files = files + _get_raw_files(r, variable)
        return files

    # single run
    path_in = os.path.join(root_data_dir, run)
    run_dirs = [r for r in sorted(glob(os.path.join(path_in,"0*")))
            if os.path.isdir(r)
            ]
    files = []
    for r in run_dirs:
        files = files + sorted(glob(os.path.join(r,"*_"+variable+"_*.nc")))

    return files

def get_raw_files_with_timeline(run):
    """ Build a pandas series with filenames indexed by date
    """
    files = _get_raw_files(run, variable)

    time = [f.split("/")[-1].split("-")[-1].replace(".nc","")
            for f in files]
    timeline = pd.to_datetime(time)
    files = pd.Series(files, index=timeline, name="files").sort_index()

    return files

def get_file_processed(files):
    """ Add boolean flag is file has already been processed

    debug: touch /work/CT1/ige2071/SHARED/mean/logs/daily_mean_gridT_20090630
    """
    log_path = os.path.join(output_dir, "logs")
    def is_processed(date):
        log_file = os.path.join(log_path,
                                "daily_mean_"
                                +variable+"_"
                                +date.strftime("%Y%m%d"),
                                )
        return os.path.isfile(log_file)
    files = files.to_frame()
    files["processed"] = (pd
                          .Series(files.index, index=files.index)
                          .map(is_processed)
                          )
    return files

if __name__ == "__main__":

    # select run
    #run="eNATL60-BLB002" # no tide
    #run="eNATL60-BLBT02-S" # with tide
    #run="eNATL60-BLBT02X-S" # with tide suite
    run = ["eNATL60-BLBT02-S", "eNATL60-BLBT02X-S"]
    # what is `eNATL60-BLB002X-S` `eNATL60-BLB002X-R`?

    files = get_raw_files_with_timeline(run)
    print("Global start: ", files.index[0])
    print("Global end: ",files.index[-1])
    print("{} files available for processing".format(files.index.size))

    # skips files already processed
    files = get_file_processed(files)
    files = files.loc[~files["processed"], "files"]

    # python script that actually performs the computation
    pyscript = "daily_mean.py"
    extra_args = None

    # get number of tasks
    ntasks = int(sys.argv[1])

    if ntasks>files.index.size:
        print("More tasks than files to process")
        print("you need to decrease task number to {}".format(files.index.size))
        sys.exit()

    print("{} tasks/files processed for now".format(ntasks))

    task_file = open("task.conf", "w")
    i=0
    for _, f in files.items():
        task = "{}-{} python {} {} {} {} ".format(i, i, pyscript, f, variable, output_dir)
        if extra_args:
            task = task + " ".join(extra_args)
        if i<ntasks:
            print(task+"\n")
            task_file.write(task+"\n")
        i+=1
