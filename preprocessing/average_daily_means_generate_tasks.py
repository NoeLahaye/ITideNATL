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
batch_size = 30 # days
n_batch = 0
suffix = str(batch)+"d_average"

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

def get_zarr_with_timeline(run):
    """ Build a pandas series with filenames indexed by date
    """
    files = sorted(glob(os.path.join(output_dir,
                                     "daily_mean_{}_*.zarr".format(variable)
                                     )
                        )
                   )

    time = [f.split("/")[-1].split("_")[-1].replace(".zarr","")
            for f in files]
    timeline = pd.to_datetime(time)
    files = pd.Series(files, index=timeline, name="files").sort_index()

    return files

def get_zarr_processed(files):
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
    #run = ["eNATL60-BLBT02-S", "eNATL60-BLBT02X-S"]
    # what is `eNATL60-BLB002X-S` `eNATL60-BLB002X-R`?

    files = get_zarr_with_timeline(run)
    print("Global start: ", files.index[0])
    print("Global end: ",files.index[-1])
    print("{} files available for processing".format(files.index.size))

    # skips files already processed
    #files = get_file_processed(files)
    #files = files.loc[~files["processed"], "files"]

    if n_batch>=0:
        file_batches = [files.iloc[i:i+batch_size]
                        for i in range(0, files.index.size, batch_size)
                        ]
        print(file_batches)
        #print(file_batches[0])
        files = file_batches[n_batch]
    #    print("Batch start: ", files.index[0])
    #    print("Batch end: ", files.index[-1])

    # python script that actually performs the computation
    pyscript = "average_daily_means.py"
    extra_args = None

    # get number of tasks
    ntasks = int(sys.argv[1])

    print("{} tasks/files processed for now".format(ntasks))

    task_file = open("task.conf", "w")
    i=0
    for _, f in files.items():
        task = "{}-{} python {} {} {} {} ".format(i, i, pyscript,
                                                  f.index[0],
                                                  f.index[-1],
                                                  variable,
                                                  output_dir)
        if extra_args:
            task = task + " ".join(extra_args)
        if i<ntasks:
            print(task+"\n")
            #task_file.write(task+"\n")
        i+=1
