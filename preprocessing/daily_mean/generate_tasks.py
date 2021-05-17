# Usage:
# python generate_tasks.py path_in output_dir pyscript

# python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH pyscript file_in

# degug
#python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH temporal_mean.py gridT

import os, sys
from glob import glob

import numpy as np
import pandas as pd

import itidenatl.utils as ut

# output directory
#output_dir="/scratch/cnt0024/ige2071/aponte/tmean/"
output_dir = ut.work_data_dir+"mean/"

# variable considered
#variable="gridT"
#variable="gridS"
variable="gridT-2D"

# path to data:
#   eNATL60-BLB002 experiment (WITHOUT explicit tidal motion)
#   eNATL60-BLBT02 experiment (WITH explicit tidal motion)
run = ["eNATL60-BLBT02-S", "eNATL60-BLBT02X-S"]

# global start end:
#2009-06-30 00:00:00
#2010-10-29 00:00:00

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

    files = ut.get_raw_files_with_timeline(run, variable)
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
