# Usage:
# python generate_tasks.py path_in path_out pyscript

# python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH pyscript file_in

# degug
#python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH temporal_mean.py gridT


import os, sys
from glob import glob

root_data_dir='/store/CT1/hmg2840/lbrodeau/eNATL60/'
# path to data:
#   eNATL60-BLB002 experiment (WITHOUT explicit tidal motion)
#   eNATL60-BLBT02 experiment (WITH explicit tidal motion)


def get_files(processing, **kwargs):

    if processing=='raw':
        run=kwargs["run"]
        file_type=kwargs["file_type"]
        return _get_raw_files(run, file_type)


def _get_raw_files(run, file_type):
    """ Return raw netcdf files
    """
    path_in = os.path.join(root_data_dir, run)
    run_dirs = [r for r in sorted(glob(os.path.join(path_in,"00*")))
            if os.path.isdir(r)
            ]

    files = []
    for r in run_dirs:
        files = files + sorted(glob(os.path.join(r,"*_"+file_type+"_*.nc")))

    return files



if __name__ == '__main__':

    # select run
    run='eNATL60-BLB002-S' # what is `eNATL60-BLB002X-S` `eNATL60-BLB002X-R`?
    file_type="gridT"
    files = get_files("raw", run=run, file_type=file_type)

    # output directory
    path_out='/scratch/cnt0024/ige2071/aponte/tmean/'

    # python script
    pyscript = "daily_mean.py"
    extra_args = None

    # get number of tasks
    ntasks = int(sys.argv[1])

    print("{} files available for processing".format(len(files)))
    print("{} tasks / files processed for now".format(ntasks))

    task_file = open("task.conf", "w")
    for i, f in enumerate(files):
        task = "{}-{} python {} {} {} {} ".format(i, i, pyscript, f, file_type, path_out)
        if extra_args:
            task = task + " ".join(extra_args)
        # could test if diagnostic has been done here, done in actual job for now
        if i<ntasks:
            task_file.write(task+"\n")

