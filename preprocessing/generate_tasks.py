# Usage:
# python generate_tasks.py path_in path_out pyscript

# python generate_tasks.py $NATL/eNATL60-BLB002-S $SCRATCH pyscript file_in

import os, sys
from glob import glob

path_in = sys.argv[1]
path_out = sys.argv[2]
pyscript = sys.argv[3]
file_in = sys.argv[4]
if len(sys.argv)>5:
  extra_args = sys.argv[5:]
else:
  extra_args = None

run_dirs = [r for r in sorted(glob(os.path.join(path_in,"00*")))
            if os.path.isdir(r)
            ]

files = []
for r in run_dirs:
    files = files + sorted(glob(os.path.join(r,"*"+file_in+"*.nc")))

print("{} files to process".format(len(files)))

print(files[0])

for i, f in enumerate(files):
    task = "{}-{} python {} {} ".format(i, i, pyscript, f)
    if extra_args:
        task = task + " ".join(extra_args)
    # add test if diagnostic done?
    print(task)
