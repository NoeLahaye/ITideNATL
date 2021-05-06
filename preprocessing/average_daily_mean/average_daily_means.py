# sbatch average_daily_means.sh
#
# otherwise for debug, quasi interactif:
# salloc --constraint=BDW28 -N 1 -n 1 -t 10:00
# srun python average_daily_means.py
# see https://www.cines.fr/calcul/faq-calcul-intensif/

import os, sys, shutil
from glob import glob
from time import sleep

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import performance_report

import itidenatl.utils as ut

# input parameters

variable = "gridT"

output_dir = ut.work_data_dir+"mean/"

# best if batch_size matches task number in daily_mean.sh (ntasks parameter)
batch_size = 30 # days
suffix = "{}d_mean_{}_".format(batch_size, variable)

depth_custom_chunk=20 # 10 goes through

local_cluster=False
dask_jobs = 16
workers_per_job = 7

# debug flag and graph outputs
debug=False
graph=debug

# variable key in datasets
vkey = ut.vmapping[variable]

def get_zarr_with_timeline():
    """ Build a pandas series with filenames indexed by date
    """
    files = sorted(glob(os.path.join(output_dir,
                                     "logs",
                                     "daily_mean_{}_*".format(variable)
                                     )
                        )
                   )
    time = [f.split("/")[-1].split("_")[-1]
            for f in files]
    timeline = pd.to_datetime(time)
    zarrs = (pd.Series(files, index=timeline, name="log")
             .sort_index()
             .to_frame()
             )
    # add zarr file
    zarrs["zarr"] = zarrs["log"].map(lambda l: l.replace("logs/","")+".zarr")
    # add flag if zarr archive exists
    zarrs["flag"] = zarrs["zarr"].map(os.path.isdir)
    return zarrs

def open_zarr(z, date):
    """ load and adjuste dataset
    """
    ds = (xr
          .open_zarr(z)
          .drop_vars("deptht_bounds", errors="ignore")
          .expand_dims({"time": [pd.Timestamp(date)]})
         )
    if debug:
        ds = ds.isel(deptht=slice(0,2))
    return ds

if __name__ == '__main__':

    if local_cluster:
        cluster, client = spin_up_cluster(type="local", n_workers=14)
    else:
        cluster, client = spin_up_cluster(type="distributed",)

    print(client)

    zarrs = get_zarr_with_timeline()

    # generate batch of zarr files
    zarr_batches = [zarrs.iloc[i:min(i+batch_size, zarrs.index.size)]
                    for i in range(0, zarrs.index.size, batch_size)
                    ]
    for batch in zarr_batches:
        print(batch.index.size)

    # loop around batches
    #batch = zarr_batches[0]
    for batch in zarr_batches:

        # name batches according to first day of the batch
        batch_name = batch.index[0].strftime("%Y%m%d")

        if ut.is_log(output_dir, suffix+batch_name):
            print(batch_name+ " processed - skips")
        else:
            print(batch_name+ " not processed")

            # debug:
            if debug:
                batch = batch.iloc[:3, :]

            ds = xr.concat([open_zarr(z, date)
                            for date, z in batch["zarr"].items()
                            ],
                           dim="time",
                          )
            print_graph(ds[vkey], "concat_"+batch_name, graph)
            #print(ds)
            print("Dataset size = {:.1f} GB".format(ds.nbytes/1e9))

            # temporal average
            #ds_processed = ds.mean("time")
            ds_processed, _ = ut.custom_distribute(ds,
                                            lambda ds: ds.mean("time"),
                                            ut.scratch_dir,
                                            deptht=depth_custom_chunk,
                                            )
            ds_processed = ds_processed.expand_dims("time")
            ds_processed["time_start"] = ("time", ds.time[[0]].values)
            ds_processed["time_end"] = ("time", ds.time[[-1]].values)

            #
            #print(ds_processed)
            print_graph(ds_processed[vkey],
                        "processed_"+batch_name,
                        graph,
                        )

            with performance_report(filename="dask-report.html"):
                zarr_archive = suffix+batch_name+".zarr"
                ds_processed.to_zarr(os.path.join(output_dir, zarr_archive), mode="w")

            # create empty file to indicate processing was completed
            log_file = ut.get_log_file(output_dir, suffix+batch_name)
            with open(log_file, "w+") as f:
                pass

            print("{} stored".format(batch_name))

    if not local_cluster:
        cluster.close()

    print("Congrats, processing is over !")
