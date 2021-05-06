# sbatch average_daily_means.sh
#
# otherwise for debug, quasi interactif:
# salloc --constraint=BDW28 -N 1 -n 1 -t 10:00
# srun python final_mean.py
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

output_dir = ut.work_data_dir+"mean/"
variable = "gridT"

suffix = "global_mean_{}".format(variable)

depth_custom_chunk=20 # 10 goes through

local_cluster=True
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
                                     "30d_mean_{}_*".format(variable)
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

    ds = xr.concat([open_zarr(z, date) for date, z in zarrs["zarr"].items()],
                    dim="time",
                  )
    ut.print_graph(ds[vkey], "concat", graph)
    print(ds)
    print("Dataset size = {:.1f} GB".format(ds.nbytes/1e9))

    # compute weights to account for inequal batch lengths
    w = (ds["time_end"]-ds["time_start"])/pd.Timedelta("1D") + 1
    w = w/w.mean()
    print(w.values)
    ds[vkey] = ds[vkey] * w

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
    ut.print_graph(ds_processed[vkey], "processed", graph)

    with performance_report(filename="dask-report.html"):
        zarr_archive = suffix+".zarr"
        ds_processed.to_zarr(os.path.join(output_dir, zarr_archive), mode="w")

    # create empty file to indicate processing was completed
    log_file = ut.get_log_file(output_dir, suffix)
    with open(log_file, "w+") as f:
        pass

    print("{} global mean stored".format(variable))

    if not local_cluster:
        cluster.close()

    print("Congrats, processing is over !")
