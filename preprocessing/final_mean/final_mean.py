# sbatch average_daily_means.sh
#
# otherwise for debug, quasi interactif:
# salloc --constraint=BDW28 -N 1 -n 1 -t 10:00
# srun python final_mean.py
# see https://www.cines.fr/calcul/faq-calcul-intensif/

import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import performance_report

import itidenatl.utils as ut

# input parameters

output_dir = ut.work_data_dir+"mean/"
#variable = "gridT"
#variable = "gridS"
variable = "gridT-2D"

suffix = "global_mean_{}".format(variable)

depth_custom_chunk=20 # 10 goes through

local_cluster=False
dask_jobs = 16
workers_per_job = 7

# debug flag and graph outputs
debug=False
graph=debug

# variable key in datasets
vkey = ut.vmapping[variable.replace("-","")]

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
        cluster, client = ut.spin_up_cluster(type="local", n_workers=14)
    else:
        cluster, client = ut.spin_up_cluster(type="distributed", 
                                             jobs=dask_jobs, 
                                             processes=workers_per_job,
                                            )

    print(client)

    zarrs = ut.get_zarr_with_timeline(output_dir, "30d_mean_"+variable)
    #print(zarrs)

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
    if "deptht" not in ds.dims:
        ds_processed = ds.mean("time")
    else:
        ds_processed, _ = ut.custom_distribute(ds,
                                lambda ds: ds.mean("time"),
                                ut.scratch_dir,
                                deptht=depth_custom_chunk,
                                )
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
