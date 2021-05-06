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

# input parameters

output_dir="/work/CT1/ige2071/SHARED/mean/"
variable = "gridT"

suffix = "global_mean_{}".format(variable)

depth_custom_chunk=20 # 10 goes through

local_cluster=True
dask_jobs = 16
workers_per_job = 7

scratch_dir="/work/CT1/ige2071/SHARED/scratch/"

# debug flag and graph outputs
debug=False
graph=debug

def print_graph(da, name, index, flag):
    """ store dask graph as png
    see: https://docs.dask.org/en/latest/diagnostics-distributed.html
    """
    if not flag:
        return
    da.data.visualize(filename='graph_average_{}_{}.png'.format(name, index),
                        optimize_graph=True,
                        color="order",
                        cmap="autumn",
                        node_attr={"penwidth": "4"},
                        )

def get_zarr_with_timeline():
    """ Build a pandas series with filenames indexed by date
    """
    files = sorted(glob(os.path.join(output_dir,
                                     "logs",
                                     "30d_average_{}_*".format(variable)
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

def get_log_file():
    """ return log file path
    """
    log_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_dir, suffix)
    return log_file

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

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def cleanup_dir(directory):
    """ Remove all files and subdirectory inside a directory
    https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def custom_distribute(ds, op, tmp_dir=None, suffix=None, root=True, **kwargs):
    """ Distribute an embarrasingly parallel calculation manually and store chunks to disk
    Parameters
    ----------
    ds: xr.Dataset
        Input data
    op: func
        Process the data and return a dataset
    tmp_dir: str, optional
        temporary output directory
    suffix: str
        suffix employed for temporary files
    **kwargs:
        dimensions with chunk size, e.g. (..., face=1) processes 1 face a time
    """

    if suffix is None:
        suffix="tmp"

    if root:
        cleanup_dir(tmp_dir)

    d = list(kwargs.keys())[0]
    c = kwargs[d]

    new_kwargs = removekey(kwargs, d)

    dim = np.arange(ds[d].size)
    chunks = [dim[i*c:(i+1)*c] for i in range((dim.size + c - 1) // c )]

    D = []
    Z = []
    for c, i in zip(chunks, range(len(chunks))):
        _ds = ds.isel(**{d: slice(c[0], c[-1]+1)})
        _suffix = suffix+"_{}".format(i)
        if new_kwargs:
            _out, _Z = custom_distribute(_ds, op, tmp_dir=tmp_dir, suffix=_suffix, root=False, **new_kwargs)
            D.append(_out)
            if root:
                print("{}: {}/{}".format(d,i,len(chunks)))
            Z.append(_Z)
        else:
            # store
            out = op(_ds)
            zarr = os.path.join(tmp_dir, _suffix)
            Z.append(zarr)
            out.to_zarr(zarr, mode="w")
            D.append(xr.open_zarr(zarr))
            #print("End reached: {}".format(_suffix))

    # merge results back and return
    ds = xr.concat(D, d) 

    return ds, Z


if __name__ == '__main__':

    if local_cluster:
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=14, threads_per_worker=1) # these may not be hardcoded
        client = Client(cluster)
    else:
        from dask_jobqueue import SLURMCluster 
        from dask.distributed import Client 
        cluster = SLURMCluster(cores=28,
                               processes=workers_per_job,
                               name='pangeo', 
                               walltime='01:00:00',
                               job_extra=['--constraint=HSW24',
                                          '--exclusive',
                                          '--nodes=1'],
                               memory="118GB",
                               interface='ib0',
                              ) 
        print(cluster.job_script())
        cluster.scale(jobs=dask_jobs)
        client = Client(cluster)
    
        flag = True
        while flag:
            wk = client.scheduler_info()["workers"]
            print("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk)>=workers_per_job*dask_jobs*0.8:
                flag = False
                print("Cluster is up, proceeding with computations")

    print(client)

    zarrs = get_zarr_with_timeline()

    ds = xr.concat([open_zarr(z, date) for date, z in zarrs["zarr"].items()],
                    dim="time",       
                  )
    print_graph(ds["votemper"], "concat", "", graph)
    print(ds)
    print("Dataset size = {:.1f} GB".format(ds.nbytes/1e9))

    # compute weights:
    w = (ds["time_end"]-ds["time_start"])/pd.Timedelta("1D") + 1
    w = w/w.sum()
    print(w.values)
    ds["votemper"] = ds["votemper"] * w
    sys.exit() 

    # temporal average
    #ds_processed = ds.mean("time")
    ds_processed, _ = custom_distribute(ds, 
                                        lambda ds: ds.mean("time"), 
                                        deptht=depth_custom_chunk,
                                        tmp_dir=scratch_dir,
                                        )
    ds_processed = ds_processed.expand_dims("time")
    ds_processed["time_start"] = ("time", ds.time[[0]].values)
    ds_processed["time_end"] = ("time", ds.time[[-1]].values)

    #
    #print(ds_processed)
    print_graph(ds_processed["votemper"], "processed", "", graph)

    with performance_report(filename="dask-report.html"):
        zarr_archive = suffix+".zarr"
        ds_processed.to_zarr(os.path.join(output_dir, zarr_archive), mode="w")

    # create empty file to indicate processing was completed
    log_file = get_log_file()
    with open(log_file, "w+") as f:
        pass

    print("{} global mean stored".format(variable))

    if not local_cluster:
        cluster.close()

    print("Congrats, processing is over !")
