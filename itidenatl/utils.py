
import os, sys, shutil
from glob import glob
from time import sleep

import numpy as np
import pandas as pd
import xarray as xr

import itidenatl.utils as ut

# ---------------------------- params, misc ------------------------------------

vmapping = dict(gridT="votemper",
                gridS="vosaline",
                gridU="vozocrtx",
                gridV="vomecrty",
                gridT2D="sossheig", # ignores all other variables for now
                )

# ---------------------------- paths -------------------------------------------

raw_data_dir = "/work/CT1/hmg2840/lbrodeau/eNATL60/"
work_data_dir = "/work/CT1/ige2071/SHARED/"
scratch_dir="/work/CT1/ige2071/SHARED/scratch/"


# ---------------------------- raw netcdf  -------------------------------------

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
    path_in = os.path.join(raw_data_dir, run)
    run_dirs = [r for r in sorted(glob(os.path.join(path_in,"0*")))
            if os.path.isdir(r)
            ]
    files = []
    for r in run_dirs:
        files = files + sorted(glob(os.path.join(r,"*_"+variable+"_*.nc")))

    return files

def get_raw_files_with_timeline(run, variable):
    """ Build a pandas series with filenames indexed by date
    """
    files = _get_raw_files(run, variable)

    time = [f.split("/")[-1].split("-")[-1].replace(".nc","")
            for f in files]
    timeline = pd.to_datetime(time)
    files = pd.Series(files, index=timeline, name="files").sort_index()
    return files


def get_zarr_with_timeline(output_dir, name):
    """ Build a pandas series with zarr paths indexed by date
    The series is deduced from log files and NOT zarr archives
    It is assumed the date is last in log filenames
    e.g. daily_mean_gridT_20100807

    Parameters
    ----------
    output_dir: str
        Path used for outputs, logs should be in output_dir+"logs/"
    name: str
        Name of the diagnostic
    """
    files = sorted(glob(os.path.join(output_dir,
                                     "logs",
                                     name+"_*"
                                     )
                        )
                   )
    #print(files)
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

def get_log_file(output_dir, name):
    """ return log file path

    Parameters
    ----------
    output_dir: str
        Path used for outputs, logs should be in output_dir+"logs/"
    name: str
        Name of the diagnostic
    """
    log_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_dir, name)
    return log_file

def is_log(*args):
    """ tells whether log file exists

    Parameters
    ----------
    output_dir: str
        Path used for outputs, logs should be in output_dir+"logs/"
    name: str
        Name of the diagnostic
    """
    log_file = get_log_file(*args)
    return os.path.isfile(log_file)

# ---------------------------- dask related ------------------------------------

def spin_up_cluster(type=None, **kwargs):
    """ Spin up a dask cluster ... or not
    Waits for workers to be up for distributed ones

    Paramaters
    ----------
    type: None, str
        Type of cluster: None=no cluster, "local", "distributed"

    """

    if type is None:
        return
    elif type=="local":
        from dask.distributed import Client, LocalCluster
        dkwargs = dict(n_workers=14, threads_per_worker=1)
        dkwargs.update(**kwargs)
        cluster = LocalCluster(**dkwargs) # these may not be hardcoded
        client = Client(cluster)
    elif type=="distributed":
        from dask_jobqueue import SLURMCluster
        from dask.distributed import Client
        assert "processes" in kwargs, "you need to specify a number of processes"
        processes = kwargs["processes"]
        assert "jobs" in kwargs, "you need to specify a number of dask-queue jobs"
        jobs = kwargs["jobs"]
        dkwargs = dict(cores=28,
                       name='pangeo',
                       walltime='03:00:00',
                       job_extra=['--constraint=HSW24',
                                  '--exclusive',
                                  '--nodes=1'],
                       memory="118GB",
                       interface='ib0',
                       )
        dkwargs.update(**kwargs)
        dkwargs = _removekey(dkwargs, "jobs")
        cluster = SLURMCluster(**dkwargs)
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        flag = True
        while flag:
            wk = client.scheduler_info()["workers"]
            print("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk)>=processes*jobs*0.8:
                flag = False
                print("Cluster is up, proceeding with computations")

    return cluster, client

def print_graph(da, name, flag):
    """ print dask graph of a DataArray as png
    see: https://docs.dask.org/en/latest/diagnostics-distributed.html

    Parameters
    ----------
    da: dask.DataArray
        Array of which we want the graph
    name: str
        Name used for figure name
    flag: boolean
        turn actual graph printing on/off
    """
    if not flag:
        return
    da.data.visualize(filename='graph_{}.png'.format(name),
                      optimize_graph=True,
                      color="order",
                      cmap="autumn",
                      node_attr={"penwidth": "4"},
                      )

def custom_distribute(ds, op, tmp_dir, suffix=None, root=True, **kwargs):
    """ Distribute an embarrasingly parallel calculation manually and store chunks to disk

    Parameters
    ----------
    ds: xr.Dataset
        Input data
    op: func
        Process the data and return a dataset
    tmp_dir: str
        temporary output directory, beware: it will be cleaned up
    suffix: str, optional
        suffix employed for temporary files, default is "tmp"
    **kwargs:
        dimensions with chunk size, e.g. (..., face=1) processes 1 face a time
    """

    if suffix is None:
        suffix="tmp"

    if root:
        cleanup_dir(tmp_dir)

    d = list(kwargs.keys())[0]
    c = kwargs[d]

    new_kwargs = _removekey(kwargs, d)

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

def _removekey(d, key):
    r = dict(d)
    del r[key]
    return r
