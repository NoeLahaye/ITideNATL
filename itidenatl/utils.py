
import os, shutil
from glob import glob
from time import sleep
from pathlib import Path

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

def get_list_files(data_path=Path(raw_data_dir), i_days=None):
    """ not sure this work if i_days is not None, it might take the order of files """
    subs = "eNATL60-BLBT02*-S/????????-????????/eNATL60-BLBT02*_1h_*_gridS_*.nc"
    list_files = list(data_path.glob(subs))
    if i_days is not None:
        i_days = list(i_days)
        list_files = [list_files[i] for i in i_days]
    return list_files

def get_dico_files(data_path=Path(raw_data_dir), i_days=None):
    """ not sure this work if i_days is not None, it might take the order of files """
    list_files = get_list_files(data_path=data_path, i_days=i_days)
    dico_files = {k.name.rstrip(".nc")[-8:]:k for k in list_files} # dico day:path
    return dico_files

def get_date_from_iday(i_days=None, data_path=Path(raw_data_dir)):
    """
    return all dates sorted if i_days is None, dates att day # i_days if i_days is in or list of int
    format yyymmdd

    Parameters:
    ___________
    i_days: int or list (optional)

    Returns:
    _______
    str or list of str with dates sorted
    """
    
    list_files = get_list_files(data_path=(Path(raw_data_dir)))
    dates = [k.name.rstrip(".nc")[-8:] for k in list_files] # list of dates (day)
    dates.sort()

    if i_days is not None:
        if isinstance(i_days, int):
            dates = dates[i_days]
        elif isinstance(i_days, list):
            dates = [dates[i] for i in i_days]
    return dates

   
def get_eNATL_path(var=None, its=None, data_path=Path(raw_data_dir)):
    """ get path of eNATL raw data given a variable name and time instants (days) 
    Parameters
    __________
    var: str (optional)
        variable name (NEMO OPA name)
        return parent directories if not provided
    it: int or list of int (optional)
        date (day of simulation). Returns all available date if not provided
    data_path: str or pathlib.Path object (optional)
        parent directory for all simulation data (default: utils.raw_data_dir)
    """

    dates = get_date_from_iday(data_path=data_path)
    dico_files = get_dico_files(data_path=data_path)
    
    ### utilitary function to get file corresponding to one time index and one variable
    map_varname = {v:k for k,v in ut.vmapping.items()}
    if map_varname["sossheig"]=="gridT2D":
        map_varname["sossheig"] = "gridT-2D"
     
    if isinstance(its, list):
        res = []
        for it in its:
            path = dico_files[dates[it]]
            if var is None:
                name = ""
            else:
                name = path.name.replace("gridS", map_varname[var])
            res.append(path.parent/name)
    elif isinstance(its, int):
        path = dico_files[dates[its]]
        if var is None:
            name = ""
        else:
            name = path.name.replace("gridS", map_varname[var])
        res = path.parent/name
    else:
        res = []
        for da in dates:
            path = dico_files[da]
            if var is None:
                name = ""
            else:
                name = path.name.replace("gridS", map_varname[var])
            res.append(dico_files[da].parent/name)
    return res
        

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


# ------------------------- various routines related to routines, options, etc. ------------- #
_name = {"et":"e3t", "ew":"e3w", "ssh":"sossheig", "mask":"tmask",
        "zt":"depth_c", "zw":"depth_l"}

def _parse_name_dict(dico, special=None):
    """ wrapper of _parse_inp_dict to use _name dict as default"""
    return _parse_inp_dict(dico, _name, special)

def _parse_inp_dict(dico, defo, special=None):
    """ parse input dictionary containg names for updating the default one """
    if dico is None:
        return defo.copy()
    else:
        newdic = defo.copy()
        if isinstance(dico, dict):
            newdic.update(dico)
        elif isinstance(dico, str):
            newdic[special] = dico
        else:
            raise ValueError('unable to parse "name" argument')
        return newdic

def _da_or_ds(ds, nam=None):
    """ return dataarray from dataset or dataarray """
    if isinstance(ds, xr.Dataset):
        return ds[nam]
    else:
        return ds
    
# ------------------------ xarray / xorca related ------------------------------- #

from xorca import orca_names

_orca_names_merged = {**orca_names.orca_coords, **orca_names.orca_variables}
# update a la mano
_orca_names_merged["vosigmainsitu"] = _orca_names_merged["votemper"]
_offset = {"c":1., "l":.5, "r":1.5}
# I could parse this from _orca_names_merged
_zdims_in_dataset = {"vosaline":"deptht", "votemper":"deptht", "vosigmainsitu":"deptht",
                     "vozocrtx":"depthu", "vomecrty":"depthv", "vovecrtz":"depthw", 
                     "sossheig":None}

def open_one_var(path, chunks="auto", varname=None, verbose=False, **kwargs):
    """ utilitary function to open datasets for one variable 
    and return dataset with fixed dimension names and minimal coordinates 
    Works for 3D (t,y,x) or 4D (t,z,y,x) avriable. Not check for others """
    ### infer targetted variable name from file names if not provided
    if not varname:
        path_ref = path[0] if isinstance(path, list) else path
        varname = next(v for v in str(path_ref).split("_") if len(v)==8 and v.startswith("vo"))
    else:
        if isinstance(path, list): # retain only variable files
            n_path = [v for v in path if varname in str(v)]
            path = path if len(n_path)==0 else n_path

            
    ### update chunk dict
    if isinstance(chunks, dict):
        chks = chunks.copy()
        if _zdims_in_dataset[varname]:
            chks[_zdims_in_dataset[varname]] = chks.pop("z")
        else:
            chks.pop("z")
        chks["time_counter"] = chks.pop("t")
    else:
        chks = chunks
        
    ### open dataset
    if isinstance(path, list):
        if verbose:
            print("opening", path, "with chunking", chks, "and kwargs", kwargs)
        ds = xr.open_mfdataset(path, chunks=chks, **kwargs)
    else:
        if verbose:
            print("opening", path, "with chunking", chks, "and kwargs", kwargs)
        ds = xr.open_dataset(path, chunks=chks, **kwargs)
        
    ### get rid of coordinates and meta variables
    if "axis_nbounds" in ds.dims:
        ds = ds.drop_dims("axis_nbounds")
    ds = ds.reset_coords(drop=True)
    ### check that we have a single variable with correct name in the end
    if len(ds.data_vars)>1:
        coords = [v for v in ds.data_vars.keys() if v != varname]
        ds = ds.set_coords(coords)
    nam = next(k for k in ds.data_vars.keys())
    assert nam==varname
    
    ### proceed to dimension renaming
    dims = []
    dims_tg = _orca_names_merged[nam]["dims"] # order "t","z","y","x"
    #if "time_counter" in ds.dims:
    #    ds = ds.rename({"time_counter":"t"})
    dims = ["time_counter",] + next(([d] for d in ds.dims if d.startswith("dep")), []) + ["y", "x"]
    ds = ds.rename({d:dims_tg[i] for i,d in enumerate(dims)})#, zdim:dims_tg[1], "y":dims_tg[2], "x":dims_tg[3]})
    if len(dims)==4:
        ds[dims_tg[1]] = np.arange(ds[dims_tg[1]].size, dtype="float32") + _offset[dims_tg[1][-1]]
        ds = ds.assign_coords({di:np.arange(ds[di].size, dtype="float32") + _offset[di[-1]] 
                               for di in dims_tg[2:]})
    else:
        ds = ds.assign_coords({di:np.arange(ds[di].size, dtype="float32") + _offset[di[-1]] 
                               for di in dims_tg[1:]})
    
    return ds

def open_one_coord(path, varname, chunks=None, verbose=False):
    """ get one coordinate avriable from, e.g. mesh mask file
    works only for one single file to read from. Does not update coords.
    returns dataset """

    ds = xr.open_dataset(path, chunks=chunks)
    dico = _orca_names_merged.get(varname)#, default=orca_names.orca_variables[varname])
    d_tg = dico["dims"]
    if "old_names" in dico:
        nam = next(d for d in dico["old_names"] if d in ds)
    else:
        nam = varname
    if verbose:
        print("fetching", nam, "from dataset")

    dimap = {k[0]:k for k in d_tg}
    res = ds.get([nam]).rename(dimap).squeeze()
    res[nam] *= dico.get("force_sign", 1)

    return res.rename({nam:varname})
