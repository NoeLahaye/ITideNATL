""" routines related to reading/writing """
import numpy as np
import xarray as xr
from . import misc as ut
# ------------------------ xarray / xorca related ------------------------------- #

from xorca import orca_names

_orca_names_merged = {**orca_names.orca_coords, **orca_names.orca_variables}
# update a la mano
_orca_names_merged["vosigmainsitu"] = _orca_names_merged["votemper"]
_offset = {"c":1., "l":.5, "r":1.5}
# I could parse this from _orca_names_merged
_zdims_in_dataset = {"vosaline":"deptht", "votemper":"deptht", "vosigmainsitu":"deptht",
                     "vozocrtx":"depthu", "vomecrty":"depthv", "vovecrtz":"depthw", 
                     "sossheig":None, "sozocrtx":None, "somecrty":None
                     }

def open_one_var(path, chunks="auto", varname=None, verbose=False, **kwargs):
    """ utilitary function to open datasets for one variable 
    and return dataset with fixed dimension names and minimal coordinates 
    Works for 3D (t,y,x) or 4D (t,z,y,x) variable. Not check for others 
    
    Parameters
    __________
    TODO

    """
    ### infer targetted variable name from file names if not provided
    if not varname:
        path_ref = path[0] if isinstance(path, list) else path
        varname = next(v for v in str(path_ref).split("_") if len(v)==8 and v.startswith("vo"))
    else:
        if isinstance(path, list): # retain only variable files
            n_path = [v for v in path if varname in str(v)]
            path = path if len(n_path)==0 else n_path
            
    ### update chunk dict
    chunk_after = kwargs.get("chunk_after", False)
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
    chk_op = "auto" if chunk_after else chks
    if verbose:
        print("opening", path, "with chunking", chk_op, "and kwargs", kwargs)
    if isinstance(path, list):
        if "parallel" not in kwargs:
            kwargs["parallel"] = True
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time_counter"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        ds = xr.open_mfdataset(path, chunks=chk_op, **kwargs)
    else:
        ds = xr.open_dataset(path, chunks=chk_op, **kwargs)
    if chunk_after:
        ds = ds.chunk(chks)
        
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
    # add axis attributes to dimension coordinates for xgcm
    for dim in [d for d in dims_tg if d[0] in "xyz"]:
        ds[dim].attrs["axis"] = ds[dim].attrs.get("axis", dim[0].upper())
        if dim.endswith("l"):
            ds[dim].attrs["c_grid_axis_shift"] = ds[dim].attrs.get("c_grid_axis_shift",-0.5)
        elif dim.endswith("r"):
            ds[dim].attrs["c_grid_axis_shift"] = ds[dim].attrs.get("c_grid_axis_shift",+0.5)
    
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

def _prep_to_zarr(ds_or_da, drop_coords=True):
    ds = ut._ds_or_da_to_ds(ds_or_da)
    ds = _rm_encoding_chunks(ds)
    ds = _singletons_as_attrs(ds)
    if drop_coords:
        ds = ds.reset_coords(drop=True)
    return ds

def _rm_encoding_chunks(ds_or_da):
    for c in ds_or_da.coords:
        ds_or_da[c].encoding.pop("chunks", None)
    if isinstance(ds_or_da, xr.Dataset):
        for c in ds_or_da.data_vars:
            ds_or_da[c].encoding.pop("chunks", None)
        else:
            ds_or_da.encoding.pop("chunks", None)
    return ds_or_da

def _singletons_as_attrs(ds):
    cs = []
    for c in ds.coords:
        if len(ds[c].dims)==0:
            ds.attrs[c] = str(ds[c].values)
            cs.append(c)
    if len(cs):
        ds = ds.reset_coords(cs, drop=True)
    return ds

