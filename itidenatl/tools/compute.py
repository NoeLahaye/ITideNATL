""" utilitaries for xarray/dask computation """
import numpy as np
from dask.distributed import wait
import xarray as xr
from collections import OrderedDict

from . import misc as ut
from . import dataio as io

def custom_distribute(ds_or_da, op=None, 
                      to_dir=None, 
                      restart=False,
                      by_chunks=True,
                      dims=None,
                      verbose=False,
                      store_coords=False,
                      _reg=None,
                      **kwargs,
                     ):
    """ Distribute an embarrasingly parallel calculation manually along desired dimensions. 
    Data can be written to a zarr store or directly persisted in memory.
    
    Example usages:
    ds_out = custom_distribute(ds, lambda ds: ds.mean("time"), dim_0=2)
    ds_out = custom_distribute(ds, lambda ds: ds.mean("time"), dim_0=2, to_dir="/path/to/store/")

    Parameters
    ----------
    ds_or_da: xr.Dataset or xr.DataArray
        Input data
    op: func, optional
        Function to apply on ds_or_da (if None, compute ds_or_da). Default: False
    to_dir: str, optional
        output directory. If None or False, persist data in memory. Default: None
    restart: bool
        For restarting (only if writing in a file). Not implemented
    by_chunks: bool
        process by chunks. Default: True
    dims: dict or list or tuple
        dict of {dimension: segment size} pairs for distributing. segment size 1 if list or tuple is provided. Segment size referes to number of chunks if by_chunks=True, else to number of elements. Dict of (dim,size) can be passed directly as kwargs. Default: None
    verbose: bool
        display some informations about the running computation. Default: False
    store_coords: bool
        ignored if not to_dir. Store coordinates as well (if False, drop them). Default: False
    **kwargs:
        kwarg alternative to providing dims as a dict. 
        Dimensions with chunk size, e.g. (..., dim_0=2) processes data sequentially in chunks
        of size 2 along dimension dim_0
    """
        
    if restart:
        #assert tmp_dir is not None, "you need to provide tmp_dir if `restart=True`"
        raise NotImplementedError("restart not implemented")
        
    if op is None:
        op = lambda ds_or_da: ds_or_da

    if isinstance(dims, dict):
        dims = OrderedDict(dims)
    elif isinstance(dims, (list, tuple)):
        dims = OrderedDict([(d,1) for d in dims])
    else:
        dims = {k:v for k,v in kwargs.items() if k in ds_or_da.dims}

    d = list(dims.keys())[0]
    c = dims.pop(d)

    Nd = ds_or_da[d].size
    if by_chunks:
        dim = np.r_[0, np.cumsum(ut.get_chunks(ds_or_da, d))]
        chunks = dim[np.r_[np.arange(0, len(dim)-1, c), -1]]
    else:
        chunks = [np.arange(0, Nd-1, c), Nd]

    res = op(ds_or_da)
    if _reg is None:
        _reg = {d:slice(None,None) for d in res.dims}
        if to_dir: # first time: create dataset
            tozarr = io._prep_to_zarr(res, drop_coords=not(store_coords))
            tozarr.to_zarr(to_dir, consolidated=True, compute=False, mode="w")
    if not to_dir:
        D = []
    if not dims:
        print(d, end=": ")

    for i1, i2 in zip(chunks[:-1], chunks[1:]):
        sli = slice(i1, i2)
        _reg[d] = sli
        _ds = ds_or_da.isel({d: sli})
        if dims:
            if verbose:
                print(f"{d}:({sli.start},{sli.stop})", end=", ")
            ds_out = custom_distribute(_ds, op=op, to_dir=to_dir, 
                                        by_chunks=by_chunks, 
                                        dims=dims, _reg=_reg,
                                        verbose=verbose
                                      )
            if not to_dir:
                D.append(ds_out)
        else:
            if verbose:
                print(f"({sli.start},{sli.stop})", end=", ")
            ds_out = op(_ds)
            if not to_dir:
                # persist data and wait for completion
                ds_out = ds_out.persist()
                _ = wait(ds_out)
                D.append(ds_out)
            else:
                # store
                tozarr = io._prep_to_zarr(ds_out, drop_coords=not(store_coords))
                tozarr.to_zarr(to_dir, compute=True, region=_reg)

    # merge results back and return
    if not to_dir:
        ds = xr.concat(D, d)
    else:
        ds = xr.open_zarr(to_dir)
    if verbose and dims:
        print("")
    return ds

