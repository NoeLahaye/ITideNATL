import xarray as xr

### Miscellaneous
def _is_complex(da):
    return da.dtype.name.startswith("complex")

def parse_inp_dict(dico, defo, special=None):
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
            raise ValueError('unable to parse "dico" argument')
        return newdic

### numpy array manipulations
def get_subsample_slice(N, ns, offset=0):
    if offset == "auto":
        rpts = N - 1 - ns * ( (N-1) // ns )
        offset = rpts//2
    return slice(offset, N, ns)

def subsample_along_axis(a, axis, ns, offset=0):
    if offset == "auto":
        ntot = a.shape[axis] - 1
        rpts = ntot - ns * (ntot//ns)
        offset = rpts//2
    return slice_along_axis(a, axis, start=offset, step=ns)

def slice_along_axis(a, axis, start=0, end=None, step=1):
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]

### for custom_distribute
def _print_dic(dico, doprint=True):
    str_prt = ", ".join([f"{k}: {v}" for k,v in dico.items()])
    if doprint:
        print(str_prt)
        return
    else:
        return str_prt

def get_chunks(ds_or_da, dim=None, firstonly=True):
    if dim is None:
        dim = tuple(ds_or_da.dims)
    if isinstance(dim, (list, tuple)):
        res = {d: get_chunks(ds_or_da, d, firstonly) for d in dim}
    elif isinstance(dim, str):
        if isinstance(ds_or_da, xr.DataArray):
            dim = ds_or_da.dims.index(dim)
        res = ds_or_da.chunks[dim]
        if firstonly:
            res = res[0]
    else:
        raise TypeError("dim must be of type str or list(str) or tuple(str)")
    return res

def _da_to_ds(ds_or_da):
    if isinstance(ds_or_da, xr.Dataset):
        return ds_or_da
    else:
        return ds_or_da.to_dataset(name=ds_or_da.name)

def _ds_to_da(ds_or_da, name):
    """ return dataarray from dataset or dataarray """
    if isinstance(ds_or_da, xr.DataArray):
        return ds_or_da
    else:
        return ds_or_da[name]
