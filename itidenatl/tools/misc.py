import xarray as xr

### Miscellaneous
def _is_complex(da):
    return da.dtype.name.startswith("complex")

### for custom_distribute
def _print_dic(dico, doprint=True):
    str_prt = ", ".join([f"{k}: {v}" for k,v in dico.items()])
    if doprint:
        print(str_prt)
        return
    else:
        return str_prt

def get_chunks(ds_or_da, dim, firstonly=True):
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

def _ds_or_da_to_ds(ds_or_da):
    if isinstance(ds_or_da, xr.Dataset):
        return ds_or_da
    else:
        return ds_or_da.to_dataset(name=ds_or_da.name)
