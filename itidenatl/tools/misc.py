import xarray as xr

### for custom_distribute
def _print_dic(dico, doprint=True):
    str_prt = ", ".join([f"{k}: {v}" for k,v in dico.items()])
    if doprint:
        print(str_prt)
        return
    else:
        return str_prt

def get_chunks(ds_or_da, dim):
    if isinstance(ds_or_da, xr.DataArray):
        dim = ds_or_da.dims.index(dim)
    return ds_or_da.chunks[dim]

def _ds_or_da_to_ds(ds_or_da):
    if isinstance(ds_or_da, xr.Dataset):
        return ds_or_da
    else:
        return ds_or_da.to_dataset(name=ds_or_da.name)

### for grid ops
def _get_dim_from_d(ds_or_da, dim):
    """ return full dimension name from first letter (e.g. "x" -> "x_c" or "x_r") """
    diname = next(d for d in ds_or_da.dims if d[0]==dim)
    return diname

