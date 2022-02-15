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

def _find_common_dims(ds_or_dalist, what="data_vars"):
    """ ds_or_dalist: xarray.Dataset or list(xarray.DataArray) """
    if isinstance(ds_or_dalist, xr.Dataset):
        if what == "coords":
            data = ds_or_dalist.coords.values()
        elif what == "data_vars":
            data = ds_or_dalist.data_vars.values()
        else:
            raise ValueError("unrecognized option 'what' for the Dataset")
    elif isinstance(ds_or_dalist, list):
        data = ds_or_dalist
    else:
        raise ValueError("ds_or_dalist must be a xarray.Dataset or list of xarray.DataArray")
                            
    return list(set.intersection(*map(set, [v.dims for v in data])))
