""" gridop
grid operations for the NEMO grid using xarray and xgcm capabilities through xorca
At this stage, most routines are for vertical grid operations, and in particular dealing with variable volume 
configuration (grid breathing) due to SSH, with z-level formulation (partial step and full step -- not checked)
"""
import numpy as np
#import dask.array as dr
import xarray as xr
from xgcm import Grid

from xorca.orca_names import z_dims
from .tools import misc as ut

# miscellaneous local routines
_name = {"et":"e3t", "ew":"e3w", "ssh":"sossheig", "mask":"tmask",
        "zt":"depth_c", "zw":"depth_l"}
def _parse_name_dict(dico, special=None):
    """ wrapper of parse_inp_dict to use _name dict as default"""
    return ut.parse_inp_dict(dico, _name, special)

# grid related utilitaries. see also in tools.misc.py
def _get_z_dim(data):
    return next(iter(dim for dim in z_dims if dim in data.dims), None)

def _get_dim_from_d(ds_or_da, dim):
    """ return full dimension name from first letter (e.g. "x" -> "x_c" or "x_r") """
    if isinstance(dim, (list, tuple)):
        diname = {d: _get_dim_from_d(ds_or_da, d) for d in dim}
    elif isinstance(dim, str):
        diname = next(d for d in ds_or_da.dims if d[0]==dim)
    else:
        raise TypeError("dim must be of type str or list(str) or tuple(str)")
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

def _has_metrics(xgrid):
    """ check if xgcm.Grid object has metrics. Preliminary version """
    return bool(len(xgrid._metrics))

def get_hbot(ds, name=None, overwrite=False):
    """ compute depth (positive) of bottom, "hbot", from grid metrics (interval) and mask 
    For xorca object, NEMO simulation
    
    Parameters
    __________
    ds: xarray Dataset
        dataset containing grid interval (eg. "e3t") and mask (eg. "tmask"). 
        z dimension is assumed to be "z_c"
    name: dict or str, optional
        name of variables to use in ds. If str, name of grid interval.
        default is {"et":"e3t", "mask":"tmask"}
        
    Returns:
        xarray DataArray names "hbot" containing bottom depth
    """
    
    if "hbot" in ds and not overwrite: # just read variable if already present
        return ds["hbot"]
    else:
        name = _parse_name_dict(name, special="et")
        et, mask = ds[name["et"]], ds[name["mask"]]
        return et.where(mask).sum("z_c").rename("hbot")

### Vertical grid metrics correction due to varying SSH
def get_del_e3z(ds, which=None, ssh=None, hbot=None, name=None):
    """ compute level thickness (metrics)  perturbation associated with SSH (domvvl) for NEMO grid
    Use either dataset containing all required fields, or DataArray of grid interval
    Grid metric perturbation is simply computed as follows: e3[t,w] * ssh / hbot
    
    Parameters
    __________
    ds: xarray Dataset or DataArray
        contains all fields (vertical grid metrics + ssh + hbot) or just grid metrics
    which: str, optional {"t", "w"}
        which grid level to compute frid perturbation for. 
        Not necessary if ds is the grid metrics DataArray.
    ssh: xarray Datarray, optional
        Sea Surface Height. Not mandatory if present in ds
    hbot: xarray DataArray, optional
        depth of bottom. Not mandatory if present in ds
    name: str or dict, optional
        name of grid metrics or name of the different variables to be used. 

    Returns
    _______
    xarray DataArray of vertical grid metrics perturbation
    
    See also
    ________
    get_rec_e3z, get_del_zlev
    """
    special = "e"+which if which is not None else None
    name = _parse_name_dict(name, special)
    #if name is None:
    #    name = _name
    #elif not isinstance(name, dict):
    #    nam = name
    #else:
    #    nam = _name.copy()
    #    nam.update(name)
    #    name = nam.copy()
    #    if which is not None:
    #        nam = name["e"+which]
        
    nam = ds.name if which is None else name["e"+which]
    
    if ssh is None:
        ssh = ds[name["ssh"]]
    if hbot is None:
        hbot = get_hbot(ds, name=name)
        
    return ut.ds_to_da(ds,nam) * ssh / hbot

def get_del_e3t(ds, **kwargs):
    """ get grid metrics perturbation at t-level from xarray Dataset or DataArray 
    See: get_del_e3z """
    return get_del_e3z(ds, which="t", **kwargs)

def get_del_e3w(ds, **kwargs):
    """ get grid metrics perturbation at w-levels from xarray Dataset or DataArray 
    See: get_del_e3z """
    return get_del_e3z(ds, which="w", **kwargs)

def get_rec_e3z(ds, which=None, ssh=None, hbot=None, name=None):
    """ compute perturbated level thickness (metrics) associated with SSH (domvvl) for NEMO grid
    Use either dataset containing all required fields, or DataArray of grid interval
    Perturbated grid metric is simply computed as follows: e3[t,w] * (1. + ssh / hbot)
    
    Parameters
    __________
    ds: xarray Dataset or DataArray
        contains all fields (vertical grid metrics + ssh + hbot) or just grid metrics
    which: str, optional {"t", "w"}
        which grid level to compute frid perturbation for. 
        Not necessary if ds is the grid metrics DataArray.
    ssh: xarray Datarray, optional
        Sea Surface Height. Not mandatory if present in ds
    hbot: xarray DataArray, optional
        depth of bottom. Not mandatory if present in ds
    name: str or dict, optional
        name of grid metrics or name of the different variables to be used. 

    Returns
    _______
    xarray DataArray of vertical grid metrics perturbation
    
    See also
    ________
    get_del_e3z, get_del_zlev
    """
    special = "e"+which if which is not None else None
    name = _parse_name_dict(name, special)
    nam = ds.name if which is None else name[special]

    if ssh is None:
        ssh = ds[name["ssh"]]
    if hbot is None:
        hbot = get_hbot(ds, name=name)
        
    return ut.ds_to_da(ds,nam) * (1. + ssh / hbot)
        
def get_rec_e3t(ds, **kwargs):
    """ get perturbated grid metrics at t-level from xarray Dataset or DataArray 
    See: get_rec_e3z """
    return get_rec_e3z(ds, which="t", **kwargs)

def get_rec_e3w(ds, **kwargs):
    """ get perturbated grid metrics at w-level from xarray Dataset or DataArray 
    See: get_rec_e3z """
    return get_rec_e3z(ds, which="w", **kwargs)

### Vertical grid levels correction due to varying SSH
def get_del_zlev(ds, which=None, ssh=None, hbot=None, name=None):
    """ compute vertical depth perturbation (positive upward) associated with SSH (domvvl) for NEMO grid
    Use either dataset containing all required fields, or DataArray of grid interval
    Level perturbation is simply computed as follows: hab * ssh / hbot, where hab is height above bottom
    
    Parameters
    __________
    ds: xarray Dataset or DataArray
        contains all fields (algebraic z at rest + ssh + hbot) or just z at rest
    which: str, optional {"t", "w"}
        which grid level to compute frid perturbation for. 
        Not necessary if ds is the grid metrics DataArray.
    ssh: xarray Datarray, optional
        Sea Surface Height. Not mandatory if present in ds
    hbot: xarray DataArray, optional
        depth of bottom. Not mandatory if present in ds
    name: str or dict, optional
        name of depth or name of the different variables to be used.

    Returns
    _______
    xarray DataArray of vertical grid level perturbation (positive upward)
    
    See also
    ________
    get_del_e3z, get_rec_e3z, comp_delz_ssh
    """    
    special = ds.name if which is None else "z"+which
    name = _parse_name_dict(name, special)
    nam = ds.name if which is None else name[special]

    #if which is None:
    #    nam = ds.name
    if ssh is None:
        ssh = ds[name["ssh"]]
    if hbot is None:
        hbot = get_hbot(ds, name=name)
    
    return comp_delz_ssh(hbot + ut.ds_to_da(ds,nam), ssh, hbot)
        
def get_del_zt(ds, **kwargs):
    """ compute vertical level perturbation at T-levels, due to SSH.
    see: get_del_zlev """
    return get_del_zlev(ds, which="t", **kwargs)
        
def get_del_zw(ds, **kwargs):
    """ compute vertical level perturbation at T-levels, due to SSH.
    see: get_del_zlev """
    return get_del_zlev(ds, which="w", **kwargs)

def comp_delz_ssh(hab0, ssh, hbot):
    """ inputs: height above bottom, ssh, bottom depth (positive)"""

    return hab0*ssh/hbot

### correction of field / interpolation from moving grid to grid at rest
def corr_zbreath(ds, xgrid, hbot=None, ssh=None, which=None, name=None):
    """ correct field to account for moving vertical grid due to SSH. 
    This is equivalent to linearly interpolating data from the moving grid to the grid at rest.
    It is assumed that depth is sorted by increasing depth. 
    Computation goes as follows: f(z_rest) = f(z_vvl) - delta_z * df/dz
    delta_z is computed using get_del_zlev, and df/dz using grid metrics at rest
    
    Parameters
    __________
    ds: xarray Dataset or DataArray
        dataset containing field to interpolate and possibly other data needed for computation
        or DataArray of the field to interpolate only (not working well, still need information 
        on the complementary grid)
    xgrid: xgcm.Grid 
        grid associated with ds. Only the vertical coorinates are important. Metrics are not used.
    hbot: xarray DataArray, optional
        bottom depth (positive). Will be searched for in ds if not passed explicitly
    ssh: xarray DataArray, optional
        Sea Surface Height. Will be searched for in ds if not passed explicitly
    which: str, optional
        name of the variable to interpolate in ds. Mandatory if ds is a DataSet
    name: dict, optional
        dictionary with names of required fieldsfor the computation. Wil use default, _name if not passed
        
    Returns
    _______
    xarra DataArray of interpolated field on the grid at rest
    
    See also
    ________
    get_del_e3z, get_rec_e3z, comp_delz_ssh, get_del_zlev

    
    Todo
    ____
     - implement using metrics from grid object instead of searching for e3t or e3w
    """
    name = _parse_name_dict(name)
    data = ut.ds_to_da(ds, which)
    zdim = _get_z_dim(data)
    if zdim:
        nam = name["ew"] if zdim=="z_c" else name["et"] # inverted
        which = "t" if zdim=="z_c" else "w"
        e3z = ds[nam]
    else:
        raise ValueError("unable to find vertical dimension")
        
    if hbot is None:
        hbot = get_hbot(ds, name=name)
    if ssh is None:
        ssh = ds[name["ssh"]]
    delz = get_del_zlev(ds, which=which, hbot=hbot, ssh=ssh, name=name)
    dfdz = xgrid.interp(xgrid.diff(data, "Z", boundary="extend")/e3z, "Z", boundary="extend")
    return data + dfdz * delz # NB: dfdz=-dfdz because z sorted by increasing depth

def interp_z(da, zi=None, zt=None, xgrid=None, boundary="extrapolate", true_interp=True, e3=None):
    """ interpolate from one z-grid to another using xgcm. Weighted average if true_interp, otherwise unweighted interpolation (equivalent to mid-point linear interpolation)
    """
    if xgrid is None:
        xgrid = Grid(da, periodic=None)
    res = xgrid.interp(da, "Z", boundary=boundary)
    if true_interp:
        zd = _get_z_dim(da)[-1]
        if zt is None:
            tmp = "depth_c_3d" if zd=="w" else "depth_l_3d"
            zt = da[tmp] if tmp in da else da[tmp[:-3]]
        if zi is None:
            tmp = "depth_c_3d" if zd=="t" else "depth_l_3d"
            zi = da[tmp] if tmp in da else da[tmp[:-3]]
        if e3 is None:
            e3 = "e3t" if zd == "w" else "e3w"
            if e3 in da.coords:
                e3 = da[e3]
            else:
                e3 = zt[e3]
        res -= (zt - xgrid.interp(zi, "Z", boundary="extrapolate")) * xgrid.diff(da, "Z", boundary="extrapolate") / e3
    return res

### derivative routines
def diff_on_grid(da, dim, grid, upmask=False, diff_before_interp=True):
    """ compute derivative on the same grid
    Firs interpolate the function at mid points (unweighted average), then compute derivative back to initial points
    uses xgcm.Grid.derivative, assuming correct metrics are contained in grid object

    Parameters:
    ___________
    da: xarray.DataArray
        data to differentiate
    dim: str
        first letter of the dimension name (case sensitive) and the xgcm.Grid axis name (case insensitive)
    grid: xgcm.Grid object
        must contain correct grid metrics, otherwise you must divide the result by the grid increment
    upmask: Bool or str, optional (default: False)
        wether differentiation is computed "upwind" near masked values, in which case a mask must be provded.
        If str, will use the mask named upmask in da coordinates. If True, uses "tmask".
    diff_before_interp: bool, optional (default: True)
        Alternative where derivative is first computed at mid points, then interpolated backon initial points

    Returns:
    ________
        xarray.DataArray containing the derivated field
    """
    if diff_before_interp:
        dif = diff_on_grid_diffbeforeinterp(da, dim, grid, upmask)
    else:
        diname = _get_dim_from_d(da, dim)
        chk = da.chunks
        if chk:
            chk = chk[da.dims.index(diname)][0]
        dim = dim.upper()
        if upmask:
            mask = upmask
            tmask = da[mask] if isinstance(mask, str) else da["tmask"]
            da = da.where(tmask, 0)
        res = grid.interp(da, dim, boundary="extrapolate")
        if upmask:
            res = res * (1 + (grid.interp(tmask, dim, boundary="extrapolate")!=1))
        dif = grid.derivative(res, dim, boundary="extrapolate")
        if chk:
            dif = dif.chunk({diname:chk})
        if upmask:
            dif = dif.where(tmask)
    return dif

def diff_on_grid_diffbeforeinterp(da, dim, grid, upmask=False):
    """ compute derivative on the same grid
    dim must be the first letter of the dimension name (case sensitive) and the xgcm.Grid axis name (case insensitive)
    uses xgcm.Grid.derivative, assuming correct metrics are contained in grid object """
    diname = _get_dim_from_d(da, dim)
    chk = da.chunks
    if chk:
        chk = chk[da.dims.index(diname)][0]
    dim = dim.upper()
    dif = grid.derivative(da, dim, boundary="extrapolate")
    if upmask:
        mask = upmask
        tmask = da[mask] if isinstance(mask, str) else da["tmask"]
        umask = grid.interp(tmask, dim, boundary="extrapolate")
        dif = dif.where(umask==1, 0)
    res = grid.interp(dif, dim, boundary="extrapolate")
    if upmask:
        res = res * (1 + (grid.interp(umask, dim, boundary="extrapolate")!=1)).where(tmask)
    if chk:
        res = res.chunk({diname:chk})
    return res

### Other spatial operations: filtering
from scipy.ndimage import gaussian_filter

def fill_masked(da, limit, mask=None, dims=None):
    """ rudimentary version: replace with zeros, i.e. wrapper of xr.fillna) """
    if mask is not None:
        res = da.where(mask, 0.)
    else:
        res = da.fillna(0.) # warning this will give 0 when re-applying mask on output in gauss_filt
    return res

def gauss_filt(ds_or_da, **kwargs):
    """ apply scipy.ndimage.gaussian_filter to a xarray.DataArray, 
    along every dimension (assume isotropy and homogeneity of grid spacing)

    Parameters:
    __________
    ds_or_da: xarray.Dataset or xarray.DataArray
        data to smooth
    sigma: int, optional (default: 3)
        width of gaussian kernel in grid points, passed to scipy.ndimage.gaussian_filter
    truncate: int, optional (default: 4)
        window size in terms of sigma, passed to scipy.ndimage.gaussian_filter
    boundary: str, optional (default: "nearest")
        how to deal with boundaries. Passed to dask.map_overlap ("boundary" arg) and 
        scipy.ndimage.gaussian_filter ("mode" argument). 
        Possible values are "reflect", "nearest" and "periodic" or a scalar indicating the constant value to use
    fix_masked: bool, optional (default: True)
        whether masked value dealt with. See code for procedure
    TODO: 
        - restore coordinates
    """
    _bnds = {"reflect":"reflect", "nearest":"nearest", "periodic":"wrap"}
        
    if isinstance(ds_or_da, xr.Dataset):
        res = xr.merge([gauss_filt(da, **kwargs) 
                         for da in ds_or_da.data_vars.values()])
    else:
        sigma, truncate = kwargs.get("sigma", 3), kwargs.get("truncate", 4)
        boundary, fix_masked = kwargs.get("boundary", "nearest"), kwargs.get("fix_masked", False)
        if fix_masked:
            if "mask" in "kwargs":
                mask = kwargs["mask"]
            elif "tmask" in ds_or_da.coords:
                mask = ds_or_da["tmask"]
            elif "tmaskutil" in ds_or_da.coords:
                mask = ds_or_da["tmaskutil"]
            else:
                mask = np.isfinite(ds_or_da)
            da = fill_masked(ds_or_da, limit=sigma*truncate, mask=mask)
        else:
            da = ds_or_da
        gf_kwgs = dict(sigma=sigma, truncate=truncate)
        if isinstance(boundary, (float, int)):
            gf_kwgs.update({"mode":"constant", "cval":boundary})
        else:
            gf_kwgs.update({"mode":_bnds[boundary]})
        res = da.data.map_overlap(lambda x: gaussian_filter(x, **gf_kwgs), 
                                depth=sigma*truncate, boundary=boundary)
        res = xr.DataArray(res, dims=da.dims).rename(da.name)
        res = res.assign_coords(da.coords)
        if fix_masked: ### TODO fix this, it breaks everything
            res = res.where(mask)
    
    return res
