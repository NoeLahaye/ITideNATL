""" gridop
grid operations for the NEMO grid using xarray and xgcm capabilities through xorca
At this stage, most routines are for vertical grid operations, and in particular dealing with variable volume 
configuration (grid breathing) due to SSH, with z-level formulation (partial step and full step -- not checked)
"""
import xarray as xr

from xorca.orca_names import z_dims

from .utils import _parse_name_dict

def _get_z_dim(data):
    return next(iter(dim for dim in z_dims if dim in data.dims), None)
    
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
        
    return ds[nam] * ssh / hbot

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
        
    return ds[nam] * (1. + ssh / hbot)
        
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
    
    return comp_delz_ssh(hbot + ds[nam], ssh, hbot)
        
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
    if isinstance(ds, xr.Dataset):
        data = ds[which]
    else:
        data = ds
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
