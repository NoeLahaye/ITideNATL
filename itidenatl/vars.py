""" package for computing various fields from NEMO outputs, using xarray and xgcm through xorca library    

TODO:
     - update comp_pres* routines such that ds can be a DataArray only.     
"""
_pres_dico = dict(ssh=None, var_dens="sigmai", zmet=None, on_t_pts=True, 
                  s_dens_ano=False, red_pres=True, rho0=1026., grav=9.81, 
                  rechunk=True)

from .utils import _da_or_ds

def comp_pres(ds, xgrid, **kwargs):
    """ compute pressure anomaly by vertical integration of local in-situ density
    density interpolated on w-points, then integration goes from w -> T levels (implementation close to NEMO code)
    Pressure is computed by vertically integrating density: psurf + int_z \rho dz',
    with \rho=sigma_i (= potential density - 1000), and psurf = rho0 * ssh (+ dens ano if not variable-volume and s_dens_ano is True)
    adapted to variable volume formulation (enforced if zmet explicitly passed as an xarray)

    Parameters
    __________
    ds: xarray Dataset or DataArray
        dataset containing density - 1000. (for vertical integration)
        May also contain the ssh field and the grid interval
    xgrid: xgcm.Grid
        grid associated with ds. metrics will be used for vertical integration (unless delz is passed)
    zmet: str or xarray DataArray, optional (default: None)
        name of grid spacing array in ds dataset, or xr.DataArray containing grid spacing, or None (use xgrid metrics)
    ssh: str or xarray DataArray or False, optional
        name of ssh field in ds dataset, or xr.DataArray containing ssh, or None (read in ds based on default name) of False (use 0)
    var_dens: str, optional (default: "sigmai")
        name of density field in ds dataset (assume it is rho-1000)
    s_dens_ano: bool, optional (default: False)
        wether full density (rho0 + anomaly) is used for computing the surface pressure anomaly (at z=0) or not. 
        False is NEMO default

    Returns
    _______
    xarray.DataArray containing pressure anomaly

    """
    dico = _pres_dico.copy()
    dico.update(kwargs)
    rho0 = dico["rho0"] # pp_rau0 in CDFtools TODO have this from simulation output files
    sig0 = dico.get("sig0", rho0-1000.)   # rho0 - p_rau0
    grav = dico["grav"]  # TODO use common default with e.g. xorca
    s_dens_ano = int(dico["s_dens_ano"]) # 1 or 0: use density anomaly for surface pressure
    if xgrid.axes["Z"]._periodic:
        print("warning: xgrid is periodic in z, computation will very likely be wrong or NaNs")
    
    zmet = dico["zmet"]
    # get density at w points
    #dens = ds[dico["var_dens"]] - sig0 
    dens = _da_or_ds(ds, dico["var_dens"]) - sig0 # now dens is dens-rho0
    dens = xgrid.interp(dens, "Z", boundary="fill", fill_value=0) # rho/2 at surface
    
    ### vertical integration of density
    if zmet is None: # this will use grid metrics in xgrid
        res = xgrid.cumint(dens, "Z")
    else:
        if isinstance(zmet, str):
            zmet = ds[zmet]
        else:
            s_dens_ano = 0
        res = xgrid.cumsum(dens*zmet, "Z")

    ### compute pressure at surface (z=0)
    if dico["ssh"] is None:
        ssh = "sossheig"
    else:
        ssh = dico["ssh"]
    if ssh is not False:
        if isinstance(ssh, str):
            ssh = ds[ssh]
        psurf = ssh * (s_dens_ano*dens.isel(z_l=0)*2. + rho0) # factor 2 because of previous interp
    else:
        psurf = 0.
    
    # return pressure: multiply by gravity
    if dico["red_pres"]:
        grav /= rho0
    res = grav * (res + psurf)
    res.attrs = {k:dico[k] for k in ["grav","rho0","red_pres"]}
    res.attrs["sig0"] = sig0
    return res

def comp_pres_w(ds, xgrid, **kwargs):
    """ compute pressure anomaly by vertical integration of local potential density
    T -> w points
    Pressure is computing by vertically integrating density: psurf + int_z \rho dz',
    with \rho=sigma_i (potential density - 1000)
    Hydrostatic pressure at rest (rho_0*g*z) is not included: beware adding this contribution explicitly when computing derivatives along coordinates which z depends on.

    Parameters
    __________
    ds: xarray Dataset
        dataset containing density - 1000. (for vertical integration)
        May also contain the ssh field and the grid interval
    xgrid: xgcm.Grid
        grid associated with ds. metrics will be used for vertical integration (unless delz is passed)
    ssh: str or xarray DataArray or bool, optional
        name of ssh field in ds dataset, or xr.DataArray containing ssh, or None (automatic) of False (use 0)
    var_dens: str, optional (default: "sigmai")
        name of density field in ds dataset (assume it is rho-1000)
    zmet: str or xarray DataArray, optional (default: None)
        name of grid spacing array in ds dataset, or xr.DataArray containing grid spacing, or None (use xgrid metrics)
    on_t_pts: bool, optional (default: True)
        wether pressure field must be re-interpolated on density grid or not
    s_dens_ano: bool, optional (default: False)
        wether full density (rho0 + anomaly) is used for computing the surface pressure anomaly (at z=0) or not. False is NEMO default

    Returns
    _______
    xarray.DataArray containing pressure anomaly

    """
    rho0 = 1026. # pp_rau0 in CDFtools TODO have this from simulation output files
    sig0 = rho0-1000.   # rho0 - p_rau0
    grav = 9.81  # TODO use common default with e.g. xorca
    dico = _pres_dico.copy()
    dico.update(kwargs)
    s_dens_ano = int(dico["s_dens_ano"]) # 1 or 0: use density anomaly for surface pressure


    ### define ssh anomaly
    if dico["ssh"] is None:
        ssh = "sossheig"
    if not ssh:
        ssh = 0.
    elif isinstance(ssh, str):
        ssh = ds[ssh]

    dens = ds[dico["var_dens"]] - sig0 # now dens is dens-rho0

    ### compute pressure at surface (z=0)
    psurf = ssh * (s_dens_ano*dens.isel(z_c=0) + rho0)

    ### vertical integration of density
    zmet = dico["zmet"]
    if zmet is None: # this will use grid metrics in xgrid
        res = xgrid.cumint(dens, "Z", boundary="fill", fill_value=0.)
    else:
        if isinstance(zmet, str):
            zmet = ds[zmet]
        res = xgrid.cumsum(dens*zmet, "Z", boundary="fill", fill_value=0.)

    ### re-interpolate on t-points
    if dico["on_t_pts"]:
        res = xgrid.interp(res, "Z", boundary="extrapolate")

    # return pressure: multiply by gravity
    if dico["red_pres"]:
        grav /= rho0
    return grav * (res + psurf)
