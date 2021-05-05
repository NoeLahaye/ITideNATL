""" package for computing various fields from NEMO outputs, using xarray and xgcm through xorca library """

_pres_dico = dict(ssh=None, var_dens="sigmai", zmet=None, on_t_pts=True, s_dens_ano=False, red_pres=True)

def comp_pres(ds, xgrid, **kwargs):
    """ compute pressure anomaly by vertical integration of local potential density
    Pressure is computing by vertically integrating density: psurf + int_z \rho dz',
    with \rho=sigma_i (potential density - 1000)

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

    TODO
    ____
     - update such that ds can be a DataArray only.
     - verify wether sigmai or (sigmai - 26.0) should be used
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
    psurf = ssh * (s_dens_ano*dens + rho0)

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
