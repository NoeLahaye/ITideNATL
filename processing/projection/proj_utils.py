""" proj_utils.py
definition of functions (mostly wrappers) used by the pressure modal projection script (e.g. proj_pres_ty-loop/py)
 - proj_pres for projecting the pressure on vertical modes
 - get_pres for computing the pressure from temperature and salinity
These are had hoc routines with no genericity of the datasets (e.g. names) ensured
"""
import xarray as xr
import itidenatl.eos as eos
import itidenatl.vars as var
import itidenatl.gridop as gop

def load_grid_ds(path, chunks=None, region=None):
    ds = xr.open_zarr(path)
    if chunks is not None:
        ds = ds.chunk({k:v for k,v in chunks.items() if k in ds.dims})
    if region is not None:
        ds = ds.isel({d:region[d[0]] for d in ds.dims if d[0] in region})
    return ds

def get_pres(ds, ds_gr, ds_gm, grid):
    """ compute pressure on mean grid from temperature, salinity, grid at rest and mean grid (and SSH)
    hydrostatic pressure is not included
    WARNING: this version not used any longer
    """
    ssh_inst = ds.sossheig#.persist()
    ssh_moy = ds_gm.sossheig
    ssh_pert = (ssh_inst - ssh_moy)#.persist()
    hbot = ds_gm.hbot#.persist()

    dep = (ds_gm.depth_c_3d - ssh_moy) * (hbot + ssh_inst) / (hbot + ssh_moy)
    rhored = eos.rho_gsw_tsp(ds.votemper, ds.vosaline, -dep).rename("rhoinsitu")#.persist()

    e3w = gop.get_rec_e3w(ds_gr.e3w, ssh=ssh_inst, hbot=hbot) # e3w inst.

    ### compute pressure, with surface pressure on mean z-grid
    pres = var.comp_pres(rhored, xgrid=grid, ssh=ssh_pert, zmet=e3w, 
                            rho_kind="rho_red"
                        )#.where(ds_gm.tmask)
    if len(ds.chunks["z_c"]) != 1: # rechunk to initial chunk
        pres = pres.chunk({"z_c":ds.chunks["z_c"][0]})
    ### interpolate baroclinic part on mean grid (only baroclinic contribution)
    if True: # correction due to grid breathing
        delz = gop.get_del_zt(ds_gm.depth_c_3d, ssh=ssh_pert, hbot=hbot)
        gred = pres.grav if pres.red_pres else pres.grav * pres.rho0
        pres += rhored * gred * delz # use pres.sig0
    return pres#.where(ds_gm.tmask)#.persist()

def get_pres_one_dg(ds, dg, grid, anom=True):
    """ compute pressure on mean grid from temperature, salinity, grid at rest and mean grid (and SSH)
    hydrostatic pressure is not included
    """
    ssh_inst = ds.sossheig#.persist()
    ssh_moy = dg.sossheig
    ssh_pert = (ssh_inst - ssh_moy)#.persist()
    hbot = dg.hbot#.persist()

    dep = (dg.depth_c_m - ssh_moy) * (hbot + ssh_inst) / (hbot + ssh_moy)
    rhored = eos.rho_gsw_tsp(ds.votemper, ds.vosaline, -dep).rename("rhoinsitu")#.persist()

    e3w = gop.get_rec_e3w(dg.e3w_0, ssh=ssh_inst, hbot=hbot) # e3w inst.

    ### compute pressure, with surface pressure on mean z-grid
    pres = var.comp_pres(rhored, xgrid=grid, ssh=ssh_pert, zmet=e3w,
                         rho_kind="rho_red")
    if len(ds.chunks["z_c"]) != 1: # rechunk to initial chunk
        pres = pres.chunk({"z_c":ds.chunks["z_c"][0]})
    ### interpolate baroclinic part on mean grid (only baroclinic contribution)
    if True: # correction due to grid breathing
        delz = gop.get_del_zt(dg.depth_c_m, ssh=ssh_pert, hbot=hbot)
        gred = pres.grav if pres.red_pres else pres.grav * pres.rho0
        pres += rhored * gred * delz # use pres.sig0
    # take anomaly
    if anom:
        pres -= dg.pres_m
    # remove singletons
    pres = pres.reset_coords([c for c in pres.coords if len(pres[c].dims)==0], drop=True)
    return pres.astype(rhored.dtype)

_proj_keep_coords = {"p":[], "u":[], "v":[]} # ["llon_cc", "llat_cc"]
_proj_units = {"p":"m^2/s^2", "u":"m/s", "v":"m/s"}
_proj_long_name = {"p": "reduced pressure modal amplitude", "u":"x-velocity modal amplitude",
                    "v":"y-velocity modal amplitude"}
_proj_mask = {v:v+"mask" for v in "puv"}
_proj_e3z = {v:"e3{}_m".format(v) for v in "puv"}
_proj_names = {"p":"pres", "u":"vozocrtx", "v":"vomecrty"}

def proj_pres(pres, ds_g, **kwargs):
    """ cf in itidenatl.nemodez """
    keep_coords = kwargs.get("keep_coords", _proj_keep_coords["p"])
    units = kwargs.get("units", _proj_units["p"])
    pmod = (pres * ds_g["phi"] * ds_g["e3t_m"]).where(ds_g.tmask).sum("z_c") / ds_g.norm
    pmod = pmod.reset_coords([c for c in pmod.coords if c not in keep_coords and c not in pmod.dims], drop=True)
    pmod.attrs = {"units":units, "long_name":"reduced pressure modal amplitude"}
    pmod = pmod.astype(pres.dtype).to_dataset(name="pres")
    return pmod
                        
### wrapper for pressure projection
def calc_pmod(ds, ds_g, grid):
    """ just a wrapper """
    pres = get_pres_one_dg(ds, ds_g, grid)
    pmod = proj_pres(pres, ds_g)
    return pmod

### u-v projection
def get_uv_mean_grid(ds, grid, ds_g=None, **kwargs):
    """ put field on mean grid given instantaneous SSH 
    
    Returns
    _______
    xarray.DataArray of interpolated "which" vield from ds (u: vozocrtx or v: vomecrty
    """
    if ds_g is None:
        ds_g = ds
    if "which" not in kwargs:
        which = next(k for k,v in _proj_names.items() if v in ds.data_vars)
        if which is None:
            raise ValueError("unable to determine what I am processing")
    dim = "xy"["uv".index(which)]

    data = ds[_proj_names[which]]
    ### First compute vertical derivative of u, v (approximate: use mean grid)
    e3z = kwargs.get("e3z", _proj_e3z[which])
    dim_z = next(d for d in data.dims if d.startswith("z_"))
    chk_z = ds.chunks[dim_z][0]
    res = grid.diff(grid.interp(data, "Z", boundary="extend"), "Z", boundary="extend") 
    res = (res / ds_g[e3z]).chunk({dim_z:chk_z})

    ### compute vertical grid elevation
    ladim = next(k for k in data.dims if k.startswith(dim))
    # interpolate ssh if required
    if next(k for k in ds.sossheig.dims if k.startswith(dim)) == ladim:
        ssh_inst = ds.sossheig
    else:
        ssh_inst = grid.interp(ds.sossheig, axis=dim.upper(), boundary="extrapolate").chunk(ds.chunks[ladim][0])
    ssh_pert = (ssh_inst - ds_g.sossheig) #.persist()
    dep = "depth_{}_m".format(which)
    delz = gop.get_del_zt(ds_g[dep], ssh=ssh_pert, hbot=ds_g.hbot).astype(data.dtype)

    return (data + res * delz).rename(data.name)

def proj_puv(data, ds_g, **kwargs):
    """ cf in itidenatl.nemodez """
    if "which" not in kwargs:
        if data.name in _proj_names.values():
            which = next(k for k,v in _proj_names.items() if v == data.name)
        else:
            raise ValueError("unable to determine what I am projecting")
    e3z = kwargs.get("e3z", _proj_e3z[which])
    mask = kwargs.get("mask", _proj_mask[which])
    keep_coords = kwargs.get("keep_coords", _proj_keep_coords[which])

    amod = (data * ds_g["phi"] * ds_g[e3z]).where(ds_g[mask]).sum("z_c") / ds_g.norm
    amod = amod.reset_coords([c for c in amod.coords if c not in keep_coords and c not in amod.dims], drop=True)
    amod.attrs = {"units": kwargs.get("units", _proj_units[which]), 
                "long_name": kwargs.get("long_name",_proj_long_name[which])
                }
    amod = amod.astype(data.dtype).to_dataset(name=data.name)
    return amod

