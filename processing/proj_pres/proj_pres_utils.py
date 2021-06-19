""" proj_pres_utils.py
definition of functions (mostly wrappers) used by the pressure modal projection script (e.g. proj_pres_ty-loop/py)
 - proj_pres for projecting the pressure on vertical modes
 - get_pres for computing the pressure from temperature and salinity
These are had hoc routines with no genericity of the datasets (e.g. names) ensured
"""
import itidenatl.eos as eos
import itidenatl.vars as var
import itidenatl.gridop as gop

def get_pres(ds, ds_gr, ds_gm, grid, with_persist=False, densanom=False):
    """ compute pressure on mean grid from temperature, salinity, grid at rest and mean grid (and SSH)
    hydrostatic pressure is not included
    """
    sigmai = eos.sigmai_tsp(ds.votemper, ds.vosaline, 
                            -ds_gm.depth_c_3d, with_persist=with_persist
                           ).rename("sigmai")#.persist()
    if densanom: # note: mean sigma not interpolated on instantaneous grid here
        sigmai -= ds_gm.sigmai

    ssh_inst = ds.sossheig#.persist()
    ssh_pert = (ssh_inst-ds_gm.sossheig)#.persist()
    hbot = ds_gm.hbot#.persist()

    e3w = gop.get_rec_e3w(ds_gr.e3w, ssh=ssh_inst, hbot=hbot) # e3w inst.

    ### compute pressure, with surface pressure on mean z-grid
    if densanom:
        pres = var.comp_pres(sigmai, xgrid=grid, ssh=ssh_pert, zmet=e3w, sig0=0.)#.where(ds_gm.tmask)
    else:
        pres = var.comp_pres(sigmai, xgrid=grid, ssh=ssh_pert, zmet=e3w)#.where(ds_gm.tmask)
    if len(ds.chunks["z_c"]) != 1: # rechunk to initial chunk
        pres = pres.chunk({"z_c":ds.chunks["z_c"][0]})
    ### interpolate baroclinic part on mean grid (only baroclinic contribution)
    if True: # correction due to grid breathing
        delz = gop.get_del_zt(ds_gm.depth_c_3d, ssh=ssh_pert, hbot=hbot)
        gred = pres.grav/pres.rho0 if pres.red_pres else pres.grav
        pres += (sigmai-pres.sig0)*gred*delz # use pres.sig0
    return pres#.where(ds_gm.tmask)#.persist()

def get_pres_one_dg(ds, ds_g, grid, with_persist=False, anom=True):
    """ compute pressure on mean grid from temperature, salinity, grid at rest and mean grid (and SSH)
    hydrostatic pressure is not included
    """
    sigmai = eos.sigmai_tsp(ds.votemper, ds.vosaline, 
                            -ds_g.depth_c_m, with_persist=with_persist
                           ).rename("sigmai")#.persist()
    ssh_inst = ds.sossheig#.persist()
    ssh_pert = (ssh_inst-ds_g.sossheig)#.persist()
    hbot = ds_g.hbot#.persist()

    e3w = gop.get_rec_e3w(ds_g.e3w_0, ssh=ssh_inst, hbot=hbot) # e3w inst.

    ### compute pressure, with surface pressure on mean z-grid
    pres = var.comp_pres(sigmai, xgrid=grid, ssh=ssh_pert, zmet=e3w)#.where(ds_gm.tmask)
    if len(ds.chunks["z_c"]) != 1: # rechunk to initial chunk
        pres = pres.chunk({"z_c":ds.chunks["z_c"][0]})
    ### interpolate baroclinic part on mean grid (only baroclinic contribution)
    if True: # correction due to grid breathing
        delz = gop.get_del_zt(ds_g.depth_c_m, ssh=ssh_pert, hbot=hbot)
        gred = pres.grav/pres.rho0 if pres.red_pres else pres.grav
        pres += (sigmai-pres.sig0)*gred*delz # use pres.sig0
    # take anomaly
    if anom:
        pres -= ds_g.pres_m
    # remove singletons
    pres = pres.reset_coords([c for c in pres.coords if len(pres[c].dims)==0], drop=True)
    return pres.astype(sigmai.dtype)

def proj_pres(pres, ds_g, keep_coords=["llon_cc", "llat_cc"], units="m^2/s^2"):
    """ cf in itidenatl.nemodez """
    pmod = (pres * ds_g["phi"] * ds_g["e3t_m"]).where(ds_g.tmask).sum("z_c") / ds_g.norm
    pmod = pmod.reset_coords([c for c in pmod.coords if c not in keep_coords and c not in pmod.dims], drop=True)
    pmod.attrs = {"units":units, "long_name":"reduced pressure modal amplitude"}
    pmod = pmod.astype(pres.dtype).to_dataset(name="pres")
    return pmod
                        
