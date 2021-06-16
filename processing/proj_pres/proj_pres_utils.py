import itidenatl.eos as eos
import itidenatl.vars as var
import itidenatl.gridop as gop

def get_pres(sds, sds_gr, sds_gm, grid, with_persist=False, densanom=False):
    """ compute pressure on mean grid from temperature, salinity, grid at rest and mean grid (and SSH)
    hydrostatic pressure is not included
    """
    sigmai = eos.sigmai_tsp(sds.votemper, sds.vosaline, 
                            -sds_gm.depth_c_3d, with_persist=with_persist
                           ).rename("sigmai")#.persist()
    if densanom: # note: mean sigma not interpolated on instantaneous grid here
        sigmai -= sds_gm.sigmai

    ssh_inst = sds.sossheig#.persist()
    ssh_pert = (ssh_inst-sds_gm.sossheig)#.persist()
    hbot = sds_gm.hbot#.persist()

    e3w = gop.get_rec_e3w(sds_gr.e3w, ssh=ssh_inst, hbot=hbot) # e3w inst.

    ### compute pressure, with surface pressure on mean z-grid
    if densanom:
        pres = var.comp_pres(sigmai, xgrid=grid, ssh=ssh_pert, zmet=e3w, sig0=0.)#.where(ds_gm.tmask)
    else:
        pres = var.comp_pres(sigmai, xgrid=grid, ssh=ssh_pert, zmet=e3w)#.where(ds_gm.tmask)
    if len(sds.chunks["z_c"]) != 1: # rechunk to initial chunk
        pres = pres.chunk({"z_c":sds.chunks["z_c"][0]})
    ### interpolate baroclinic part on mean grid (only baroclinic contribution)
    if True: # correction due to grid breathing
        delz = gop.get_del_zt(sds_gm.depth_c_3d, ssh=ssh_pert, hbot=hbot)
        gred = pres.grav/pres.rho0 if pres.red_pres else pres.grav
        pres += (sigmai-pres.sig0)*gred*delz # use pres.sig0
    return pres#.where(sds_gm.tmask)#.persist()
