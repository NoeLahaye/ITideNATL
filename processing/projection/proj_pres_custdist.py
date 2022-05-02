#!/usr/bin/env python
# coding: utf-8

#from glob import glob
from pathlib import Path
import os, time, sys, shutil
import logging
logging.basicConfig(format='[{}] %(asctime)s -- %(message)s'.format(sys.argv[0]), 
        level=logging.INFO, stream=sys.stdout)

#import numpy as np

import xarray as xr
from xgcm import Grid

from itidenatl.tools import files as uf, dataio as io, compute as uc
from proj_utils import load_grid_ds#, proj_pres, get_pres_one_dg as get_pres
from itidenatl import eos, vars as var, gridop as gop

homedir = Path(os.getenv("HOME"))
workdir = Path(os.getenv("WORKDIR"))
scratch = Path(os.getenv("SCRATCHDIR"))
worksha = Path("/work/CT1/ige2071/SHARED")
grid_mode_path = workdir/"eNATL60_grid_vmodes_proj_pres.zarr" 
    
if len(sys.argv)==1:
    raise ValueError("iday ?")
iday = int(sys.argv[1])
inp_chks = {"t":1, "z":30, "y":100, "x":-1}
wrk_chks = {"t":1, "z":30, "y":100, "x":-1}
str_chks = {"mode":1, "t":24, "y_c":400, "x_c":-1}
finish = True
if len(sys.argv)==3:
    if sys.argv[2] == "finish":
        regy = None
    else:
        regy = slice(int(sys.argv[2]), None)
elif len(sys.argv)==4:
    regy = slice(int(sys.argv[2]), int(sys.argv[3]))
    finish = False
else:
    regy = slice(0, None)
sk_y, sk_t = 100, 6
store_ts, store_rho, store_pres = False, False, True ### warning other than F, F, T may not work

da = uf.get_date_from_iday(iday)
log_dir = homedir/"working_on/processing/log_proj_pres"
log_file = "proj_pres_{}.log" #.format(i_day)
from_files = {v:uf.get_eNATL_path(v, iday) for v in ["sossheig", "votemper", "vosaline"]}
from_files["grid_mode"] = grid_mode_path
file_list = scratch/f"prov_proj-p_{iday}_file-list.txt"
tmp_fil = scratch/f"prov_proj-p_{iday}.zarr"
out_file = worksha/f"modal_proj/modamp_pres/modamp_pres_global_{da}.zarr" 
    
### local routines
def calc_rho(ds, dg):
    ssh_inst = ds.sossheig
    ssh_moy = dg.sossheig
    hbot = dg.hbot

    dep = (dg.depth_c_m - ssh_moy) * (hbot + ssh_inst) / (hbot + ssh_moy)
    rhored = eos.rho_gsw_tsp(ds.votemper, ds.vosaline, -dep).rename("rhoinsitu")
    return rhored

def comp_pres(ds, dg, grid, anom=True):
    ssh_inst = ds.sossheig
    ssh_pert = ssh_inst - dg.sossheig
    hbot = dg.hbot
    if "rhoinsitu" not in ds:
        rhored = calc_rho(ds, dg)
    else:
        rhored = ds.rhoinsitu
    e3w = gop.get_rec_e3w(dg.e3w_0, ssh=ssh_inst, hbot=hbot)
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

def calc_pmod(ds, dg, grid):
    pres = comp_pres(ds, dg, grid)
    pmod = (pres * dg.phi).sum("z_c")/dg.norm
    pmod = pmod.to_dataset(name="pres") #proj_pres(pres, dg)
    return pmod
    
def pre_proj_ychk(ds, regy, xgrid):
    sds = ds.isel(y_c=regy)
    ds_g = load_grid_ds(grid_mode_path, chunks=wrk_chks, 
            region={"y":regy}
                       )
    dg = ds_g.get(["depth_c_m", "e3w_0", "pres_m", "sossheig"]).persist()
    ds_g.close(); del ds_g
    if store_ts:
        sds.to_zarr(tmp_fil, mode="w", compute=True, consolidated=True)
        sds = xr.open_zarr(tmp_fil)

    if store_rho:
        rho = calc_rho(sds, dg).reset_coords(drop=True).to_dataset()
        rho["sossheig"] = sds.sossheig
        rho.to_zarr(tmp_fil, mode="w", compute=True, consolidated=True)
        sds.close; del sds
        sds = xr.open_zarr(tmp_fil)

    if store_pres:
        pres = comp_pres(sds, dg, xgrid).rename("pres").reset_coords(drop=True)
        uc.custom_distribute(pres, to_dir=tmp_fil, dims={"t":sk_t})
        sds.close(); del sds
        sds = xr.open_zarr(tmp_fil)
    return sds
    
def proj_ychk(ds, regy, xgrid):
    ds_g = load_grid_ds(grid_mode_path, chunks=wrk_chks, 
                        region={"y":regy}
                       )
    dg = (ds_g.phi * ds_g.e3t_m).to_dataset(name="phi")
    dg["norm"] = ds_g.norm
    dg = dg.persist()
    ds_g.close(); del ds_g
    
    if not store_pres:
        pmod = calc_pmod(sds, dg, xgrid)
    else:
        pmod = (sds.pres * dg.phi).sum("z_c")/dg.norm
        pmod = pmod.to_dataset(name="pres") #proj_pres(sds.pres, ds_g)
        fname = f"modamp_pres_{iday}_y{regy.start}-{regy.stop}.zarr"
        uc.custom_distribute(pmod, to_dir=scratch/fname, dims={"t":sk_t})
    return scratch/fname
    
################################################################################ 
##########   - - -   Start Main Work   - - -   ################################# 
################################################################################ 
if __name__ == "__main__":
    if (log_dir/log_file.format(da)).exists():
        raise ValueError("{} already processed? found corresponding log file".format(da))
        os._exit()
    logging.info("will process date {} (i_day {})".format(da,iday))

    from dask.distributed import LocalCluster, Client#, wait
    import dask
    dask.config.set({"distributed.workers.memory.spill":0.8})
    if False:
        cluster = LocalCluster(n_workers=7, local_directory=scratch)
        client = Client(cluster)
    else:
        from dask_mpi import initialize
        initialize(nthreads=8, interface="ib0", memory_limit=10e9,
                    dashboard=False, local_directory=scratch)
        client = Client()
    logging.info("Cluster should be connected -- dashboard at {}".format(client.dashboard_link))
    
    if regy is not None:
        tmes = time.time()
        ds = io.open_one_var(from_files["sossheig"],
                            chunks=inp_chks, varname="sossheig"\
                            ).reset_coords(drop=True)
        for v in ["vosaline", "votemper"]:
            ds = ds.merge(io.open_one_var(from_files[v], 
                                        chunks=inp_chks, varname=v\
                                        ).reset_coords(drop=True)
                         )
        logging.info("opened T, S and SSH data -- ellapsed time {:.1f} s".format(time.time()-tmes))
    
        ds_g = load_grid_ds(grid_mode_path, chunks=wrk_chks)
        xgrid = Grid(ds_g, periodic=False)
        if regy is not None and regy.stop is None:
            regy = slice(0, ds_g.y_c.size)
        ds_g.close(); del ds_g
        
        logging.info("opened static fields, reading from {}".format(grid_mode_path.name) + "; starting y-loop")
        for jy in range(regy.start, regy.stop, sk_y):
            regy = slice(jy, jy+sk_y)
            tmes, tmeg = time.time(), time.time()
            sds = pre_proj_ychk(ds, regy, xgrid)
            logging.info("prep. pressure, took {:.1f} s".format(time.time()-tmes))
            tmes = time.time()
            filnam = proj_ychk(sds, regy, xgrid)
            del sds
            with open(file_list, "a") as fp:
                fp.write(str(filnam)+"\n")
            logging.info("projected and stored, took {:.1f} s".format(time.time()-tmes))
            logging.info("jy={}-{} took {:.1f} s".format(jy,jy+sk_y, time.time()-tmeg))
            #client.restart()
        shutil.rmtree(tmp_fil)
    
### read files, rechunk and store
    if finish:
        tmes = time.time()
        with open(file_list, "r") as fp:
            tmp_files = list(set(r.rstrip('\n') for r in fp.readlines()))
        ds = xr.open_mfdataset(tmp_files, engine="zarr")
        ds = ds.chunk(str_chks)
        ds.pres.encoding.pop("chunks", None)
        attrs = dict(description = "pressure modal amplitude, resulting from the projection "
                         "of the pressure (anomaly) on the vertical modes",
                     day_of_simulation = da, iday = iday,
                     simulation = "eNATL60 (with tides)",
                     from_files = [str(v) for v in from_files.values()],
                     generating_script = sys.argv[0],
                     date_generated = logging.time.strftime("%Y-%m-%d (%a at %H h)"),
                     creator = "N. Lahaye (noe.lahaye@inria.fr)"
                     )
        ds.attrs = attrs
        ds.pres.attrs.update(dict(long_name="reduced pressure anomaly modal amplitude",
                                      units="m^2/s^2"
                                                          ))
        ds.to_zarr(out_file, mode="w", compute=True, consolidated=True, safe_chunks=True)
        logging.info(f"rechunked and stored final file in {out_file}" 
                       "; took {:.1f} s; THE END".format(time.time()-tmes))
        for f in tmp_files:
            shutil.rmtree(f)
        file_list.unlink()
    
        ### create log file with some information
        with open(log_dir/log_file.format(da), "w") as fp:
            fp.write("JOB ID: {}\n".format(os.getenv("SLURM_JOBID")))
            fp.write("python script: {}\n".format(sys.argv[0]))
            fp.write(f"i_day {iday}, date {da}\n")
            fp.write(f"nk_t {sk_t}, sk_y {sk_y}\n")
            fp.write(f"read chunks {inp_chks}, working chunks {wrk_chks}, store chunks {str_chks}\n")
    
        #client.close(); cluster.close()
