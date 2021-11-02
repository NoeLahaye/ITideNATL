#!/usr/bin/env python
# coding: utf-8
""" these version read static variables in various files; Chunking of zarr archive made the script stopping quite randomly, at the time of persisting. See test_simple_comp_v?.py in testings/. 
This script is deprecated in favour of proj_pres_ty-loop.py
"""

# """ this version makes a loop on y + time if needed """
#from glob import glob
from pathlib import Path
import os, time, sys
import logging
logging.basicConfig(format='[{}] %(asctime)s -- %(message)s'.format(sys.argv[0]), 
        level=logging.INFO, stream=sys.stdout)

import numpy as np

import xarray as xr
#from xorca.lib import load_xorca_dataset
from xgcm import Grid

from itidenatl.nemodez import Vmodes
#import itidenatl.eos as eos
#import itidenatl.vars as var
from proj_utils import get_pres
#import itidenatl.gridop as gop
import itidenatl.utils as ut

### Intialize dask (https://mpi.dask.org/en/latest/batch.html)
from distributed import Client, performance_report
#from dask.distributed import progress
scratch = Path(os.getenv("SCRATCHDIR"))
# method 1: dask-based script
if True:
    from dask_mpi import initialize
    #initialize(nthreads=3, interface="ib0", memory_limit=17e9, 
    initialize(nthreads=4, interface="ib0", memory_limit=21e9, 
            dashboard=True, local_directory=scratch)
    client=Client()
else:
    client = Client(scheduler_file="scheduler.json")
    #from dask.distributed import LocalCluster
    #cluster = LocalCluster(n_workers=6) #n_workers=24, threads_per_worker=1, memory_limit=8e6,silence_logs=50
    #client = Client(cluster)
logging.info("Cluster should be connected")

### define paths
workdir = Path("/work/CT1/ige2071/nlahaye")
worksha = (workdir/"../SHARED").resolve()

data_path = Path("/work/CT1/hmg2840/lbrodeau/eNATL60")
grid_rest_path = workdir/"eNATL60_rest_grid_cz10.zarr" # "mesh_mask_eNATL60_3.6.nc" #
grid_mean_path = scratch/"eNATL60_mean_grid_z_cz10.zarr" #"eNATL60_mean_grid_z_one-chk-z.zarr"
strat_path = scratch/"global_mean_bvf_chk200.zarr"
presm_path = scratch/"global_mean_pres.zarr"
vmod_path = worksha/"vmodes/vmodes_10.zarr"
ssh_path = None #Path("/store/albert7a/eNATL60/zarr/eNATL60-BLBT02-SSH-1h") # automatic (None) or zarr
out_dir = worksha/"modal_proj"
out_file = "modamp_pres_global_{}.zarr" #"modamp_subdom_{}.zarr" #.format(date)
log_dir = Path(os.getenv("HOME"))/"log"
log_file = "proj_pres_{}.log" #.format(i_day)

### define chunking and computational subdomains (y, t)
chunks = {"t":1, "z":20, "y":100, "x":-1}
nk_t = 1 # process nk_t instants at a time (must be a divider of 24)
sk_y = 200 # process y-subdomains of size sk_y at a time (choose it a multiple of chunk size)
assert 24%nk_t == 0 and sk_y%chunks["y"] == 0
restart = 12 # continue previously stopped job (y segments). 0, False or None to start from beginning

### read time ("day of simu in data") from sys.argv, or use here-defined value
if len(sys.argv)>1: #N.B.: we can process several days
    i_days = [int(k) for k in sys.argv[1:]]
else:
    i_days = [0] # default value (list)

### get date (day) and check for existing log files
sim_dates = ut.get_date_from_iday(i_days)
for da in sim_dates:
    if (log_dir/log_file.format(da)).exists():
        raise ValueError("{} already processed? found corresponding log file".format(da))
    logging.info("will process date {}".format(da))

drop_land_x = True  ### wether to drop land points in x (must be land for every y)
region = {"t":slice(0,None), "x":slice(0,None), "y":slice(0,None)}
if region["x"].start > 0 and drop_land_x:
    logging.info("reverting drop_land_x to False because subdomain is not implemented")
    drop_land_x = False
    
###  ----------------------------------------------------------------------  ###
###  ---------------------- End of user-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

# replicate chunks in terms of target dimensions
chunks_tg = {di:chunks[di[0]] for di in ["z_c", "z_l", "t"]}
for di in ("x", "y"):
    chunks_tg.update({di+"_"+su:chunks[di] for su in ["c","r"]})
logging.info("finished setting up parameters, starting to load data")

### Load grid at rest: I only need e3w
vars_keep = ["e3w"]
if grid_rest_path.suffix==".zarr": # read zarr
    ds_gr = xr.open_zarr(grid_rest_path).get(vars_keep).reset_coords(vars_keep).reset_coords(drop=True)
    if True: # remove trailing dimensions
        dims = []
        for d in ds_gr.data_vars.values():
            dims += list(d.dims)
        ds_gr = ds_gr.drop_dims([d for d in ds_gr.dims if d not in dims])
    ds_gr = ds_gr.chunk({k:v for k,v in chunks_tg.items() if k in ds_gr.dims})
else:
    v = vars_keep[0]
    ds_gr = ut.open_one_coord(grid_rest_path, chunks=chunks, varname=v)
    for v in vars_keep[1:]:
        ds_gr = ds_gr.merge(ut.open_one_coord(grid_rest_path, chunks=chunks, varname=v))

### load mean grid (associated with year-mean ssh)
ds_gm = xr.open_zarr(grid_mean_path)
ds_gm = ds_gm.set_coords([c for c in ds_gm.data_vars if c != "sossheig"])
ds_gm = ds_gm.chunk({k:v for k,v in chunks_tg.items() if k in ds_gm.dims})
grid = Grid(ds_gm, periodic=False)

### take a region for every dimensions except z
ds_gr = ds_gr.isel({d:region[d[0]] for d in ds_gr.dims if d[0] in region})
ds_gm = ds_gm.isel({d:region[d[0]] for d in ds_gm.dims if d[0] in region})

### load stratif, align chunks and merge with mean grid
for p in [strat_path, presm_path]:
    ds_st = xr.open_zarr(p).reset_coords(drop=True)
    ds_st = ds_st.chunk({k:chunks[k[0]] for k in ds_st.dims if k[0] in chunks})
    ds_gm = ds_gm.merge(ds_st, join="inner")

### load vertical modes
dm = xr.open_zarr(vmod_path).unify_chunks()
dm = dm.chunk({k:chunks[k[0]] for k in dm.dims if k[0] in chunks})
dm = dm.isel(x_c=region["x"], y_c=region["y"])
logging.info("opened stratif and vertical modes")

### open temperature, salinity and ssh
les_var = ["vosaline", "votemper"]
v = les_var[0]
tmes = time.time()
ds = ut.open_one_var(ut.get_eNATL_path(v, i_days), chunks=chunks, varname=v)
for v in les_var[1:]:
    ds = ds.merge(ut.open_one_var(ut.get_eNATL_path(v,i_days), chunks=chunks, varname=v))
# Load SSH 
v = "sossheig"
if ssh_path is None: # reading from netCDF
    dssh = ut.open_one_var(ut.get_eNATL_path(v, i_days), chunks=chunks, varname=v)
else: # reading from zarr
    dssh = ut.open_one_var(ssh_path, chunks="auto", varname=v, engine="zarr")
    dssh = dssh.chunk({d:chunks_tg[d] for d in dssh.dims})
ds = ds.merge(dssh.reset_coords(drop=True), join="inner")
ds = ds.isel({d:region[d[0]] for d in ds.dims if d[0] in region})
logging.info("opened T, S and SSH data -- ellapsed time {:.1f} s".format(time.time()-tmes))

###  ---------------------------------------------------------------------  ###
###  -----------------  Start Computation  -------------------------------  ###
###  ---------------------------------------------------------------------  ###

### Initialization of global domain objects (not computed)
vmods = Vmodes(ds_gm, grid, nmodes=dm.nmodes, free_surf=dm.free_surf, persist=False, 
                chunks = {k:v[0] for k,v in dm.chunks.items()}
              )
pres = get_pres(ds.isel(t=slice(0,24)), ds_gr, ds_gm, grid, with_persist=False)
pmod = vmods.project_puv(pres)
logging.info("created pmod object, total size {:.1f} GB".format(pmod.nbytes/1e9))
# cleaning pmod
if "llon_cc" not in pmod.coords:
    pmod = pmod.assign_coords({v:dm[v] for v in ["llon_cc", "llat_cc"]})
pmod = pmod.reset_coords([c for c in pmod.coords 
                          if c not in ["llon_cc", "llat_cc"] and c not in pmod.dims], 
                          drop=True
                        )
for c in pmod.coords:
    pmod[c].encoding.pop("chunks", None)
pmod = pmod.to_dataset(name="pres")

### create zarr archives
if restart is None or restart is False:
    restart = 0
if restart:
    logging.info("continuing previous job at iy={}, will append in zarr archives".format(restart))
else:
    for da in sim_dates: 
        tmes = time.time()
        pmod.to_zarr(out_dir/out_file.format(da), compute=False, mode="w", consolidated=True)
        logging.info("creating zarr took {:.2f} s".format(time.time()-tmes))
    logging.info("created zarr archives for storing modal projection")

### loop over y and time (computation happens here)
Nt = pmod.t.size
ind_y = np.r_[np.arange(0, int(ds.y_c.size), sk_y), int(ds.y_c.size)]
logging.info("starting loop over y and time, \n computing {0} segments of size {1} in time and {2} segments of size {3} in y".format(Nt//nk_t,nk_t,len(ind_y)-1, sk_y)
            )
region = {d:slice(0,None) for d in pmod.dims}
chk_x = max(1, chunks["x"]) # this is bypassing chunking in x if it has size -1 (for to_zarr with region)

tmp = "{}-{}".format(i_days[0],i_days[-1])
with performance_report(filename="perf-report_proj-pres_{}.html".format(tmp)):
    for jj in range(restart,len(ind_y)-1):
        logging.info("starting y segment # {}".format(jj))
        sliy = slice(ind_y[jj], ind_y[jj+1])
        region["y_c"] = sliy
        sds_gm = ds_gm.isel({d:sliy for d in ds_gm.dims if d.startswith("y_")})
        sds_gr = ds_gr.isel({d:sliy for d in ds_gr.dims if d.startswith("y_")})
        sds = ds.isel({d:sliy for d in ds.dims if d.startswith("y_")})
        sdm = dm.isel({d:sliy for d in dm.dims if d.startswith("y_")})
    
        ### here select subdomain
        if drop_land_x:
            index = sds_gm.x_c.where(sds_gm.tmaskutil.max("y_c"), drop=True)
            slix = slice((int(index.x_c[0]-1)//chk_x)*chk_x, int(index.x_c[-1]))
            sds_gr = sds_gr.isel({d:slix for d in sds_gr.dims if d.startswith("x_")})
            sds_gm = sds_gm.isel({d:slix for d in sds_gm.dims if d.startswith("x_")})
            sds = sds.isel({d:slix for d in sds.dims if d.startswith("x_")})
            sdm = sdm.isel({d:slix for d in sdm.dims if d.startswith("x_")})
            logging.info("x subdomain from {} to {}, size {}".format(
                          slix.start, slix.stop, slix.stop-slix.start)
                        )
        else:
            slix = slice(0,None)
        region["x_c"] = slix
        sds_gr = sds_gr.persist()
        logging.info("persisted sds_gr")
        # some fields of the mean grid
        for c in ["depth_c_3d", "hbot", "sossheig", "pres"]:
            sds_gm[c] = sds_gm[c].persist()
        logging.info("persisted sds_gm")
        # some fields of the dm object:
        sdm = sdm.assign_coords({c:sds_gm[c].persist() for c in ["tmask","e3t"]})
        vmods.ds = sdm
        for c in ["norm","phi"]:
            vmods.ds[c] = vmods.ds[c].persist()
        logging.info("persisted vmods")
        # compute pres and project on vertical modes + clean pmod
        pres = get_pres(sds, sds_gr, sds_gm, grid, with_persist=False) - sds_gm.pres
        pmod = vmods.project_puv(pres)
        logging.info("created pmod")
        if "llon_cc" not in pmod.coords:
            pmod = pmod.assign_coords({v:dm[v] for v in ["llon_cc", "llat_cc"]})
        pmod = pmod.reset_coords([c for c in pmod.coords 
                                  if c not in ["llon_cc", "llat_cc"] and c not in pmod.dims], 
                                  drop=True
                                )
        for c in pmod.coords:
            pmod[c].encoding.pop("chunks", None)
        pmod = pmod.to_dataset(name="pres")
    
        ### loop in time for computing and storing
        logging.info("starting time loop")
        tmea = time.time()
        for it in range(Nt//nk_t):
            tmes = time.time()
            slit = slice(it*nk_t,(it+1)*nk_t)
            region["t"] = slit
            da = sim_dates[it//24]
            pmod.isel(t=slit).to_zarr(out_dir/out_file.format(da), mode="a", compute=True, region=region)
            logging.info("iteration y={}, t={}, ellapsed time {:.1f} s".format(jj, it, time.time()-tmes))
        logging.info("chunk y={} took {:.2f} min (mean /x-point/t: {:.2f} ms)".format(
                jj, (time.time()-tmea)/60., (time.time()-tmea)/(24*(slix.stop-slix.start))*1e3)
               )
        client.restart()

logging.info("  ----  FINISHED  ----  ")

### create log file with some information
for i,da in enumerate(sim_dates):
    with open(log_dir/log_file.format(da), "w") as fp:
        fp.write("JOB ID: {}\n".format(os.getenv("SLURM_JOBID")))
        fp.write("python script: {}\n".format(sys.argv[0]))
        fp.write("i_day {}\n".format(i_days[i]))
        fp.write("nk_t {}, sk_y {}, chunks {}\n".format(nk_t, sk_y, chunks))

