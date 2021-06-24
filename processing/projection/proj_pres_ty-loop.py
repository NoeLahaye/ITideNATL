#!/usr/bin/env python
# coding: utf-8

""" proj_pres_ty-loop.py
script for computing the pressure anomaly and projecting it on vertical modes
Dedicated to the analysis of eNATL60. All "static" fields (grid, vertical modes, mean pressure...) must have been computed previously and put in a zarr archive

Strategy: to limit memory usage and gain computational time, this script makes a loop over y segments and iterate over time within this loop. 
This allows to operate over subdomains olong x coordinate by selecting non-land bands for each y segment, and to persist smaller portions of the static fields once for each time.

It is nonetheless theoretically possible to treat the whole space domain, and/or the whol time domain. If processing several days, distinct input files are read and output files will be created. It is therefore mandatory to keep a loop in time over these files (e.g. nk_t<=24)

N. Lahaye, June 2021
"""

from pathlib import Path
import os, time, sys
import logging
logging.basicConfig(format='[{}] %(asctime)s -- %(message)s'.format(sys.argv[0]), 
        level=logging.INFO, stream=sys.stdout)

import numpy as np

import xarray as xr
from xgcm import Grid

from proj_utils import proj_pres, get_pres_one_dg as get_pres
import itidenatl.utils as ut

### Intialize dask (https://mpi.dask.org/en/latest/batch.html)
from distributed import Client, performance_report
scratch = Path(os.getenv("SCRATCHDIR"))
# method 1: dask-based script
if True:
    from dask_mpi import initialize
    #initialize(nthreads=4, interface="ib0", memory_limit=21e9, 
    #initialize(nthreads=2, interface="ib0", memory_limit=11e9, 
    initialize(nthreads=3, interface="ib0", memory_limit=17e9, 
            dashboard=True, local_directory=scratch)
    client=Client()
else:
    client = Client(scheduler_file="scheduler.json")
logging.info("Cluster should be connected -- dashboard at {}".format(client.dashboard_link))

###  ----------------------------------------------------------------------  ###
###  ---------------------- User-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

### define chunking and computational subdomains (y, t)
chunks = {"t":1, "z":10, "y":100, "x":-1}
nk_t = 4 # process nk_t instants at a time (must be a divider of 24)
sk_y = 200 # process y-subdomains of size sk_y at a time (choose it a multiple of chunk size)
assert 24%nk_t == 0 and sk_y%chunks["y"] == 0
restart = 0 # continue previously stopped job (y segments). 0, False or None to start from beginning

### read time ("day of simu in data") from sys.argv, or use here-defined value
if len(sys.argv)>1: #N.B.: we can process several days
    i_days = [int(k) for k in sys.argv[1:]]
else:
    i_days = [0] # default value (list)

drop_land_x = True  ### wether to drop land points in x (must be land for every y)
### In theory, this script could process an arbitrary subdomain, but I never tested it (use notebook instead)
region = {"t":slice(0,None), "x":slice(0,None), "y":slice(0,None)}

### define paths
#workdir = Path("/work/CT1/ige2071/nlahaye")
worksha = Path("/work/CT1/ige2071/SHARED")

data_path = Path("/work/CT1/hmg2840/lbrodeau/eNATL60")
grid_mode_path = scratch/"eNATL60_grid_vmodes_proj_pres.zarr" 
out_dir = worksha/"modal_proj"
out_file = "modamp_pres_global_{}.zarr" #"modamp_subdom_{}.zarr" #.format(date)
log_dir = Path(os.getenv("HOME"))/"working_on/processing/log_proj_pres"
log_file = "proj_pres_{}.log" #.format(i_day)

###  ----------------------------------------------------------------------  ###
###  ---------------------- End of user-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

### get date (day) and check for existing log files
sim_dates = ut.get_date_from_iday(i_days)
for da in sim_dates:
    if (log_dir/log_file.format(da)).exists():
        raise ValueError("{} already processed? found corresponding log file".format(da))
    logging.info("will process date {}".format(da))

if region["x"].start > 0 and drop_land_x:
    logging.info("reverting drop_land_x to False because subdomain is not implemented")
    drop_land_x = False
    
# replicate chunks in terms of target dimensions
chunks_tg = {di:chunks[di[0]] for di in ["z_c", "z_l", "t"]}
for di in ("x", "y"):
    chunks_tg.update({di+"_"+su:chunks[di] for su in ["c","r"]})
logging.info("finished setting up parameters, starting to load data")

### Load static fields (grid, vmodes, pres...) and select region
ds_g = xr.open_zarr(grid_mode_path)
ds_g = ds_g.chunk({k:v for k,v in chunks_tg.items() if k in ds_g.dims})
ds_g = ds_g.isel({d:region[d[0]] for d in ds_g.dims if d[0] in region})
grid = Grid(ds_g, periodic=False)
logging.info("opened static fields, reading from {}".format(grid_mode_path.name))

### open temperature, salinity and ssh
les_var = ["vosaline", "votemper"]
v = les_var[0]
tmes = time.time()
ds = ut.open_one_var(ut.get_eNATL_path(v, i_days), chunks=chunks, varname=v)
for v in les_var[1:]:
    ds = ds.merge(ut.open_one_var(ut.get_eNATL_path(v,i_days), chunks=chunks, varname=v))
# Load SSH 
v = "sossheig"
dssh = ut.open_one_var(ut.get_eNATL_path(v, i_days), chunks=chunks, varname=v)
ds = ds.merge(dssh.reset_coords(drop=True), join="inner")
ds = ds.isel({d:region[d[0]] for d in ds.dims if d[0] in region})
logging.info("opened T, S and SSH data -- ellapsed time {:.1f} s".format(time.time()-tmes))

###  ---------------------------------------------------------------------  ###
###  -----------------  Start Computation  -------------------------------  ###
###  ---------------------------------------------------------------------  ###

pres = get_pres(ds.isel(t=slice(0,24)), ds_g, grid, with_persist=False)
pmod = proj_pres(pres, ds_g)
logging.info("created pmod object, total size {:.1f} GB".format(pmod.nbytes/1e9))

### create zarr archives
if restart is None or restart is False:
    restart = 0
if restart:
    logging.info("continuing previous job at iy={}, will append in zarr archives".format(restart))
else:
    for da in sim_dates: 
        tmes = time.time()
        pmod.to_zarr(out_dir/out_file.format(da), compute=False, mode="w", consolidated=True, safe_chunks=False)
        logging.info("creating zarr took {:.2f} s".format(time.time()-tmes))
    logging.info("created zarr archives for storing modal projection")

### loop over y and time (computation happens here)
Nt = pmod.t.size
ind_y = np.r_[np.arange(0, int(ds.y_c.size), sk_y), int(ds.y_c.size)]
logging.info("starting loop over y and time, \n computing {0} segments of size {1} in time and {2} segments of size {3} in y".format(Nt//nk_t,nk_t,len(ind_y)-1, sk_y)
            )
region = {d:slice(0,None) for d in pmod.dims}
chk_x = max(1, chunks["x"]) # this is bypassing chunking in x if it has size -1 (for to_zarr with region)

tmp = "{}-{}".format(i_days[0],i_days[-1]) if len(i_days)>1 else str(i_days[0])
with performance_report(filename="perf-report_proj-pres_{}.html".format(tmp)):
    for jj in range(restart,len(ind_y)-1):
        logging.info("starting y segment # {}".format(jj))
        sliy = slice(ind_y[jj], ind_y[jj+1])
        region["y_c"] = sliy
        sds_g = ds_g.isel({d:sliy for d in ds_g.dims if d.startswith("y_")})
        sds = ds.isel({d:sliy for d in ds.dims if d.startswith("y_")})
    
        ### here select subdomain
        if drop_land_x:
            index = sds_g.x_c.where(sds_g.tmaskutil.max("y_c"), drop=True)
            slix = slice((int(index.x_c[0]-1)//chk_x)*chk_x, int(index.x_c[-1]))
            sds_g = sds_g.isel({d:slix for d in sds_g.dims if d.startswith("x_")})
            sds = sds.isel({d:slix for d in sds.dims if d.startswith("x_")})
            logging.info("x subdomain from {} to {}, size {}".format(
                          slix.start, slix.stop, slix.stop-slix.start)
                        )
        else:
            slix = slice(0,None)
        region["x_c"] = slix
        sds_g = sds_g.persist()
        logging.info("persisted sds_g")
        # compute pres and project on vertical modes + clean pmod
        pres = get_pres(sds, sds_g, grid, with_persist=False)
        pmod = proj_pres(pres, sds_g)
    
        ### loop in time for computing and storing
        logging.info("starting time loop")
        tmea = time.time()
        for it in range(Nt//nk_t):
            tmes = time.time()
            slit = slice(it*nk_t,(it+1)*nk_t)
            region["t"] = slit
            da = sim_dates[it//24]
            pmod.isel(t=slit).to_zarr(out_dir/out_file.format(da), mode="a", compute=True, region=region, safe_chunks=False)
            logging.info("iteration y={}, t={}, ellapsed time {:.1f} s".format(jj, it, time.time()-tmes))
        logging.info("chunk y={} took {:.2f} min (mean /x-point/t: {:.2f} ms)".format(
                jj, (time.time()-tmea)/60., (time.time()-tmea)/(24*(slix.stop-slix.start))*1e3)
               )
        #del sds_g, pres, pmod
        client.restart()

logging.info("  ----  FINISHED  ----  ")

### create log file with some information
for i,da in enumerate(sim_dates):
    with open(log_dir/log_file.format(da), "w") as fp:
        fp.write("JOB ID: {}\n".format(os.getenv("SLURM_JOBID")))
        fp.write("python script: {}\n".format(sys.argv[0]))
        fp.write("i_day {}\n".format(i_days[i]))
        fp.write("nk_t {}, sk_y {}, chunks {}\n".format(nk_t, sk_y, chunks))

