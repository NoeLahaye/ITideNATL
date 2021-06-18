#!/usr/bin/env python
# coding: utf-8

""" remove persisting of mods. Only use e3t and tmask from mean grid. Read only ssh. 
Treat whole x domain.
Create object within a loop on y and compute it slicing along t in a loop
Result: passes with 6 workers (4 threads & 21 GB each), passes with 12 workers
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
import itidenatl.utils as ut

### Intialize dask (https://mpi.dask.org/en/latest/batch.html)
from distributed import Client#, performance_report
scratch = Path(os.getenv("SCRATCHDIR"))
# method 1: dask-based script
if True:
    from dask_mpi import initialize
    #initialize(nthreads=2, interface="ib0", memory_limit=3e9, dashboard=False)
    initialize(nthreads=4, interface="ib0", memory_limit=21e9, 
            dashboard=True, local_directory=scratch)
    client=Client()
else:
    client = Client(scheduler_file="scheduler.json")
logging.info("Cluster should be connected")

### define paths
grid_path = Path("/work/CT1/ige2071/nlahaye/eNATL60_rest_grid_cz10.zarr") # scratch/"mesh_mask_eNATL60_3.6.nc" #
data_path = Path("/work/CT1/hmg2840/lbrodeau/eNATL60")

### define chunking
chunks = {"t":1, "z":10, "y":100, "x":-1}
i_day = 11
sk_y = 200
nk_t = 1

###  ----------------------------------------------------------------------  ###
###  ---------------------- End of user-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

# replicate chunks in terms of target dimensions
chunks_tg = {di:chunks[di[0]] for di in ["z_c", "z_l", "t"]}
for di in ("x", "y"):
    chunks_tg.update({di+"_"+su:chunks[di] for su in ["c","r"]})
logging.info("finished setting up parameters, starting to load data")

### load mean grid (associated with year-mean ssh)
if grid_path.suffix == ".zarr": # reading from zarr
    if False:
        ds_gm = xr.open_zarr(grid_path)
        if False: # apply chunking. WARNING: this seems to be the cause of work halting
            ds_gm = ds_gm.chunk({k:v for k,v in chunks_tg.items() if k in ds_gm.dims})
    else:
        ds_gm = xr.open_zarr(grid_path, chunks={k:v for k,v in chunks_tg.items() if k not in ["t"]})
else:
    ds_gm = ut.open_one_coord(grid_path, "e3t", chunks=chunks)
    ds_gm = ds_gm.merge(ut.open_one_coord(grid_path, "tmask", chunks=chunks))
    
### open temperature, salinity and ssh
# Load SSH 
v = "sossheig"
ds = ut.open_one_var(ut.get_eNATL_path(v, i_day), chunks=chunks, varname=v)
logging.info("opened SSH data")

###  ---------------------------------------------------------------------  ###
###  -----------------  Start Computation  -------------------------------  ###
###  ---------------------------------------------------------------------  ###

### loop over y and time
Nt = ds.t.size
ind_y = np.r_[np.arange(0, int(ds.y_c.size), sk_y), int(ds.y_c.size)]
logging.info("starting loop over y and time, \n computing {0} segments of size {1} in time and {2} segments of size {3} in y".format(Nt//nk_t,nk_t,len(ind_y)-1, sk_y)
            )
region = {d:slice(0,None) for d in ds.dims if not d.startswith("z")}
region["mode"] = slice(0,None)

for jj in range(0,len(ind_y)-1):
    logging.info("starting y segment # {}".format(jj))
    sliy = slice(ind_y[jj], ind_y[jj+1])
    region["y_c"] = sliy
    sds = ds.isel({d:sliy for d in ds.dims if d.startswith("y_")})
    sds_gm = ds_gm.isel({d:sliy for d in ds_gm.dims if d.startswith("y_")})
    # some fields of the mean grid
    for c in ["e3t", "tmask"]:
        sds_gm[c] = sds_gm[c].persist()

    pmod = ( sds.sossheig * sds_gm.e3t
            ).where(sds_gm.tmask).sum("z_c").mean(dim=("x_c","y_c"))

    ### loop in time for computing and storing
    logging.info("starting time loop")
    tmea = time.time()
    for it in range(Nt//nk_t):
        tmes = time.time()
        slit = slice(it*nk_t,(it+1)*nk_t)
        region["t"] = slit
        logging.info("result of computation")
        print(pmod.isel(t=slit).values)
        logging.info("iteration y={}, t={}, ellapsed time {:.1f} s".format(jj, it, time.time()-tmes))
    logging.info("chunk y={} took {:.2f} min (mean /x-point/t: {:.2f} ms)".format(
            jj, (time.time()-tmea)/60., (time.time()-tmea)/(24*int(ds.x_c.size))*1e3)
           )
    client.restart()

logging.info("  ----  FINISHED  ----  ")

