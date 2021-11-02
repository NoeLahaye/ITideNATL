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

### define chunking
chunks = {"z":10, "y":100, "x":-1}
sk_y = 200

###  ----------------------------------------------------------------------  ###
###  ---------------------- End of user-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

# replicate chunks in terms of target dimensions
chunks_tg = {di:chunks[di[0]] for di in ["z_c", "z_l"]}
for di in ("x", "y"):
    chunks_tg.update({di+"_"+su:chunks[di] for su in ["c","r"]})
logging.info("finished setting up parameters, starting to load data")

### load mean grid (associated with year-mean ssh)
if grid_path.suffix == ".zarr": # reading from zarr
    if True:
        ds_gm = xr.open_zarr(grid_path)
        if True: # apply chunking. WARNING: this seems to be the cause of work halting
            ds_gm = ds_gm.chunk({k:v for k,v in chunks_tg.items() if k in ds_gm.dims})
    else:
        ds_gm = xr.open_zarr(grid_path, chunks={k:v for k,v in chunks_tg.items() if k not in ["t"]})
else:
    ds_gm = ut.open_one_coord(grid_path, "e3t", chunks=chunks)
    ds_gm = ds_gm.merge(ut.open_one_coord(grid_path, "tmask", chunks=chunks))
    
print("chunking:", ds_gm.chunks)
###  ---------------------------------------------------------------------  ###
###  -----------------  Start Computation  -------------------------------  ###
###  ---------------------------------------------------------------------  ###

### loop over y and time
ind_y = np.r_[np.arange(0, int(ds_gm.y_c.size), sk_y), int(ds_gm.y_c.size)]
logging.info("starting loop over y , \n computing {0} segments of size {1} in y".format(len(ind_y)-1, sk_y)
            )
for jj in range(0,len(ind_y)-1):
    tmes = time.time()
    logging.info("starting y segment # {}".format(jj))
    sliy = slice(ind_y[jj], ind_y[jj+1])
    sds_gm = ds_gm.isel({d:sliy for d in ds_gm.dims if d.startswith("y_")})
    ## persist: is this a problem? 
    for c in ["e3t", "tmask"]:
        sds_gm[c] = sds_gm[c].persist()

    pmod = sds_gm.e3t.where(sds_gm.tmask).sum("z_c").mean(dim=("x_c","y_c"))

    print(pmod.values)
    logging.info("iteration y={}, ellapsed time {:.1f} s".format(jj, time.time()-tmes))
    #client.restart()

logging.info("  ----  FINISHED  ----  ")

