#!/usr/bin/env python
# coding: utf-8
""" compute_vmodes.py [date mean]
Compute vertical modes based on mean stratification

Strategy: process sequentially over y-segment, droping land points (columns) for each segment. This reduces the memory usage and is well adapted for a single-node dask cluster, but doesn't work if one wants to process large segments using a larger cluster. The efficiency of this script could be improved by having a mean not to load land points in memory within the distributed loop, e.g. through a map_blocks or something like that

#### Timing and memory consumption:
* not droping land: up to 90 GB
* droping lands: depends on the segment, but typically max at 55 GB
* first 2 y-segments of size 200: 50 & 60 minutes (dropping lands), 50 & 76 minutes (not dropping lands)

This script is for processing eNATL60 data on occigen

N. Lahaye, June-October 2021
"""

from pathlib import Path
import os, time, datetime, sys
import logging

import xarray as xr
from xgcm import Grid
from itidenatl.nemodez import Vmodes

### Intialize dask (https://mpi.dask.org/en/latest/batch.html)
from distributed import Client
scratch = Path(os.getenv("SCRATCHDIR"))
# method 1: dask-based script
if True:
    from dask_mpi import initialize
    initialize(nthreads=4, interface="ib0", memory_limit=6e9, 
            dashboard=False, local_directory=scratch)
    client=Client()
else:
    client = Client(scheduler_file="scheduler.json")

logging.basicConfig(format='[{}] %(asctime)s -- %(message)s'.format(sys.argv[0]), 
        level=logging.INFO, stream=sys.stdout)
logging.info("Cluster should be connected -- dashboard at {}".format(client.dashboard_link))
##########################  - - - PARAMETERS  - - -  ############################
### define paths
scratch = Path(os.getenv("SCRATCHDIR"))
works = Path("/work/CT1/ige2071/SHARED")
grid_path = scratch #Path("/store/CT1/hmg2840/lbrodeau/eNATL60/eNATL60-I/")
mean_path = scratch #works/"mean"

avg_type = "30d" # "30d" or "global"
avg_date = sys.argv[1] if len(sys.argv)>1 else "20090630" # will be ignored if avg_type is "global"
app = "_"+avg_date if avg_type == "30d" else ""
zgrid_fname = f"eNATL60_{avg_type}-mean_z-grid{app}.zarr" 
strat_fname = f"eNATL60_{avg_type}-mean_bvf{app}.zarr"

zgrid_file = scratch/zgrid_fname
strat_file = scratch/strat_fname
out_file = works/f"vmodes/eNATL60_{avg_type}-mean_vmodes{app}.zarr"
tmp_file = scratch/f"prov_vmodes{app}.zarr"

### processing parameters
nmodes = 10
out_chk = {"mode":1, "x_c":-1, "z_c": 30, "z_l":30}
wrk_chk = {"x_c":200}
nseg_y = 200 # y-segment size: choose it a multiple or a divider of chunk size
drop_land = True
restart = 7 #False # False or jy

#############################  - - -  PROCESSING  - - -  ########################
dg = xr.open_zarr(zgrid_file).astype("float32")
ds = xr.open_zarr(strat_file).astype("float32")

coord_copy = ["e3t", "e3w", "hbot", "tmask", "depth_c"] # WARNING remove depth_c_3d
ds = ds.assign_coords({co:dg[co] for co in coord_copy}).chunk(wrk_chk).unify_chunks()
# remove chunks from encoding, for zarr storage
for var in ds.coords.values(): 
    var.encoding.pop("chunks", None)
chunks = {k:v[0] for k,v in ds.chunks.items()}
ds

### create vmods object and zarr archive (delayed mode) 
logging.info("creating Vmodes object and zarr store")
vmods = Vmodes(ds, Grid(ds, periodic=False), nmodes=nmodes, free_surf=True, \
                persist=False, chunks=out_chk)
put_attrs = {"from_files":[str(zgrid_file), str(strat_file)], 
             "simulation": "eNATL60", "processing notebook":"compute_vmodes.ipynb",
             "date_processed":datetime.datetime.today().isoformat(timespec="minutes"),
             "process_params_wrk_chk": [(k,v) for k,v in wrk_chk.items()]
             }
put_attrs.update({f"process_params_{k}":eval(k) for k in ["nseg_y", "drop_land"]})
vmods.ds.attrs = put_attrs
if not restart:
    vmods.store(out_file, coords=False, mode="w", compute=False, consolidated=True)

### Compute and store, looping over y-segments
Ny = ds.y_c.size
region = {"z_c": slice(0,None), "z_l":slice(0,None), "mode":slice(0,None)}

def get_subds(ds):
    """ wrapper to get rid of land points (columns).
    Warning: this works only if x_c increment is 1 """
    lnd_pts = (ds.tmaskutil==0).sum().values
    logging.info("number of land points: {} ({:.1f}%)".format(lnd_pts, 
        lnd_pts*100/ds.tmaskutil.size))
    index = ds.tmaskutil.max("y_c")
    index = index.where(index, drop=True).x_c - index.x_c[0] #
    index = slice(int(index[0]), int(index[-1])+1)
    sds = ds.isel(x_c=index)
    lnd_pts = (sds.tmaskutil==0).sum().values
    logging.info("after selection: {} ({:.1f}%)".format(lnd_pts, lnd_pts*100/sds.tmaskutil.size))
    return sds, index

### this is the loop
logging.info("now starting loop, processing {} y-segments of size {}".format(Ny//nseg_y, nseg_y))
jy_0 = restart*nseg_y if restart else 0
for jy in range(jy_0, Ny, nseg_y):
    tmes = time.time()
    sliy = slice(jy, min(jy+nseg_y, Ny))
    if drop_land:
        sds, slix = get_subds(ds.isel(y_c=sliy))
    else:
        sds = ds.isel(y_c=sliy)
        slix = slice(0, None)
    grid = Grid(sds, periodic=False)
    region.update({"y_c":sliy, "x_c":slix})
    vmods = Vmodes(sds, grid, modes=nmodes, free_surf=True, persist=False)#, chunks=out_chk)
    vmods.ds = vmods.ds.where(vmods.ds.tmaskutil)
    vmods.ds.reset_coords(drop=True).to_zarr(tmp_file, mode="w", compute=True)
    #vmods.store(tmp_file, coords=False, mode="w", compute=True)
    logging.info("computed and stored // now rechunking")
    vmods.ds = xr.open_zarr(tmp_file).chunk(out_chk).unify_chunks()
    vmods.store(out_file, coords=False, mode="a", compute=True, region=region)
    logging.info("segment {0}-{1} done, x-size {2}, {3:.1f} min".format(jy, jy+nseg_y, 
                                                    sds.x_c.size, (time.time()-tmes)/60)
         )

