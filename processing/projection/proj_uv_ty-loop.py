#!/usr/bin/env python
# coding: utf-8

""" python proj_uv_ty-loop.py [u,v] [i_days]
script for projecting the horizontal velocity anomaly on vertical modes
Dedicated to the analysis of eNATL60. All "static" fields (grid, vertical modes, mean pressure...) must have been computed previously and put in a zarr archive

Strategy: to limit memory usage and gain computational time, this script makes a loop over y segments and iterate over time within this loop. 
This allows to operate over subdomains olong x coordinate by selecting non-land bands for each y segment, and to persist smaller portions of the static fields once for each time.
A different chunking is employed when storing the results, which must be chosen with care.

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

from proj_utils import proj_puv, get_uv_mean_grid
from itidenatl.tools import files as uf
from itidenatl.tools import dataio as io

### Intialize dask (https://mpi.dask.org/en/latest/batch.html)
from distributed import Client, performance_report
scratch = Path(os.getenv("SCRATCHDIR"))
# method 1: dask-based script
if True:
    from dask_mpi import initialize
    #initialize(nthreads=4, interface="ib0", memory_limit=21e9, 
    #initialize(nthreads=3, interface="ib0", memory_limit=17e9, 
    initialize(nthreads=2, interface="ib0", memory_limit=5e9, 
            dashboard=False, local_directory=scratch)
    client=Client()
else:
    client = Client(scheduler_file="scheduler.json")
logging.info("Cluster should be connected -- dashboard at {}".format(client.dashboard_link))

###  ----------------------------------------------------------------------  ###
###  ---------------------- User-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

### define chunking and computational subdomains (y, t)
chunks = {"t":1, "z":10, "y":100, "x":-1}
chk_store = {"t":-1, "mode":1, "y":400, "x":-1} 
nk_t = 1 # process nk_t instants at a time (must be a divider of nt_f)
sk_y = 100 # process y-subdomains of size sk_y at a time (choose it a multiple of chunk size)
if len(sys.argv)>1:
    var = sys.argv[1]
else:
    var = "u" # choose "u" or "v"
restart = False # continue previously stopped job (y segments). False or None starts from beginning creating a new zarr store 

### read time ("day of simu in data") from sys.argv, or use here-defined value
if len(sys.argv)>2: #N.B.: we can process several days
    i_days = [int(k) for k in sys.argv[2:]]
else:
    i_days = [0] # default value (list)

drop_land_x = True  ### wether to drop land points in x (must be land for every y)
### In theory, this script could process an arbitrary subdomain, but I never tested it (use notebook instead)
region = {"t":slice(0,None), "x":slice(0,None), "y":slice(0,None)}

### define paths
#workdir = Path("/work/CT1/ige2071/nlahaye")
worksha = Path("/work/CT1/ige2071/SHARED")

#data_path = Path("/work/CT1/hmg2840/lbrodeau/eNATL60")
grid_uv_path = worksha/"vmodes/phi_{}_10.zarr".format(var)
grid_mode_path = scratch/"eNATL60_grid_vmodes_proj_pres.zarr" 
out_dir = worksha/"modal_proj/modamp_{}".format(var)
out_file = "modamp_{}_global_{}.zarr".format(var, "{}") #.format(date)
log_dir = Path(os.getenv("HOME"))/"working_on/processing/log_proj_uv"
log_file = "proj_{}_{}.log".format(var, "{}") #.format(i_day)

###  ----------------------------------------------------------------------  ###
###  ---------------------- End of user-defined parameters ----------------  ###
###  ----------------------------------------------------------------------  ###

nt_f = 24 # time instant per file
assert nt_f%nk_t == 0 and sk_y%chunks["y"] == 0
### get date (day) and check for existing log files
sim_dates = uf.get_date_from_iday(i_days)
for da in sim_dates:
    if (log_dir/log_file.format(da)).exists():
        raise ValueError("{} already processed? found corresponding log file".format(da))
        os._exit()
    logging.info("will process date {} (i_day {}), variable {}".format(da,i_days,var))

if region["x"].start > 0 and drop_land_x:
    logging.info("reverting drop_land_x to False because subdomain is not implemented")
    drop_land_x = False
    
# replicate chunks in terms of target dimensions
chunks_tg = {di:chunks[di[0]] for di in ["z_c", "z_l", "t"]}
for di in ("x", "y"):
    chunks_tg.update({di+"_"+su:chunks[di] for su in ["c","r"]})
logging.info("finished setting up parameters, starting to load data")

### Load static fields (grid, vmodes, pres...) and select region
ds_g = xr.open_zarr(grid_uv_path) 
ds_g = ds_g.chunk({k:v for k,v in chunks_tg.items() if k in ds_g.dims})
ds_g = ds_g.isel({d:region[d[0]] for d in ds_g.dims if d[0] in region})
ds_x = xr.open_zarr(grid_mode_path) # need this to get some info on the z_l grid
logging.info("opened static fields, reading from {}, {}".format(grid_uv_path.name, grid_mode_path.name))

### open velocity and ssh
uv_name = {"u":"vozocrtx", "v":"vomecrty"}[var]
dim_itp = "xy"["uv".index(var)]
les_var = [uv_name, "sossheig"]
v = les_var[0]
tmes = time.time()
ds = io.open_one_var(uf.get_eNATL_path(v, i_days), chunks=chunks, varname=v)
for v in les_var[1:]:
    ds = ds.merge(io.open_one_var(uf.get_eNATL_path(v,i_days), chunks=chunks, varname=v)\
                    .reset_coords(drop=True))
ds = ds.isel({d:region[d[0]] for d in ds.dims if d[0] in region})
logging.info("opened velocity and SSH data -- ellapsed time {:.1f} s".format(time.time()-tmes))

# dataset for grid only
ds_xb = ds.get(list(ds.dims.keys())).merge(ds_x.get(list(ds_x.dims.keys()))).reset_coords(drop=True)
grid = Grid(ds_xb)
ds_x.close()

# interp ssh
ds["sossheig"] = grid.interp(ds["sossheig"], dim_itp.upper(), boundary="extend")
ds["sossheig"] = ds["sossheig"].chunk({dim_itp+"_r":chunks[dim_itp]})

###  ---------------------------------------------------------------------  ###
###  -----------------  Start Computation  -------------------------------  ###
###  ---------------------------------------------------------------------  ###

uvec = get_uv_mean_grid(ds, grid, ds_g)
amod = proj_puv(uvec, ds_g)
# store chunks in terms of target dimensions
chk_store = {d:chk_store[next(k for k in chk_store.keys() if d.startswith(k))] for d in amod.dims}
amod = amod.chunk(chk_store)
logging.info("created amod object, total size {:.1f} GB".format(amod.nbytes/1e9))

### create zarr archives
if not (restart is None or restart is False):
    logging.info("continuing previous job at iy={}, will append in zarr archives".format(restart))
else:
    for i,da in enumerate(sim_dates): 
        from_files = ", ".join([str(grid_mode_path), str(grid_uv_path),
                                str(uf.get_eNATL_path(uv_name, i_days[i])), 
                               str(uf.get_eNATL_path("sossheig", i_days[i]))
                               ])
        amod.attrs = {"from_files": from_files, "generating_script": sys.argv[0], 
                      "by": "N. Lahaye (noe.lahaye@inria.fr)", 
                      "date generated": logging.time.asctime(), 
                      "simulation": "eNATL60 (with tides)",
                      "day of simulation": da, "i_day": i_days[i]
                     }
        tmes = time.time()
        slit = slice(i*nt_f,(i+1)*nt_f)
        amod.isel(t=slit).to_zarr(out_dir/out_file.format(da), compute=False, 
                                 mode="w", consolidated=True)
        logging.info("creating zarr took {:.2f} s".format(time.time()-tmes))
    logging.info("created zarr archives for storing modal projection")
    restart = 0

### loop over y and time (computation happens here)
Nt = amod.t.size
dim_h = {k:next(d for d in amod.dims if d.startswith(k+"_")) for k in ["x", "y"]}
ind_y = np.r_[ np.arange(0, ds.dims[dim_h["y"]], sk_y), ds.dims[dim_h["y"]] ]
logging.info("starting loop over y and time, \n computing {0} segments of size {1} \
in time and {2} segments of size {3} in y".format(Nt//nk_t,nk_t,len(ind_y)-1, sk_y)
            )
region = {d:slice(0,None) for d in amod.dims}
chk_x = max(1, chunks["x"]) # this is bypassing chunking in x if it has size -1 (for to_zarr with region)

tmp = "{}-{}".format(i_days[0],i_days[-1]) if len(i_days)>1 else str(i_days[0])
with performance_report(filename="perf-report_proj-{}_{}.html".format(var,tmp)):
    for jj in range(restart,len(ind_y)-1):
        logging.info("starting y segment # {}".format(jj))
        sliy = slice(ind_y[jj], ind_y[jj+1])
        region[dim_h["y"]] = sliy
        sds_g = ds_g.isel({d:sliy for d in ds_g.dims if d.startswith("y_")})
        sds = ds.isel({d:sliy for d in ds.dims if d.startswith("y_")})
    
        ### here select subdomain
        if drop_land_x:
            index = sds_g[dim_h["x"]].where(sds_g[var+"maskutil"].max(dim_h["y"]), drop=True)
            slix = slice((int(index[dim_h["x"]][0]-1)//chk_x)*chk_x, int(index[dim_h["x"]][-1]))
            sds_g = sds_g.isel({d:slix for d in sds_g.dims if d.startswith("x_")})
            sds = sds.isel({d:slix for d in sds.dims if d.startswith("x_")})
            logging.info("x subdomain from {} to {}, size {}".format(
                          slix.start, slix.stop, slix.stop-slix.start)
                        )
        else:
            slix = slice(0,None)
        region[dim_h["x"]] = slix
        sds_g = sds_g.persist()
        logging.info("persisted sds_g")
        # compute pres and project on vertical modes + clean amod
        uvec = get_uv_mean_grid(sds, grid, sds_g)
        amod = proj_puv(uvec, sds_g).chunk({dim_h["y"]:chk_store[dim_h["y"]]})
    
        ### loop in time for computing and storing
        logging.info("starting time loop")
        tmea = time.time()
        for it in range(Nt//nk_t):
            tmes = time.time()
            slit = slice(it*nk_t,(it+1)*nk_t)
            region["t"] = slice( (it*nk_t)%nt_f, ((it+1)*nk_t-1)%nt_f + 1 )
            da = sim_dates[(it*nk_t)//nt_f]
            amod.isel(t=slit).chunk({"t":-1}).to_zarr(out_dir/out_file.format(da), mode="a", 
                                                      compute=True, region=region, safe_chunks=False
                                                      )
            logging.info("iteration y={}, t={}, ellapsed time {:.1f} s".format(jj, it, time.time()-tmes))
        logging.info("chunk y={} took {:.2f} min (mean /x-point/t: {:.2f} ms)".format(
                jj, (time.time()-tmea)/60., (time.time()-tmea)/(24*(slix.stop-slix.start))*1e3)
               )
        client.restart()

logging.info("  ----  FINISHED  ----  ")
ds_g.close()
ds.close()

### create log file with some information
for i,da in enumerate(sim_dates):
    with open(log_dir/log_file.format(da), "w") as fp:
        fp.write("JOB ID: {}\n".format(os.getenv("SLURM_JOBID")))
        fp.write("python script: {}\n".format(sys.argv[0]))
        fp.write("i_day {}, i_days {}, date {}\n".format(i_days[i],i_days,sim_dates))
        fp.write("nk_t {}, sk_y {}, working chunks {}, store chunks {}\n".format(nk_t, sk_y, chunks, chk_store))

