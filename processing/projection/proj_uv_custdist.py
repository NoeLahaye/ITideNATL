### ONGOING WORK not finished because not sure it is useful.

from pathlib import Path
import os, time, sys, shutil
import logging
logging.basicConfig(format='[{}] %(asctime)s -- %(message)s'.format(sys.argv[0]), 
        level=logging.INFO, stream=sys.stdout)

import xarray as xr
from xgcm import Grid

from itidenatl.tools import files as uf, dataio as io, compute as uc
#from proj_utils import load_grid_ds#, proj_pres, get_pres_one_dg as get_pres
from itidenatl import gridop as gop

if len(sys.argv)==1:
    print("usage: proj_uv_custdist.py iday [u/v [jy_beg [jy_end]]]")
    os._exit()
iday = sys.argv[1]
da = uf.get_date_from_iday(iday)
var = sys.argv[2] if len(sys.argv)>2 else "u"

homedir = Path(os.getenv("HOME"))
workdir = Path(os.getenv("WORKDIR"))
scratch = Path(os.getenv("SCRATCHDIR"))
worksha = Path("/work/CT1/ige2071/SHARED")
grid_mode_pres = workdir/"eNATL60_grid_vmodes_proj_pres.zarr" 
grid_mode_uv = workdir/f"eNATL60_grid_vmodes_proj_{var}.zarr"
    
inp_chks = {"t":1, "z":10, "y":100, "x":-1}
wrk_chks = {"t":1, "z":10, "y":100, "x":-1}
ydim = "y_r" if var=="v" else "y_c"
xdim = "x_r" if var=="u" else "x_c"
str_chks = {"mode":1, "t":24, ydim:400, xdim:-1}
finish = True

if len(sys.argv)==4:
    if sys.argv[3] == "finish":
        regy = None
    else:
        regy = slice(int(sys.argv[3]), None)
elif len(sys.argv)==5:
    regy = slice(int(sys.argv[3]), int(sys.argv[4]))
    finish = False
else:
    regy = slice(0, None)
sk_y, sk_t = 100, 6
store_grid, store_uv = True, False

out_file = worksha/f"modal_proj/modamp_{var}/modamp_{var}_global_{da}.zarr" 
log_dir = homedir/"working_on/processing/log_proj_uv"
log_file = f"proj_{var}_{iday}.log" 
from_files = {"grid_mode_uv": grid_mode_uv, "grid_mode_pres": grid_mode_pres}
varname = {"u":"vozocrtx", "v":"vomecrty"}[var]
from_files[varname] = uf.get_eNATL_path(varname, iday)
from_files["sossheig"] = uf.get_eNATL_path("sossheig", iday)
file_list = scratch/f"prov_proj-{var}_{iday}_file-list.txt"
tmp_fil = scratch/f"prov_proj_{var}_{iday}.zarr"

ditp = "x" if var == "u" else "y"
    
### local routines
verbose = False

### local routines
def open_grid_file(isel=None, add_zl=False):
    ds = xr.open_zarr(from_files["grid_mode_uv"])
    ds = ds.reset_coords([f"{var}mask", f"{var}maskutil"], drop=True)
    if add_zl:
        dg = xr.open_zarr(from_files["grid_mode_pres"])
        ds = ds.assign_coords(z_l=dg.z_l)
        dg.close(); del dg
    ds = ds.chunk({d:wrk_chks[d[0]] for d in ds.dims if d[0] in wrk_chks})
    if isel is not None:
        ds = ds.isel(isel)
    return ds

def interp_xy(da, xgrid):

    res = xgrid.interp(da, ditp.upper(), boundary="extend")
    return res.chunk({f"{ditp}_r":wrk_chks[ditp]})

def prep_grid_ychk(ds, regy):
    """ load grid and ssh, interpolate velocity on mean grid """
    dg = open_grid_file(add_zl=True) # open uv grid, with both vertical grids
    ssh_path = from_files["sossheig"] # add ssh 
    dg = dg.merge(io.open_one_var(ssh_path, chunks=wrk_chks, varname="sossheig")\
                  .rename({"sossheig":"ssh_inst"})
                 ) # add ssh -- this also add T grid
    xgrid = Grid(dg, periodic=False)
    dg["sossheig"] = interp_xy(dg.ssh_inst, xgrid) - dg["sossheig"] # ssh pert on uv grid
    sdg = dg.drop("ssh_inst").drop_dims([ditp+"_c"], errors="ignore")\
            .get([f"depth_{var}_m", "sossheig", f"e3{var}_m", "hbot"])\
            .isel({ydim:regy}).copy() # take only what is needed and y-subdomain

    print("opened grid")
    if store_grid:
        for v in list(sdg.coords)+list(sdg.data_vars):
            sdg[v].encoding.pop("chunks", None)
        sdg.to_zarr(tmp_fil, mode="w", compute=True, consolidated=True)
        sdg.close(); del sdg
        sdg = xr.open_zarr(tmp_fil)
        print("stored grid")
    sdg = xr.merge([sdg.drop("sossheig").persist(), sdg.sossheig]) # persist a few needed grid components
    print("persisted grid -- now interpolating velocity")
    return sdg, xgrid

def zinterp_uv_ychk(ds, dg, xgrid):
    """interpolate velocity on mean grid. 
    Computing u*e3u for subsequent projection
    expects a subdomain
    """
    data = ds[varname]
    res = xgrid.diff(xgrid.interp(data, "Z", boundary="extrapolate"),
                     "Z", boundary="extend").chunk({"z_c":wrk_chks["z"]})
    delz = gop.get_del_zt(dg[f"depth_{var}_m"], ssh=dg.sossheig, hbot=dg.hbot).astype("float32")
    sds = ( data * dg[f"e3{var}_m"] + delz * res
          ).to_dataset(name=data.name).reset_coords(drop=True)
    
    if store_uv:
        uc.custom_distribute(sds, to_dir=tmp_fil, dims={"t":sk_t}, verbose=verbose)
        del sds, dg
        sds = xr.open_zarr(tmp_fil)
        print("stored interpolated velocity")
    return sds
                        
def proj_ychk(ds, regy):
    dg = open_grid_file(isel={ydim:regy})
    #dg = ds_g.get(["phi", "norm"]).reset_coords(drop=True).copy()
    phi = dg.phi.reset_coords(drop=True).persist()
    norm = dg.norm.reset_coords(drop=True).persist()
    dg.close(); del dg
    amod = (ds[varname] * phi).sum("z_c")/norm
    
    amod = amod.to_dataset(name=varname) #proj_pres(sds.pres, ds_g)
    fname = f"modamp_{var}_{iday}_y{regy.start}-{regy.stop}.zarr"
    uc.custom_distribute(amod, to_dir=scratch/fname, dims={"t":sk_t}, verbose=verbose)
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
        ds = io.open_one_var(from_files[varname], 
                             chunks=inp_chks, varname=varname\
                            ).reset_coords(drop=True)
        logging.info("opened uv data -- ellapsed time {:.1f} s".format(time.time()-tmes))
    
        ds_g = open_grid_file()
        if regy is not None and regy.stop is None:
            regy = slice(0, ds_g.dims[ydim])
        ds_g.close(); del ds_g
        
        for jy in range(regy.start, regy.stop, sk_y):
            regy = slice(jy, jy+sk_y)

            tmes, tmeg = time.time(), time.time()
            sdg, xgrid = prep_grid_ychk(ds, regy)
            logging.info("prep grid, took {:.1f} s".format(time.time()-tmes))

            tmes = time.time()
            sds = zinterp_uv_ychk(ds, sdg, xgrid)
            logging.info("interp uv on mean grid, took {:.1f} s".format(time.time()-tmes))

            tmes = time.time()
            filnam = proj_ychk(sds, regy)
            logging.info("projected and stored, took {:.1f} s".format(time.time()-tmes))

            if not store_uv:
                del sdg
            del sds
            with open(file_list, "a") as fp:
                fp.write(str(filnam)+"\n")
            logging.info("jy={}-{} took {:.1f} s".format(jy,jy+sk_y, time.time()-tmeg))
            #client.restart()
        shutil.rmtree(tmp_fil)
    
### read files, rechunk and store
    if finish:
        tmes = time.time()
        with open(file_list, "r") as fp:
            file_listes = list(set(r.rstrip('\n') for r in fp.readlines()))
        ds = xr.open_mfdataset(file_listes, engine="zarr")
        ds = ds.chunk(str_chks)
        ds[varname].encoding.pop("chunks")
        attrs = dict(description = f"{var}-velocityp modal amplitude, resulting from the projection "
                         "of the horizontal velocity on the vertical modes",
                     day_of_simulation = da, iday = iday,
                     simulation = "eNATL60 (with tides)",
                     from_files = [str(v) for v in from_files.values()],
                     generating_script = sys.argv[0],
                     date_generated = logging.time.strftime("%Y-%m-%d (%a at %H h)"),
                     creator = "N. Lahaye (noe.lahaye@inria.fr)"
                     )
        ds.attrs = attrs
        ds[varname].attrs.update(dict(long_name=f"{ditp}-component of horizontal velocity modal amplitude",
                                      units="m/s"
                                                          ))
        ds.to_zarr(out_file, mode="w", compute=True, consolidated=True, safe_chunks=True)
        logging.info(f"rechunked and stored final file in {out_file}" 
                       "; took {:.1f} s; THE END".format(time.time()-tmes))
        for f in file_listes:
            shutil.rmtree(f)
        #file_list.unlink()
    
        ### create log file with some information
        with open(log_dir/log_file.format(da), "w") as fp:
            fp.write("JOB ID: {}\n".format(os.getenv("SLURM_JOBID")))
            fp.write("python script: {}\n".format(sys.argv[0]))
            fp.write(f"i_day {iday}, date {da}\n")
            fp.write(f"nk_t {sk_t}, sk_y {sk_y}\n")
            fp.write(f"read chunks {inp_chks}, working chunks {wrk_chks}, store chunks {str_chks}\n")
    
