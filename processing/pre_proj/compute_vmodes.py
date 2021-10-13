#!/usr/bin/env python
# coding: utf-8

# # compute_vmodes
# Compute vertical modes based on mean stratification
# 
# Strategy: process sequentially over y-segment, droping land points (columns) for each segment. This reduces the memory usage and is well adapted for a single-node dask cluster, but doesn't work if one wants to process large segments using a larger cluster. The efficiency of this script could be improved by having a mean not to load land points in memory within the distributed loop, e.g. through a map_blocks or something like that
# 
# #### Timing and memory consumption:
# * not droping land: up to 90 GB
# * droping lands: depends on the segment, but typically max at 55 GB
# * first 2 y-segments of size 200: 50 & 60 minutes (dropping lands), 50 & 76 minutes (not dropping lands)
# 
# This notebook is for processing eNATL60 data on occigen
# 

# In[1]:


from pathlib import Path
import os, time, datetime

import xarray as xr
from xgcm import Grid
from itidenatl.nemodez import Vmodes, load_vmodes


# In[2]:


from dask.distributed import Client, LocalCluster, wait
scratch = os.getenv("SCRATCHDIR")
cluster = LocalCluster(n_workers=8, local_directory=scratch) #n_workers=24, threads_per_worker=1, memory_limit=8e6,silence_logs=50
client = Client(cluster)
client


# In[3]:


### define paths
scratch = Path(os.getenv("SCRATCHDIR"))
grid_path = scratch #Path("/store/CT1/hmg2840/lbrodeau/eNATL60/eNATL60-I/")
mean_path = scratch #Path("/work/CT1/ige2071/SHARED/mean")

avg_type = "30d" # "30d" or "global"
avg_date = "20090630" # will be ignored if avg_type is "global"
app = "_"+avg_date if avg_type == "30d" else ""
zgrid_fname = f"eNATL60_{avg_type}-mean_z-grid{app}.zarr" 
strat_fname = f"eNATL60_{avg_type}-mean_bvf{app}.zarr"

zgrid_file = scratch/zgrid_fname
strat_file = scratch/strat_fname
out_file = scratch/f"eNATL60_{avg_type}-mean_vmodes{app}.zarr"

### processing parameters
nmodes = 10
out_chk = {"mode":1, "x_c":-1, "z_c": 30, "z_l":30}
wrk_chk = {"x_c":600}
nseg_y = 200 # y-segment size: choose it a multiple or a divider of chunk size
drop_land = True


# In[4]:


dg = xr.open_zarr(zgrid_file).astype("float32")
ds = xr.open_zarr(strat_file).astype("float32")

coord_copy = ["e3t", "e3w", "hbot", "tmask", "depth_c"] # WARNING remove depth_c_3d
ds = ds.assign_coords({co:dg[co] for co in coord_copy}).chunk(wrk_chk).unify_chunks()
# remove chunks from encoding, for zarr storage
for var in ds.coords.values(): 
    var.encoding.pop("chunks", None)
chunks = {k:v[0] for k,v in ds.chunks.items()}
ds


# In[5]:


get_ipython().run_cell_magic('time', '', '### create vmods object and zarr archive (delayed mode) \nvmods = Vmodes(ds, Grid(ds, periodic=False), nmodes=nmodes, free_surf=True, \n               persist=False, chunks=out_chk)\nput_attrs = {"from_files":[str(zgrid_file), str(strat_file)],\n             "simulation": "eNATL60", "processing notebook":"compute_vmodes.ipynb",\n             "date_processed":datetime.datetime.today().isoformat(timespec="minutes"),\n             "process_params_wrk_chk": [(k,v) for k,v in wrk_chk.items()]\n            }\nput_attrs.update({f"process_params_{k}":eval(k) for k in ["nseg_y", "drop_land"]})\nvmods.ds.attrs = put_attrs\nvmods.store(out_file, coords=False, mode="w", compute=False, consolidated=True)\nvmods.ds')


# In[ ]:


get_ipython().run_cell_magic('time', '', '### Compute and store, looping over y-segments\n\nNy = ds.y_c.size\nregion = {"z_c": slice(0,None), "z_l":slice(0,None), "mode":slice(0,None)}\n\ndef get_subds(ds):\n    """ wrapper to get rid of land points (columns).\n    Warning: this works only if x_c increment is 1 """\n    lnd_pts = (ds.tmaskutil==0).sum().values\n    print("number of land points: {} ({:.1f}%)".format(lnd_pts, lnd_pts*100/ds.tmaskutil.size),\n         end="; ")\n    index = ds.tmaskutil.max("y_c")\n    index = index.where(index, drop=True).x_c - index.x_c[0] #\n    #index = slice((int(index[0])//wrk_chk["x_c"])*wrk_chk["x_c"], int(index[-1])+1)\n    index = slice(int(index[0]), int(index[-1])+1)\n    sds = ds.isel(x_c=index)\n    lnd_pts = (sds.tmaskutil==0).sum().values\n    print("after selection: {} ({:.1f}%)".format(lnd_pts, lnd_pts*100/sds.tmaskutil.size))\n    return sds, index\n\n### this is the loop\nfor jy in range(0, 2*chunks["y_c"], nseg_y):\n    tmes = time.time()\n    sliy = slice(jy, min(jy+nseg_y, Ny))\n    if drop_land:\n        sds, slix = get_subds(ds.isel(y_c=sliy))\n    else:\n        sds = ds.isel(y_c=sliy)\n        slix = slice(0, None)\n    grid = Grid(sds, periodic=False)\n    region.update({"y_c":sliy, "x_c":slix})\n    vmods = Vmodes(sds, grid, modes=nmodes, free_surf=True, persist=False, \n               chunks=out_chk)\n    vmods.ds = vmods.ds.where(vmods.ds.tmaskutil)\n    vmods.store(out_file, coords=False, mode="a", compute=True, region=region)\n    print("segment {0} done, size {1}, {2:.1f} min".format(jy, sds.x_c.size, \n                                                          (time.time()-tmes)/60)\n         )')


# In[8]:


vmods.ds


# ### todo 
# essayer avec une distribution de chunks en x

# In[15]:


sds.x_c[0]


# In[9]:


get_ipython().run_cell_magic('time', '', '### custom distribute along x\nnchk_x = 2\n\nn_subx = int(sds.x_c.size)//(nchk_x*chunks["x_c"])+1\nprint("will do {:0d} sub-domain computation along x".format(n_subx))\ns_subx = chunks["x_c"]*nchk_x\nfor ix in range(n_subx):\n    tmes = time.time()\n    slix = slice(ix*s_subx, (ix+1)*s_subx)\n    ssd = sds.isel(x_c=slix)\n    region["x_c"] = slice(int(ssd.x_c[0])-1, int(ssd.x_c[-1]))\n    vmods = Vmodes(ssd, grid, nmodes=nmodes, free_surf=True, persist=False, chunks={"mode":1})\n    vmods.store(out_file, coords=False, mode="a", compute=True, region=region)\n    twall, npts = time.time()-tmes, int(ssd.tmaskutil.sum())\n    print("slice {0}, {1}, {2:.1f} min for {3} points".format(\n                ix, region["x_c"], twall/60, npts),\n          "({:.2f} ms/pt)".format(twall/npts*1e3)\n         )')


# In[9]:


get_ipython().run_cell_magic('time', '', '### custom distribute along x\nnchk_x = 2\n\nn_subx = int(sds.x_c.size)//(nchk_x*chunks["x_c"])+1\nprint("will do {:0d} sub-domain computation along x".format(n_subx))\ns_subx = chunks["x_c"]*nchk_x\nfor ix in range(n_subx):\n    tmes = time.time()\n    slix = slice(ix*s_subx, (ix+1)*s_subx)\n    ssd = sds.isel(x_c=slix)\n    region["x_c"] = slice(int(ssd.x_c[0])-1, int(ssd.x_c[-1]))\n    vmods = Vmodes(ssd, grid, nmodes=nmodes, free_surf=True, persist=False, chunks={"mode":1})\n    vmods.store(out_file, coords=False, mode="a", compute=True, region=region)\n    print("slice {0}, {1}, {2:.1f} min for {3} points".format(\n                ix, region["x_c"], (time.time()-tmes)/60, int(ssd.tmaskutil.sum()))\n         )')


# In[20]:


get_ipython().run_cell_magic('time', '', '### custom distribute along x\nnchk_x = 4\n\nn_subx = int(sds.x_c.size)//(nchk_x*chunks["x_c"])+1\nprint("will do {:0d} sub-domain computation along x".format(n_subx))\ns_subx = chunks["x_c"]*nchk_x\nfor ix in range(n_subx):\n    tmes = time.time()\n    slix = slice(ix*s_subx, (ix+1)*s_subx)\n    ssd = sds.isel(x_c=slix)\n    region["x_c"] = slice(int(ssd.x_c[0])-1, int(ssd.x_c[-1]))\n    vmods = Vmodes(ssd, grid, nmodes=nmodes, free_surf=True, persist=False, chunks={"mode":1})\n    vmods.store(out_file, coords=False, mode="a", compute=True, region=region)\n    print("slice {0}, {1}, {2:.1f} min for {3} points".format(\n                ix, region["x_c"], (time.time()-tmes)/60, int(ssd.tmaskutil.sum()))\n         )')


# In[11]:


ds_re = xr.open_zarr(out_file)
sds_re = ds_re.isel(x_c=slice(0,None,4), y_c=slice(0,None,4))


# In[10]:


plt.figure()
sds_re.c.isel(mode=0).plot()

plt.figure()
(sds_re.c*sds_re.mode).isel(mode=slice(1,None)).plot(col="mode", col_wrap=5, 
                                       cbar_kwargs={"orientation":"horizontal"})


# In[18]:


sds_re


# In[19]:


sds_re.norm.isel(mode=1).where(ds.tmaskutil).plot()


# ## renormalization
# because computation was wrong...
# WARNING: I should make sure not to copy llon_cc and llat_cc because I don't need them...

# In[8]:


#nmodes = 10
work = (Path(os.getenv("WORK1"))/"../SHARED").resolve()
out_new = work/"vmodes/vmodes_{}.zarr".format(nmodes)
out_new


# In[36]:


get_ipython().run_cell_magic('time', '', 'ds_re.get(["c"]).to_zarr(out_new, mode="w", consolidated=True)')


# In[11]:


hocean = dg.e3t.where(dg.tmask).sum("z_c")#.persist()


# In[43]:


get_ipython().run_cell_magic('time', '', '(ds_re.get(["norm"]) * hocean / ds_re.norm).to_zarr(out_new, mode="a")')


# In[12]:


renorm = (hocean / ds_re.norm)**.5
renorm = renorm.persist()


# In[15]:


get_ipython().run_cell_magic('time', '', '\nvar = "phiw"\n\nregion = {"x_c":slice(0,None), "y_c":slice(0,None), "mode":None}\nif var == "phiw":\n    region["z_l"] = slice(0,None)\nelse:\n    region["z_c"] = slice(0,None)\n    \nfor imod in range(ds_re.mode.size):\n    tmes = time.time()\n    dsa = ds_re.get([var]).isel(mode=[imod])\n    region["mode"] = slice(imod,imod+1)\n    (dsa * renorm.isel(mode=[imod])).to_zarr(out_new, mode="a", region=region)\n    print("mode {0} done: {1:.1f} s".format(imod,time.time()-tmes))')


# In[22]:


### add attributes (I lost them somehow during the copying process...)
import zarr

nc = zarr.open(str(out_new), mode="r+")
for k,v in ds_re.attrs.items():
    print(k,v)
    nc.attrs[k] = v


# In[30]:


ds_new = xr.open_zarr(out_new)
ds_new


# In[17]:


ds_new.norm.isel(mode=0).plot()


# In[45]:


## to reload vertical modes: don't use nemodez.load_vmodes
dm = xr.open_zarr(out_new).unify_chunks()
vmods = Vmodes(ds, Grid(ds, periodic=False), nmodes=dm.nmodes, free_surf=dm.free_surf,
              persist=False, chunks={k:v[0] for k,v in dm.chunks.items()})
for v in dm.data_vars:
    vmods.ds[v] = dm[v]
vmods.ds


# ## Old stuff

# In[14]:


get_ipython().run_cell_magic('time', '', '### just testing: compute and store with sub-dataset\nvmods = Vmodes(sds, grid, nmodes=10, free_surf=True, persist=False, chunks={"mode":1})\nvmods.store(scratch/"prov.zarr", coords=False, mode="w", compute=True, \n            consolidated=True)')


# In[8]:


vmods = Vmodes(ds, Grid(ds, periodic=False), nmodes=10, free_surf=True, 
               persist=False, chunks={"mode":1})
vmods.ds


# In[10]:


get_ipython().run_cell_magic('time', '', 'vmods.ds.isel(x_c=slice(0,10), y_c=slice(0,10)).compute()')


# In[ ]:





# In[12]:


dm = xr.open_zarr(scratch/"vmodes.zarr")
dm


# In[17]:


dm.phi.isel(x_c=1, y_c=1).plot.line(y="z_c")


# In[10]:


get_ipython().run_cell_magic('time', '', 'vmods.ds.reset_coords(drop=True).to_zarr("vmodes.zarr", mode="w")')


# In[15]:


i_rng = slice(4014-1, 4134)
j_rng = slice(1438-1, 1576)

sds = ds_re.isel(x_c=i_rng, y_c=j_rng)
sds.bvf.mean(dim=("x_c", "y_c")).plot(y="depth_l")
plt.grid(True)


# In[ ]:


bvf_moy = sds.bvf.mean(dim=("x_c", "y_c")).


# ## testings
# 

# In[20]:


### look at impact of chunking on proportion of non-computed tiles
nchk = 400
print("fraction of land points: {:.2f}".format((ds.tmaskutil==0).mean().values))
comp_tile = ds.tmaskutil.coarsen(x_c=nchk, y_c=nchk, boundary="trim", side="left").max()
print("would compute {} over {} (fraction not computed: {:.2f})".format(comp_tile.sum().values, 
                                                           comp_tile.size,
                                                           1-comp_tile.mean().values)
     )
comp_tile.plot()


# In[9]:


from itertools import product
import time

def dist_write_chunks(ds, path, verbose=True, nchk=None):
    """ utilitary function that loops over chunks to store a DataArray in a zarr archive (append mode)"""
    #dim_chk = [next(di for di in da.dims if di.startswith(dim)) for dim in dims]
    if verbose:
        tmei = time.time()
    if isinstance(ds, xr.DataArray):
        name = ds.name if ds.name else "data"
        ds = ds.to_dataset(name=name)
    ### warning this is for testing
    ds = ds.chunk({"z_l":-1})
    
    dims = [*ds.chunks.keys()]
    chks = [np.r_[0, np.array(chk).cumsum()] for chk in ds.chunks.values()]
    if nchk:
        chks = [np.r_[chk[:-1:nchk], chk[-1]] for chk in chks]
        #ds = ds.chunk({di:ds.chunks[di][0]*nchk for di in ["x_c","y_c"] })
    #nks = [len(chk) for chk in ds.chunks.values()]
    nks = [len(chk)-1 for chk in chks] 
    if verbose:
        print("total number of chunks:", np.product(nks), dims, nks)
    ext = [np.arange(nk) for nk in nks]
    for chs in product(*ext):
        if verbose:
            tmes = time.time()
        isel = {dim:slice(*chks[ii][chs[ii]:chs[ii]+2]) for ii,dim in enumerate(dims)}
        ds.isel(isel).to_zarr(path, mode="a", region=isel)
        if verbose:
            print("chunk {}: {:.0f} s".format(chs, time.time()-tmes), end="; ")
            print(isel)
    if verbose:
        print("\n finished. Ellapsed time: {:.1f} min".format((time.time()-tmei)/60))


# In[10]:


i_rng = slice(0, 2000) #slice(4014-1, 4134)
j_rng = slice(0, 1000) #slice(1438-1, 1576)

sds = ds.isel(x_c=i_rng, y_c=j_rng)
sds


# In[20]:


client.restart()


# In[16]:


get_bvf(sds, return_ds=True).to_zarr(out_file, mode="w", consolidated=True, compute=False)


# In[16]:


get_ipython().run_cell_magic('time', '', '### no sub-chunking in z for computation (expect truediv done by one thread)\ndist_write_chunks(get_bvf(sds, return_ds=True), out_file)')


# In[12]:


get_ipython().run_cell_magic('time', '', 'dist_write_chunks(get_bvf(sds.chunk({"z_l":50, "z_c":50, "x_c":400, "y_c":400}), return_ds=True), \n                  out_file, nchk=None)')


# In[12]:


get_ipython().run_cell_magic('time', '', '### this is the fastest I found\ndist_write_chunks(get_bvf(sds.chunk({"z_l":50, "z_c":50, "x_c":400, "y_c":400}), return_ds=True), \n                  out_file, nchk=2)')


# In[21]:


get_ipython().run_cell_magic('time', '', '### this is the fastest I found\ndist_write_chunks(get_bvf(sds.chunk({"z_l":20, "z_c":20, "x_c":400, "y_c":400}), return_ds=True), \n                  out_file, nchk=2)')


# In[14]:


ds_re = xr.open_zarr(out_file)
bvf_moy = ds_re.bvf.mean(dim=("x_c","y_c")).persist()
bvf_moy.plot(y="depth_l")


# In[21]:


sds_re = xr.open_zarr(out_file)
ds_re = xr.open_zarr(scratch/"global_mean_bvf.zarr")
#bvf_re = ds_re.bvf.isel(x_c=1999, y_)
bvf_sr = sds_re.bvf.isel(x_c=[-1], y_c=[-1])


# In[24]:


sds_re, ds_re = xr.align(sds_re.isel(x_c=[-1], y_c=[-1]), ds_re)
ds_re


# In[27]:


sds_re.bvf.isel(z_l=slice(-52,None)).plot(y="depth_l")


# In[13]:


get_ipython().run_cell_magic('time', '', 'bvf2(sds).compute()')


# ## Old stuff
# 

# In[5]:


get_ipython().run_cell_magic('time', '', '#chk_z = 20\nds_tot = load_xorca_dataset(data_files=[], aux_files=grid_files,\n                              decode_cf=True, model_config="nest"#, target_ds_chunks={"z_c":chk_z, "z_l":chk_z}\n                             ).reset_coords(drop=True)\nprint("dataset is {:.1f} GB".format(ds_tot.nbytes/1e9))')


# In[6]:


ds_tot


# In[13]:


### First load zarr of corected vertical grid
ds_grz = xr.open_zarr(grid_path/zgrid_name)
ds_tot = ds_tot.merge(ds_grz.set_coords("hbot")).drop_vars("sossheig").unify_chunk()


# In[7]:


i_rng = slice(4014-1, 4134)
j_rng = slice(1438-1, 1576)

sds = ds_tot.isel(x_c=i_rng, x_r=i_rng, y_c=j_rng, y_r=j_rng)
sds


# In[14]:


bvf = get_bvf(sds.where(sds.tmask))
bvf


# In[15]:


get_ipython().run_cell_magic('time', '', 'bvf.compute()')


# In[ ]:




