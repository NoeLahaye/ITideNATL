# Notes eNATL60 on occigen

Files physical structure:

```
dimensions:
	axis_nbounds = 2 ;
	x = 8354 ;
	y = 4729 ;
	deptht = 300 ;
	time_counter = UNLIMITED ; // (24 currently)

	float votemper(time_counter, deptht, y, x) ;
		votemper:standard_name = "sea_water_potential_temperature" ;

		votemper:_Storage = "chunked" ;
		votemper:_ChunkSizes = 1, 1, 10, 8354 ;
		votemper:_DeflateLevel = 2 ;
		votemper:_Endianness = "little" ;

```


[Notes Aurélie](https://github.com/auraoupa/scripts-occigen-for-Arne):

- Environment created on datarmor:
```
conda create -n occigen -c conda-forge python=3.8 dask dask-jobqueue \
            xarray zarr netcdf4 python-graphviz \
            tqdm \
            jupyterlab ipywidgets \
            cartopy geopandas descartes \
            scikit-learn seaborn \
            hvplot geoviews datashader nodejs \
            intake-xarray gcsfs \
            cmocean gsw \
            xhistogram \
            pytide pyinterp
conda activate occigen
pip install git+https://github.com/xgcm/xgcm.git
pip install git+https://github.com/MITgcm/xmitgcm.git
pip install git+https://github.com/xgcm/xrft.git
pip install rechunker
```

- Need to install `conda-pack` prior to conda packing see [this post](https://litingchen16.medium.com/how-to-use-conda-pack-to-relocate-your-condo-environment-622b68e077df):
```
conda install conda-pack
conda pack -n occigen
[copy to occigen occigen.tar.gz and follow Aurélie's instructions]
[activate with: source /scratch/cnt0024/ige2071/aponte/conda/occigen/bin/activate ]
```

- Dehydrate a couple of files
```
cd /store/CT1/hmg2840/lbrodeau/eNATL60/eNATL60-BLBT02-S/00388801-00399600
ncdump -sh eNATL60-BLBT02_1h_20090630_20090704_gridT_20090630-20090630.nc
ncdump -sh eNATL60-BLBT02_1h_20090630_20090704_gridT_20090701-20090701.nc
ncdump -sh eNATL60-BLBT02_1h_20090630_20090704_gridT_20090702-20090702.nc
```


[eNATL60](https://github.com/ocean-next/eNATL60/blob/master/02_experiment-setup.md)
- `eNATL60-BLB002` experiment (WITHOUT explicit tidal motion)
- `eNATL60-BLBT02` experiment (WITH explicit tidal motion)
- Files are in `/store/CT1/hmg2840/lbrodeau/eNATL60/`

[xarray netcdf chunking](https://github.com/pydata/xarray/issues/1440)

[CINES stockage](https://www.cines.fr/calcul/organisation-des-espaces-de-donnees/espaces-de-donnees-quotas-disques-restaurations-de-fichiers/)

Create vnc server on dunree:

- on dunree: `vncserver :2 -localhost -geometry 1400x1000`
- to kill server: `vncserver -kill :2`
- on baliste: `ssh -L 1234:localhost:5902 -C -N -l aponte dunree`
- on baliste: launch tigervnc with the address `localhost:1234` and appropriate password

Submit/Kill job:
```
sbatch job.sh
scancel <jobid>
```


Create dask config files and increase timeouts:
```
git clone https://github.com/dask/distributed.git
mkdir .config
mkdir .config/dask
cp distributed/distributed/distributed.yaml .config/dask/
# change timeout in distributed.yaml file:
timeouts:
	connect: 600s          # time before connecting fails
	tcp: 600s              # time before calling an unresponsive connection dead
```



```
aponte@login2:~/ITideNATL/preprocessing$ job_info 11695228-01
               JobID    JobName      User  Partition        NodeList    Elapsed      State ExitCode     MaxRSS                        AllocTRES
-------------------- ---------- --------- ---------- --------------- ---------- ---------- -------- ---------- --------------------------------
            11695228       TAVE    aponte      bdw28           n3383   02:14:36  COMPLETED      0:0            billing=56,cpu=56,energy=123369+
      11695228.batch      batch                                n3383   02:14:36  COMPLETED      0:0      9118K         cpu=28,mem=59000M,node=1
          11695228.0  task.conf                                n3383   02:14:30  COMPLETED      0:0  27786460K          cpu=2,mem=59000M,node=1
```


Interactif: https://www.cines.fr/calcul/faq-calcul-intensif/

```
salloc --constraint=BDW28 -N 1 -n 1 -t 10:00
srun mon_executable
```
