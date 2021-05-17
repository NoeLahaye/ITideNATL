# debug, quasi interactif:
# conda_activate
# salloc --constraint=BDW28 -N 1 -n 1 -t 10:00
# srun python quick_plot.py  ... variables (deptht=...)
# ( https://www.cines.fr/calcul/faq-calcul-intensif/ )

# srun python quick_plot.py mean/30d_average_gridT_20090630.zarr votemper deptht=0
# srun python quick_plot.py /work/CT1/ige2071/SHARED/mean/global_mean_gridT.zarr votemper deptht=0

import os, sys
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

# input parameters

root_dir="/work/CT1/ige2071/SHARED/"
output_dir="/work/CT1/ige2071/SHARED/figs/"

if __name__ == '__main__':

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=14, threads_per_worker=1) # these may not be hardcoded
    client = Client(cluster)
    print(client)

     # collect arguments
    file_in = sys.argv[1]
    v = sys.argv[2]

    islice = {"time": 0, 
              "deptht": 0,
              "x": slice(0, None, 4),
              "y": slice(0, None, 4),
             }

    suffix=""
    if len(sys.argv)>3:
        for s in range(3,len(sys.argv)):
           variable, level = sys.argv[s].split("=")
           if ":" in level:
               level = list(map(int, level.split(":")))
               level = slice(*level)
           else:
               level = int(level)
           islice[variable] = level
           suffix=suffix+"_"+variable+str(level)

    da = xr.open_zarr(os.path.join(root_dir, file_in))[v]

    # update islice based on dimensions
    islice = dict((k,v) for k,v in islice.items() if k in da.dims)

    da = da.isel(**islice)

    plt.switch_backend("agg")

    fig, ax = plt.subplots(1,1)
    da.plot(ax=ax)

    fig_name=file_in
    if "/" in file_in:
        fig_name = fig_name.split("/")[-1]
    fig_name=fig_name.replace(".zarr", suffix+".png")
    fig_path = os.path.join(output_dir, fig_name)
    fig.savefig(fig_path, dpi=150)

    print("Congrats, processing is over !")
