# debug, quasi interactif

# salloc --constraint=BDW28 -N 1 -n 1 -t 10:00
# srun python average_daily_means.py

# https://www.cines.fr/calcul/faq-calcul-intensif/

import os, sys
from glob import glob

import numpy as np
import pandas as pd

import numpy as np
import xarray as xr

#from dask.distributed import Client, LocalCluster
#from dask.distributed import performance_report

# input parameters

output_dir="/work/CT1/ige2071/SHARED/mean/"
variable = "gridT"

# best if batch_size matches task number in daily_mean.sh (ntasks parameter)
batch_size = 30 # days
suffix = str(batch_size)+"d_average_"

# flag for graph outputs
graph=False


def print_graph(da, name, index, flag):
    """ store dask graph as png
    see: https://docs.dask.org/en/latest/diagnostics-distributed.html
    """
    if not flag:
        return
    da.data.visualize(filename='graph_average_{}_{}.png'.format(name, index),
                        optimize_graph=True,
                        color="order",
                        cmap="autumn",
                        node_attr={"penwidth": "4"},
                        )

def get_zarr_with_timeline():
    """ Build a pandas series with filenames indexed by date
    """
    files = sorted(glob(os.path.join(output_dir,
                                     "logs",
                                     "daily_mean_{}_*".format(variable)
                                     )
                        )
                   )
    time = [f.split("/")[-1].split("_")[-1]
            for f in files]
    timeline = pd.to_datetime(time)
    zarrs = (pd.Series(files, index=timeline, name="log")
             .sort_index()
             .to_frame()
             )
    # add zarr file
    zarrs["zarr"] = zarrs["log"].map(lambda l: l.replace("logs/","")+".zarr")
    # add flag is zarr file exists
    zarrs["flag"] = zarrs["zarr"].map(os.path.isdir)
    return zarrs

def is_batch_processed(batch_name):
    """ checks wether batch has been processed
    """
    log_dir = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_dir, suffix+variable+"_"+batch_name)
    return os.path.isfile(log_file)


if __name__ == '__main__':

    #cluster = LocalCluster(n_workers=14, threads_per_worker=1) # these may not be hardcoded
    #client = Client(cluster)
    #print(client)

    zarrs = get_zarr_with_timeline()

    # generate batch of zarr files
    zarr_batches = [zarrs.iloc[i:i+batch_size]
                    for i in range(0, zarrs.index.size, batch_size)
                    ]

    # loop around batches
    #for batch in zarr_batches:
    batch = zarr_batches[0]
    # name batches according to first day of the batch
    batch_name = batch.index[0].strftime("%Y%m%d")
    if is_batch_processed(batch_name):
        print(batch_name+ " processed - skips")
    else:
        print(batch_name+ " not processed")
        # check all zarr files are available, exit with error message otherwise
        # process_batch(batch)
        z = batch.iloc[0, "zarr"]
        print(z)

        # debug:
        #batch = batch.iloc[:5, :]

        #ds = xr.concat([xr.open_zarr(z) for z batch["zarr"].items()],
        # dim="time_counter")

        # temporal average
        #ds_processed = ds.mean("time_counter")

        #
        #print(ds_processed)
        #print_graph(ds_processed["votemper"], "processed", date, graph)

        #with performance_report(filename="dask-report.html"):
        #zarr_archive = suffix+variable+"_"+batch_name+".zarr"
        #ds_processed.to_zarr(os.path.join(output_dir, zarr_archive), mode="w")

        # create empty file to indicate processing was completed
        #log_dir = os.path.join(output_dir, "logs")
        #log_file = os.path.join(log_dir, suffix+variable+"_"+batch_name)
        #with open(log_file, "w+") as f:
        #    pass

    print("Congrats, processing is over !")
