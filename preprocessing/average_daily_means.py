import os, sys

import numpy as np
import xarray as xr

from dask.distributed import Client, LocalCluster
from dask.distributed import performance_report

# input parameters

output_dir="/work/CT1/ige2071/SHARED/mean/"
variable = "gridT"

# best if batch_size matches task number in daily_mean.sh (ntasks parameter)
batch_size = 30 # days
suffix = str(batch)+"d_average"

# flag for graph outputs
graph=False


def print_graph(da, name, index, flag):
    """ store dask graph as png
    see: https://docs.dask.org/en/latest/diagnostics-distributed.html
    """
    if not flag:
        return
    da.data.visualize(filename='graph_{}_{}.png'.format(name, index),
                        optimize_graph=True,
                        color="order",cmap="autumn", node_attr={"penwidth": "4"})

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
    zarrs["zarr"] = zarr["log"].map(lambda l: l.replace("logs/","")+".zarr")
    # add flag is zarr file exists
    zarrs["flag"] = zarrs["zarr"].map(os.path.isdir)
    return zarrs

def is_batch_processed(batch_name):
    """ checks wether batch has been processed
    """
    log_path = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_path, suffix+variable+"_"+batch_name)
    return os.path.isfile(log_file):


if __name__ == '__main__':

    #cluster = LocalCluster(n_workers=14, threads_per_worker=1) # these may not be hardcoded
    #client = Client(cluster)
    #print(client)

    zarrs = get_zarr_with_timeline()
    print(zarrs)

    print("File being processed: "+file_in)

    zarr_batches = [zarrs.iloc[i:i+batch_size]
                    for i in range(0, zarrs.index.size, batch_size)
                    ]
    #batch_names = [b.index.iloc[0].strftime("%Y%m%d") for b in zarr_batches]
    print(zarr_batches)
    #print(batch_names)
    #print(file_batches[0])
    #files = file_batches[n_batch]

    # loop around batches
    for batch in zarr_batches:
        batch_name = batch.index.iloc[0].strftime("%Y%m%d")
        if is_batch_processed(batch_name):
            print(batch_name+ " processed - skips")
        else:
            print(batch_name+ " not processed")
            # process_batch(batch)
    #   check batch has been processed
    #   average batch if no

    sys.exit()

    ds = xr.open_dataset(file_in, chunks={"time_counter": -1, "deptht": 1, "y": 400,"x": -1})
    #ds = xr.open_dataset(file_in, chunks={"time_counter": -1, "deptht": 1, "y": -1,"x": -1})

    # auto chunks leads to chunks that are 24, 1182, 1182, i.e. 33530976 points
    # 33530976/8354 = 4013
    # data variables are 24 x 300 x 4729 x 8354

    # drop redundant variables
    ds = ds.drop_vars(["nav_lon", "nav_lat"])

    # check log file existence and exit if True
    date = str(ds["time_counter"].dt.strftime("%Y%m%d")[0].values)
    log_path = os.path.join(output_dir, "logs")
    log_file = os.path.join(log_path, "daily_mean_"+variable+"_"+date)
    if os.path.isfile(log_file):
        print(" File {} exists, skiping".format(log_file))
        sys.exit()

    print_graph(ds["votemper"], "open", date, graph)

    # debug
    #ds = ds.isel(deptht=slice(0,5))
    #print_graph(ds["votemper"], "isel", date, graph)

    # temporal average
    ds_processed = ds.mean("time_counter")
    ds_processed = ds_processed.chunk({"y":-1})
    #
    print(ds_processed)
    print_graph(ds_processed["votemper"], "processed", date, graph)

    #with performance_report(filename="dask-report.html"):
    zarr_archive = "daily_mean_"+variable+"_"+date+".zarr"
    ds_processed.to_zarr(os.path.join(output_dir, zarr_archive), mode="w")

    # create empty file to indicate processing was completed
    log_path = os.path.join(output_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    with open(log_file, "w+") as f:
        pass

    print("Congrats, processing is over !")
