import os, sys

import numpy as np
import xarray as xr

from dask.distributed import Client, LocalCluster
from dask.distributed import performance_report

def print_graph(da, name, index, flag):
    """ store dask graph as png
    see: https://docs.dask.org/en/latest/diagnostics-distributed.html
    """
    if not flag:
        return
    da.data.visualize(filename='graph_{}_{}.png'.format(name, index),
                        optimize_graph=True,
                        color="order",cmap="autumn", node_attr={"penwidth": "4"})


if __name__ == '__main__':

    #from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=7, threads_per_worker=1) # these may not be hardcoded
    client = Client(cluster)
    print(client)

    # collect arguments
    file_in = sys.argv[1]
    variable = sys.argv[2]
    output_dir = sys.argv[3]

    # flag for graph outputs
    graph=False

    print("File being processed: "+file_in)

    #ds = xr.open_dataset(file_in)
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
