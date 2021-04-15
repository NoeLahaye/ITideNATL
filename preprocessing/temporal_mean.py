import os, sys

import numpy as np
import xarray as xr

from dask.distributed import Client, LocalCluster
from dask.distributed import performance_report


def print_graph(da, name, index):
    da.data.visualize(filename='graph_{}_{}.png'.format(name, index), 
                        optimize_graph=True,
                        color="order",cmap="autumn", node_attr={"penwidth": "4"})
    

if __name__ == '__main__':

    #from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()
    client = Client(cluster)

    print(client)

    # collect arguments
    file_in = sys.argv[1]
    path_out = sys.argv[2]

    print("File being processed: "+file_in)

    # debug
    #rs = np.random.RandomState(0)
    #array1 = xr.DataArray(rs.randn(1000, 100000), dims=["place", "time"])  # 800MB
    #chunked1 = array1.chunk({"place": 10}) 
    #mean_values = chunked1.mean("time").compute()
    #print(mean_values)

    #ds = xr.open_dataset(file_in)
    #ds = xr.open_dataset(file_in, chunks={"time_counter": 1, "deptht": 1, "y": -1,"x": -1})
    #ds = xr.open_dataset(file_in, chunks={"time_counter": -1, "deptht": 1, "y": 200,"x": -1})
    #ds = xr.open_dataset(file_in, chunks={"time_counter": -1, "deptht": 100, "y": 2000,"x": -1})
    ds = xr.open_dataset(file_in, chunks={"time_counter": -1, "deptht": 1, "y": -1,"x": -1})
    ds = ds.drop_vars(["nav_lon", "nav_lat"])
    date = ds["time_counter"].dt.strftime("%Y%m%d")[0].values

    print_graph(ds["votemper"], "open", date)
    #ds = ds.isel(y=slice(0,400))
    #ds = ds.isel(deptht=slice(0,5))
    # auto chunks leads to chunks that are 24, 1182, 1182, i.e. 33530976 points
    # 33530976/8354 = 4013 
    # data variables are 24 x 300 x 4729 x 8354
    print(ds, flush=True)
    print_graph(ds["votemper"], "isel", date)

    # temporal average
    ds_processed = ds.mean("time_counter")
    # now just select one slice
    #ds_processed = ds.isel(time_counter=0, deptht=0)
    #print(ds_processed.time)
    # https://docs.dask.org/en/latest/diagnostics-distributed.html
    print(ds_processed)
    print_graph(ds_processed["votemper"], "processed", date)
    ds_processed["votemper"].data.visualize(filename='graph_{}.png'.format(date), optimize_graph=True, 
                        color="order",cmap="autumn", node_attr={"penwidth": "4"})
    with performance_report(filename="dask-report.html"):
        ds_processed.to_zarr(os.path.join(path_out, "mean_"+date+".zarr"), mode="w")

    print("Congrats, processing is over !")

