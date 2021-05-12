import os, sys

import numpy as np
import xarray as xr

from dask.distributed import performance_report

import itidenatl.utils as ut

if __name__ == '__main__':

    cluster, client = ut.spin_up_cluster("local", n_workers=7)
    print(client)

    # collect arguments
    file_in = sys.argv[1]
    variable = sys.argv[2]
    output_dir = sys.argv[3]

    # flag for graph outputs
    graph=False

    # variable key in datasets
    vkey = ut.vmapping[variable.replace("-","")]

    print("File being processed: "+file_in)

    _chunks = dict(time_counter=-1, y=400, x=-1)
    if variable!="gridT-2D":
        _chunks["deptht"] = 1

    ds = xr.open_dataset(file_in, chunks=_chunks)

    # auto chunks leads to chunks that are 24, 1182, 1182, i.e. 33530976 points
    # 33530976/8354 = 4013
    # data variables are 24 x 300 x 4729 x 8354

    # drop redundant variables
    ds = ds.drop_vars(["nav_lon", "nav_lat"])

    # check log file existence and exit if True
    date = str(ds["time_counter"].dt.strftime("%Y%m%d")[0].values)
    log_file = ut.get_log_file(output_dir, "daily_mean_"+variable+"_"+date)
    if os.path.isfile(log_file):
        print(" File {} exists, skiping".format(log_file))
        sys.exit()

    ut.print_graph(ds[vkey], "open_"+date, graph)

    # debug
    #ds = ds.isel(deptht=slice(0,5))
    #ut.print_graph(ds[vkey], "isel_"+date, graph)

    # temporal average
    ds_processed = ds.mean("time_counter")
    ds_processed = ds_processed.chunk({"y":-1})
    #
    print(ds_processed)
    ut.print_graph(ds_processed[vkey], "processed_"+date, graph)

    #with performance_report(filename="dask-report.html"):
    zarr_archive = "daily_mean_"+variable+"_"+date+".zarr"
    ds_processed.to_zarr(os.path.join(output_dir, zarr_archive), mode="w")

    # create empty file to indicate processing was completed
    with open(log_file, "w+") as f:
        pass

    print("Congrats, processing is over !")
