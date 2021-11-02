## Projection of dynamical fields onto vertical modes
this part of the processing consists of loading the vertical mode and associated grid on the one hand, and the dynamical fields on the other hand, and computing the projection (vertical integration). 

#### Strategy:
* for the pressure, density is computed from T and S, and pressure results of a vertical integration of the latter. Correction for grid variations (grid breathing) are computed (using the density for $\partia_z p$, and using a vertical derivative for the horizontal velocities).
* computation is run sequentially along a time and lat loop, to limit memory usage. This allows to use only one (two for the pressure) nodes, because I did have bad scaling performance. Probably this could be fixed, and the loop can be discarded.

#### Scripts: 
* `proj_pres_ty-loop.py` and `proj_uv_ty-loop.py`: python scripts for doing the computation. `proj_utils.py` contains a few utilitary routines (and wrappers)
* `job_proj_?.sh`: PBS job submission scripts for Occigen
* `launch_tasks.py` python script to generate and launch jobs

#### Remarks
* Execution speed decreases sometimes, for potentially several reasons. I was not able to understand precisely the problem, but:
	* It looks like having the same grid/vmodes zarr store shared by different processes is one cause for this
	* Maybe having different processes reading in the same folder is another source

For this reason, I rhave three duplicated grid/vmodes files and launch jobs sequentially within each eNATL60 output folder, three folders (sequence of jobs) in parallel.

#### TODO: 
* implement a script to monitor the timing of reading data, stop the jobs if execution speed is decreasing and launch them only if reading time is enough. cf `test_reading_timing.ipynb`.