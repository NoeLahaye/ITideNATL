Create mean grid and mean stratification, then compute the vertical modes and interpolate them horizontally on `u, v` points.

inputs: time-averaged temperature, salinity and SSH. There is an assumption here, which is that the mean density is well approximated by $\rho(\langle T\rangle, \langle S \rangle)$. For sure, the error on the vertical modes computed using the resulting mean $N^2$ is negligible.

### Processing chain for computing vertical modes
1. **Compute mean grid** with `comp_mean_grid.ipynb`. Inputs: NEMO vertical grid file, time-averaged SSH. Output: mean grid zarr archive `eNATL60_mean_grid_z.zarr` with masks and metrics + hbot and ssh.
2. **Compute mean Brunt-Vaisala frequency** with `compute_mean_stratif.ipynb`. Inputs: mean grid file, mean temperature and mean salinity. Output: zarr archive with mean $N^2$ profile (e.g. `eNATL60_mean_bvf.zarr`).
3. **Compute vertical modes** with `compute_vmodvizes.ipynb`. Inputs: vertical grid (and metrics) and stratification. Output: zarr store containing vertical modes ($\phi, \phi'$ and $c$).

### Further processing for field projection
1. **re-chunk** (and gather necessary fields from the mean grid) using `make_projpres_gridmode_dataset.ipynb`
2. **interpolate on u-v points** (with adequate chunking) using `interp_vmode_phi_uv.ipynb`

### Size of datasets
* `vmodes_10.zarr`: 226 GB
* `phi_u_10.zarr`, `phi_v_10.zarr`: 125 GB
* mean vertical grid `eNATL60_mean_grid_z.zarr`: 60 GB
* specific grid & mode file for pressure projection `eNATL60_grid_vmodes_proj_pres.zarr`: 133 GB
* `global_mean_bvf.zarr`: 10 GB