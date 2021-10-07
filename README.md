# Internal Tide analysis in NEMO run eNATL60

## Organisation of the repository
* docs: working documents and notes
* training: folder for trials et al (sandbox for root directory)
* itidenatl: core library containing useful objects/methods
* pre-processing: computing mean fields (temperature, salinity, SSH) prior to mean stratification and vertical modes computation
* processing: compute mean stratification, mean grid and vertical modes + project variables on vertical modes
* post-processing: analyse modally projected variables

Warning: some processing notebooks (in clean form) can be at the root of the repository.

### preprocessing

The temporal average is performed in three steps:

- computation of daily means with `preprocessing/daily_mean/daily_mean.sh`
- computation of monthly means with `preprocessing/average_daily_mean/average_daily_means.sh`
- computation of the global mean with `preprocessing/final_mean/final_mean.sh`

### processing

 - `pre-proj/` contains notebooks for computing the ean stratification, computing/creating the mean grid zarr store 
 - `vmodes` contains the notebooks for computing the vertical modes
 -  `projection` contains scripts for projecting the variables on the vertical modes (at this stage: pressure and horizontal velocity)

### post-processing

 - `analysis` contains a few scripts for post processing, plotting, etc.