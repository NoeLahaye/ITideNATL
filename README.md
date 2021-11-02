# Internal Tide analysis in NEMO run eNATL60

This repo contains python scripts and notebooks for processing and analysing outputs from the eNATL60 simulation. 
This analysis involves a vertical mode decomposition of the dynamics, and heavily uses xarray/dask.

requirements: [xorca](https://github.com/willirath/xorca) library + installation of "itidenatl" library (running `python setup.py instal`)

## Organisation of the repository
* **docs/**: working documents and notes
* **dev/**: folder for trials et al (sandbox for root directory)
* **itidenatl/**: core library containing objects/methods
* **preprocessing/**: computing mean fields: temperature, salinity, SSH. (prior to mean stratification and vertical modes computation)
* **processing/**: compute mean stratification, mean grid and vertical modes + project variables on vertical modes
* **postpro_ana/**: analysis of modally projected variables
* **misc/**: side stuff (e.g. plotting topo, dealing with path, etc.)

Further information is given in each sub-directory (search for `README_*.md` files)

