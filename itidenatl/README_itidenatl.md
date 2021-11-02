# itidenatl

library containing python routines and class for performing a vertical mode projection on eNATL60 data

### Contains
* **eos.py:** equation-of-state related functions, for computing the density and the Brunt-Väisälä frequency. The routines are borrowed from the [CDFTOOLS](https://github.com/meom-group/CDFTOOLS) fortran package
* **gridop.py:** grid-related routines: computing the bottom depth, the vertical grid spacing (with correction associated with the SSH)...
* **nemodez.py:** class `Vmodes` and associated routines and method for computing the vertical modes and doing the projection on these vertical modes. Some stuff are deprecated or need to be implemented, but basically works
* **utils.py:**: all kind of practical routines (dealing with path, dask clusters, tasks, reading data...) 
* **vars.py:** for computing some variables (pressure only for now)

Non-python stuff:

* **dico\_data\_path.json:** python dictionary stored as a json file (generated using `make_dico_path.ipynb` and used in e.g. `utils.py`)