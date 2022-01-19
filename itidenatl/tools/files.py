""" routines related to data files -- WARNING system (occigen) dependent """
from pathlib import Path
import numpy as np
import pandas as pd

import itidenatl.tools.files as uf # old version

# ---------------------------- params, misc ------------------------------------
# I think this should be different
vmapping = dict(gridT="votemper",
                gridS="vosaline",
                gridU="vozocrtx",
                gridV="vomecrty",
                gridT2D="sossheig", 
                gridU2D="sozocrtx",
                gridV2D="somecrty" # ignores all other variables for now
                )
# ---------------------------- paths -------------------------------------------

raw_data_dir = "/work/CT1/hmg2840/lbrodeau/eNATL60/"
work_data_dir = "/work/CT1/ige2071/SHARED/"
scratch_dir = "/work/CT1/ige2071/SHARED/scratch/"
dico_path = "/home/nlahaye/Coding/ITideNATL/itidenatl/dico_data_path.json"

# ---------------------------- raw netcdf  -------------------------------------

def make_list_files(data_path=Path(raw_data_dir), i_days=None):
    """ not sure this work if i_days is not None, it might take the order of files """
    subs = "eNATL60-BLBT02*-S/????????-????????/eNATL60-BLBT02*_1h_*_gridS_*.nc"
    list_files = list(data_path.glob(subs))
    if i_days is not None:
        i_days = list(i_days)
        list_files = [list_files[i] for i in i_days]
    return list_files

def get_list_files(dico_path=dico_path, i_days=None):
    df = pd.read_json(dico_path).full_path.apply(lambda x: Path(x))
    if i_days is None:
        return df.to_list()
    else:
        if isinstance(i_days, int):
            i_days = [i_days]
        return df[i_days].to_list()

def get_dico_files(dico_path=dico_path, i_days=None):
    """ construct a dict with date:path for i_days """
    df = pd.read_json(dico_path)
    if i_days is not None:
        df = df.iloc[i_days]
    dico_files = { dd["date"]:Path(dd["full_path"]) 
                        for ii,dd in df[["date","full_path"]].iterrows()
                 }
    #list_files = get_list_files(data_path=data_path, i_days=i_days)
    #dico_files = {k.name.rstrip(".nc")[-8:]:k for k in list_files} # dico day:path
    return dico_files

def get_date_from_iday(i_days=None, dico_path=dico_path): #data_path=Path(raw_data_dir)):
    """
    return all dates sorted if i_days is None, dates att day # i_days if i_days is in or list of int
    format yyymmdd

    Parameters:
    ___________
    i_days: int or list (optional)

    Returns:
    _______
    str or list of str with dates sorted
    """
    
    #list_files = get_list_files(data_path=(Path(raw_data_dir)))
    #dates = [k.name.rstrip(".nc")[-8:] for k in list_files] # list of dates (day)
    #dates.sort()
    #if i_days is not None:
        #if isinstance(i_days, int):
            #dates = dates[i_days]
        #elif isinstance(i_days, list):
            #dates = [dates[i] for i in i_days]
    #return dates
    df = pd.read_json(dico_path).date
    if i_days is None:
        return df.to_list()
    elif isinstance(i_days, list):
        return df[i_days].to_list()
    else:
        return df[i_days]

def get_eNATL_path(var=None, its=None, dico_path=dico_path): #data_path=Path(raw_data_dir)):
    """ get path of eNATL raw data given a variable name and time instants (days) 
    Parameters
    __________
    var: str (optional)
        variable name (NEMO OPA name)
        return parent directories if not provided
    it: int or list of int (optional)
        date (day of simulation). Returns all available date if not provided
    data_path: str or pathlib.Path object (optional)
        parent directory for all simulation data (default: utils.raw_data_dir)
    """

    #dates = get_date_from_iday(data_path=data_path)
    #dico_files = get_dico_files(data_path=data_path)
    dates = get_date_from_iday(dico_path=dico_path)
    dico_files = get_dico_files(dico_path=dico_path)
    
    ### utilitary function to get file corresponding to one time index and one variable
    map_varname = {v:k for k,v in uf.vmapping.items()}
    #if map_varname["sossheig"]=="gridT2D":
        #map_varname["sossheig"] = "gridT-2D"
     
    if isinstance(its, (list, np.ndarray)):
        res = []
        for it in its:
            path = dico_files[dates[it]]
            if var is None:
                name = ""
            else:
                name = path.name.replace("gridS", map_varname[var].replace("2D","-2D"))
            res.append(path.parent/name)
    elif isinstance(its, int):
        path = dico_files[dates[its]]
        if var is None:
            name = ""
        else:
            name = path.name.replace("gridS", map_varname[var].replace("2D","-2D"))
        res = path.parent/name
    else:
        res = []
        for da in dates:
            path = dico_files[da]
            if var is None:
                name = ""
            else:
                name = path.name.replace("gridS", map_varname[var].replace("2D","-2D"))
            res.append(dico_files[da].parent/name)
    return res
        
