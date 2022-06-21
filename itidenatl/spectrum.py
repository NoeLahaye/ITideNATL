""" basically wrappers around xrft routines """

import numpy as np
import xrft
from .tools import misc as ut
#from . import gridop as gop

### local dict / variables / parameters
# common to xy and t
_spectrum_dico = dict(detrend='linear', scaling="spectrum",
                     window="hamming", window_correction=True,
                     mask = ["tmask", "umask", "vmask", "tmaskutil", "umaskutil", "vmaskutil"]
                    )
# specific to xy
_spectrum_xy_dico = _spectrum_dico.copy()
_spectrum_xy_dico.update(dict( barlett = False,
                               truncate = True,
                               calc_scaled_k = True,
                               mask_threshold = None,
                               interp_k = True
                         ))

### time spectra
_spectrum_t_dico = _spectrum_dico.copy()
_spectrum_t_dico.update(dict( barlett = True,
                              dim = "t"
                       ))

### other
_map_var_spectrum = dict(ssh = "sossheig",
                         hke = [["vozocrtx", "vomecrty"], ["sozocrtx","somecrty"]]
                        )

_stacker = xrft.xrft._stack_chunks


######################################################################################
####################  - - -  time spectra  - - -  ####################################
######################################################################################

def spectrum_1D_t(da, **kwargs):
    """
    wrapper around xrft.power_spectrum
    """
    dico = ut.parse_inp_dict(kwargs, _spectrum_t_dico)
    # fix dimension
    dim = dico["dim"]
    if dim not in da.dims:
        dim = next(d for d in da.dims if d.startswith(dim))
        print("processing along dimension", dim)

    # get mask
    mask = dico.get("mask", dico.get("mask_name"))
    if isinstance(mask, str):
        mask = da[mask]
    elif isinstance(mask, list):
        mask = da[next(d for d in mask if d in da.coords)]

    psd_key = ["window", "scaling", "detrend", "window_correction"]
    opts = {k:dico[k] for k in psd_key}
    chks = da.chunks[da.dims.index(dim)]
    if not dico["barlett"] and len(chks)>1:
        da = da.chunk({dim:-1})
    elif np.unique(chks).size>1:
        print(f"Warning: removing {chks[-1]} last elements for even chunking")
        da = da.isel(t=slice(0,-chks[-1]))
    opts["chunks_to_segments"] = dico["barlett"]

    psd = xrft.power_spectrum(da.where(mask, 0.), dim=[dim], real_dim=dim, **opts
                             ).where(mask)

    if dico["barlett"]:
        psd = psd.mean(f"{dim}_segment")
    psd = psd.assign_coords(cpd = psd[f"freq_{dim}"] * 3600*24.)
    return psd
