""" timeop.py: various time operation for analysis of eNATL60 outputs
time filtering (wrapping IIR scipy.signal filters through xarray.aply_ufunc) and complex demodulation
WARNING: in the routines, not all configurations of input optional parameters have ben tested
NJAL May 2022
"""
import numpy as np
import scipy.signal as sig
import xarray as xr
from .tools import misc as ut

###########################   - - -  Utilitary  - - -   ########################
def datetime_to_dth(ds_or_da, t_name="t", t_ref=None, it_ref=0):
    """ convert datetime series to elapsed time in hour
    reference time is either passed explicitly or taken at given index
    
    Parameters
    _________
    ds_or_da: xr.Dataset or xr.DataArray
        dataset containing time or directly dataarray of type datetime

    t_name: str, optional (default: "t")
        name of time datarray if ds_or_da is a dataset

    t_ref: np.datetime object, optional
        time reference

    it_ref: int, optional (default: 0)
        ignored if t_ref is provided. Otherwise, time reference will be the "it_ref"th element in the time dataarray.

    Returns
    _______
    xr.DataArray of dtype float containing ellapsed time in hours

    """
    da = ut._ds_to_da(ds_or_da, t_name)
    if t_ref is None:
        t_ref = da[it_ref]
    elif isinstance(t_ref, str):
        t_ref = np.datetime64(t_ref)
    dth = (da - t_ref).dt
    dth = dth.days*24. + dth.seconds/3600. 
    dth.attrs = {"t_ref": np.datetime_as_string(t_ref, unit="s"), # todo convert to string
                 "long_name": "time ellapsed since t_ref",
                 "units":"h"
                }
    return dth

#######################  - - -   time filtering   - - -  #######################
_dico_iirfilt = dict(order=4, dim="t", routine="sosfiltfilt", 
                    subsample=False, offset="auto")

def iir_filter_wrapper(da, Wn, btype, **kwargs):
    """ wrapper for dask/xarray of scipy.signal Infinite Impulse Response filtering routines
    this will apply butterworth filter using scipy.signal.filtfilt or scipy.signal.sosfiltfilt (default)
    Uses xarray.apply_ufunc and scipy.signal functions -- see corresponding doc
    N.B.: there must be only one chunk along the filter axis: beware of chunking and memory issues
    Optionally, subsampling can be applied before returning the result (e.g. for lowpass filtering)
    Internally uses _dico_iirfilt for default values of keyword parameters

    Parameters
    __________
    da: xr.DataArray
        time series to filter
    Wn: float or 2-elements array
        Nyquist frequency (filter cutoff in dimensions of samples)
    btype: str
        type of filter ("lowpass", "bandpass" or "highpass"). Forgotten "pass" is tolerated

    Other Parameters
    _______________
    order: int
        filter order (default: 4)
    dim: str
        dimension along which to apply the filter (default: "t")
    routine: str, optional
        which filtering routine to use (default: sosfiltfilt)
    subsample: bool or int
        whether to subsample the result along the filtering axis. Default is False. int type will subsample with a step of value "subsample" 
    offset: str or int
        is subsampling, what offset to apply for the first element. Default ("auto") centers the subsampled time series.
    mask: str or xr.DataArray
        name of mask to use (coordinate of da) or DataArray containing mask. If not provided, will try to find a mask in the coordinates.

    Returns
    _______
    DataArray containing the (subsampled) filtered time series 

    See also
    ________
    iir_filter: convinence wrapper that calls this routine
    """

    dico = ut.parse_inp_dict(kwargs, _dico_iirfilt)
    ford, dim = dico["order"], dico["dim"]
    iax_filt = da.dims.index(dim)
    if not btype[-4:]=="pass":
        btype += "pass"
        
    # dealing with mask
    mask = dico.get("mask", None)
    if isinstance(mask, str):
        mask = da[dico[mask]]
    elif mask is None:
        masklist = [c for c in da.coords if "mask" in c] # ! WARNING not checking dims, dangerous!
        if len(masklist)>0:
            mask = da[masklist[0]]
    if mask is not None:
        data = da.where(mask, 0.)
    
    # pre-process data
    if btype[:3] != "low": # remove mean for band-pass or high-pass filtering (maybe useful near edges? -- actually not sure...)
        data = data - data.mean(dim)
    # kwargs passed to apply_ufunc
    au_kwargs = dict(dask="parallelized", output_dtypes=[data.dtype])
    
    # subsample directly the numpy arrays
    if dico["subsample"]:
        slit = ut.get_subsample_slice(da[dim].size, dico["subsample"], dico["offset"])
    else:
        slit = slice(0,None)
        
    # design filter and choose routine
    if dico["routine"] == "sosfiltfilt":
        sos = sig.butter(ford, Wn, btype=btype, output="sos")
        fun = lambda x: sig.sosfiltfilt(sos, x, axis=iax_filt, padtype="even")
    elif dico["routine"] == "sfiltfilt":
        bb, aa = sig.butter(ford, Wn, btype=btype, output="ba")
        fun = lambda x: sig.filtfilt(bb, aa, x, axis=iax_filt, method="gust")

    res = xr.apply_ufunc(fun, data.chunk({dim:-1}), **au_kwargs).isel({dim:slit})
    if dico["subsample"]:
        res.attrs.update({k:dico[k] for k in ("subsample","offset")})
    if mask is not None:
        res = res.where(mask)
    res.attrs.update(dict(filt_routine=dico["routine"]))
    app = {"lowpass":"lf", "bandpass":"bp", "highpass":"hf"}[btype]
    if da.name is not None:
        res = res.rename(da.name+"_"+app)
    return res
        
def iir_filter(da, btype, fcut=1., fwidth=.2, dt=1., **kwargs):
    """ 1D Infinite Impulse Response filter an xarray.DataArray
    This routine calls iir_filter_wrapper, see the corresponding doc for further explanations
    N.B.: there must be only one chunk along the filter axis: beware of chunking and memory issues
    Internally uses _dico_iirfilt for default values of keyword parameters

    Parameters
    __________
    da: xr.DataArray
        time series to filter
    btype: str
        type of filter ("lowpass", "bandpass" or "highpass"). Forgotten "pass" is tolerated
    fcut: float, default=1.
        frequency cutoff
    fwidth: float, optional
        relative bandwith for bandpass filter (ignored for other types of filters). Default: .2. 
        Bandwidth is fcut*fwidth
    dt: float, default=1.
        time step of input time series

    Other Parameters
    ________________
    order: int
        filter order (default: 4)
    dim: str
        dimension along which to apply the filter (default: "t")
    routine: str, optional
        which filtering routine to use (default: sosfiltfilt)
    subsample: bool or int
        whether to subsample the result along the filtering axis. Default is False. int type will subsample with a step of value "subsample" 
    offset: str or int
        is subsampling, what offset to apply for the first element. Default ("auto") centers the subsampled time series.
    mask: str or xr.DataArray
        name of mask to use (coordinate of da) or DataArray containing mask. If not provided, will try to find a mask in the coordinates.

    Returns
    _______
    DataArray containing the (subsampled) filtered time series 

    See also
    ________
    iir_filter_wrapper: convinence wrapper that calls this routine
    """

    Wn = 2. * dt * fcut
    if btype[:4]=="band":
         Wn += Wn * fwidth/2. * np.array([-1, 1]) # algebraic symmetry
         Wn *= np.array([ (1+fwidth/2.)**(-1), 1+fwidth/2. ]) # geometric symmetry
    dico = ut.parse_inp_dict(kwargs, _dico_iirfilt)
    res = iir_filter_wrapper(da, Wn, btype, **dico)
    res.attrs.update(dict(filt_btype=btype, filt_fcut=fcut, filt_tcut=1./fcut))
    if btype[:4]=="band":
        res.attrs["filt_bandwidth_abs"] = fwidth * fcut
        res.attrs["filt_bandwidth_rel"] = fwidth 
    return res

#######################  - - -   complex demodulation   - - -  #######################
_dico_demod = dict(coord=None, dim="t", fcut_rel=1./6, 
                    tref=np.datetime64("2009-06-30T00:30"),
                    subsample=False, subsample_rel=1./3, offset="auto",
                    interp_reco_method="quadratic"
                  )

def complex_demod(da, fdemod, **kwargs):
    """ perform complex demodulation of a DataArray, running a lowpass filter on 
    the time series multiplied by the phase term at minus the targeted frequency.
    The result is multiplied by 2, so that the reconstructed signal is Re(A_cpdmod * exp(i*omega*(t-tref))
    
    Internally uses _dico_demod for optional parameters (plus default parameters for iir_filter)

    Parameters
    __________
    da: xr.DataArray
        time series on which to perform complex demodulation
    fdemod: float
        frequency of complex demodulation in cycle/hour

    Other Parameters
    ________________
    fcut: float, optional.
        frequency cutoff of low-pass filter, equivalent to half the bandwidth of bandpass filter around the demodulation frequency. Will use fcut_rel if not specified
    fcut_rel: float, default=1/6
        cutting frequency of low-pass filter relative to the demodulation frequency
    dim: str, default: "t"
        dimension along which to perform the operation 
    coord: str, optional
        coordinate to use to compute ellpased time in hours. Uses "dim" if not provided.
    tref: str or np.datetime64 or int
        reference time for computing ellapsed time in hour. str will be converted into datetime64 object (hence it must be parseable by np.datetime64). If int, uses the corresponding position in the input time series.
    subsample: bool or int, optional
        whether to subsample the result along the filtering axis. Default is False. int type will subsample with a step of value "subsample" 
    offset: str or int, default: "auto"
        is subsampling, what offset to apply for the first element. Default ("auto") centers the subsampled time series.
    **kwargs:
        arguments passed to iir_filter routine

    Returns
    _______
    DataArray containing the (subsampled) complex demodulated amplitude time series

    See also
    ________
    iir_filter: filtering routine used for lowpass filtering
    reco_cpdmod: routine for reconstructing time-series from complex demodulated amplitude
    """
    dico = ut.parse_inp_dict(kwargs, _dico_demod)
    dim, tref = dico["dim"], dico["tref"]
    coord = dim if dico["coord"] is None else dico["coord"]
    fcut = dico.get("fcut", dico["fcut_rel"]*fdemod)

    if isinstance(tref, int):
        dth = datetime_to_dth(da[coord], it_ref=tref)
    else:
        if isinstance(tref, str):
            tref = np.datetime64(tref)
        dth = datetime_to_dth(da[coord], t_ref=tref)
        
    omt = 2*np.pi * fdemod * dth
    dt = dth[:2].diff(dim).values[0]
    kwgs = dict(fcut=fcut, dt=dt)

    subsample = dico["subsample"]
    if subsample:
        if not isinstance(subsample, int):
            subsample = int(round(dico["fsubsample_rel"]/fcut))
        kwgs.update(subsample=subsample, offset=dico["offset"])
    out_dtype = f"complex{int(str(da.dtype)[-2:])*2}"
    
    prov = (2 * da * np.exp(-1.j*omt)).assign_coords(t_ellapse=dth.reset_coords(drop=True))
    res = iir_filter(prov, btype="low", **kwgs).astype(out_dtype)
    res.attrs.update({"demod_freq":fdemod, "t_ref":dth.t_ref})
    return res.rename(da.name+"_cpdmod")

### reconstruct time series (equivalent to bandpass filtering)
def reco_cpdmod(da, newt=None, newdth=None, **kwargs):
    """ reconstruct time series from complex demodulated amplitudes, computing Re(da * exp(1.j*omega*t)
    potentially interpolate on a finer grid prior to reconstructing the time series (e.g. for reconstructing from a downsampled time series)
    
    Internally uses _dico_demod for optional parameters 

    Parameters
    __________
    da: xr.DataArray
        time series of complex demodulated amplitude
        Must have the following coordinates and attributes: 
         - t_ref (reference time), if newdth is not passed
         - demod_freq: frequency demodulation
    newt: str or xr.DataArray, optional
        time array (or name of corresponding coordinate in da) on which to evaluate/interpolate the reconstructed time series. If none is provided, will not interpolate. Must match with time dimension (same type, same range of value) for interpolation.
    newdth: str or xr.DataArray, optional
        array of ellapsed timefor evaluating the phase term. If none is provided, will compute it freom newt (if interpolating) or search for "t_ellapse" coordinate in da (no interpolation).
    method: str, default: "quadratic"
        interpolation method. Use value in interp_reco_method if not provided. Ignored if no interpolation
    interp_reco_method: str, default: "quadratic"
        interpolation method. Ignored if no interpolation
        frequency of complex demodulation in cycle/hour

    Other Parameters
    ________________
    dim: str, default: "t"
        dimension along which to perform the operation 

    Returns
    _______
    DataArray containing the reconstructed time series

    See also
    ________
    complex_demod: routine for performing complex demodulation
    """
    dico = ut.parse_inp_dict(kwargs, _dico_demod)
    tref, dim = da.t_ref, dico["dim"]
    if newt is not None:
        method = dico.get("method", dico["interp_reco_method"])
        if newdth is None:
            newdth = datetime_to_dth(newt, t_ref=tref)
        newda = da.interp({dim:newt}, assume_sorted=True, 
                          method=method, kwargs={"fill_value":"extrapolate"}
                         )
    else:
        newda = da
        if newdth is None:
            if "t_ellapse" in da.coords:
                newdth = da.t_ellapse
            else:
                newdth = datetime_to_dth(da[dim], t_ref=tref)
        else:
            if isinstance(newdth, str):
                newdth = da[newdth]
            
    exp_omt = np.exp(2.j*np.pi * da.demod_freq * newdth).astype(da.dtype)
    reco = ( newda * exp_omt ).real.rename(da.name.replace("cpdmod", "cpdrec"))
    reco.attrs = da.attrs.copy()
    return reco
