""" timeop.py: various time operation for analysis of eNATL60 outputs
 - time filtering (wrapping IIR scipy.signal filters through xarray.aply_ufunc) 
 - complex demodulation
 - correlation
 - harmonic fit

WARNING: in the routines, not all configurations of input optional parameters have ben tested
NJAL May 2022
"""
import numpy as np
import scipy.signal as sig
from scipy.optimize import least_squares
import xarray as xr
from .tools import misc as ut

### local constants, dict, etc.

_freq_cpdmod = 1. / 12.2 # complex demodulation frequency, cph
_tide_period_h = {"M2": 12.42, "S2": 12.00, "N2": 12.66, "K2": 11.96}
_fcomp = ["M", "S", "N", "K"]
_t_ref = np.datetime64("2009-06-30T00:30:00")

def get_delom(om, om_ref=2*np.pi*_freq_cpdmod):
    """ difference between 2 frequencies """
    return om - om_ref

_delom_dict = {k: get_delom(2*np.pi/_tide_period_h[f"{k}2"]) for k in _fcomp}

_dico_iirfilt = dict(order=4, dim="t", routine="sosfiltfilt", 
                    subsample=False, offset="auto")

_dico_demod = dict(coord=None, dim="t", fcut_rel=1./5, 
                    tref=_t_ref,
                    subsample=False, subsample_rel=1./3, offset="auto",
                    interp_reco_method="quadratic"
                  )

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

#######################  - - -   detrending   - - -  #######################
def detrend_dim(da, dim, deg=1):
    """detrend along a single dimension
    taken from https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    modify to take into account complex values -- fitting real and imag separately
    but why da.polyfit does not work with complex values, while np.polyfit does? """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    res = da - fit
    if da.dtype.kind == "c":
        p_i = da.imag.polyfit(dim=dim, deg=deg)
        fit_i = xr.polyval(da[dim], p_i.polyfit_coefficients)
        res -= 1.j * fit_i
    return res


#######################  - - -   time filtering   - - -  #######################

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
    else:
        data = da
    
    # pre-process data
    if btype[:3] != "low": # remove mean for band-pass or high-pass filtering (maybe useful near edges? -- actually not sure...)
        data -= data.mean(dim)
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
        fractional bandwith for bandpass filter (ignored for other types of filters). Default: .2. 
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
         #Wn *= np.array([ (1+fwidth/2.)**(-1), 1+fwidth/2. ]) # geometric symmetry
    dico = ut.parse_inp_dict(kwargs, _dico_iirfilt)
    res = iir_filter_wrapper(da, Wn, btype, **dico)
    res.attrs.update(dict(filt_btype=btype, filt_fcut=fcut, filt_tcut=1./fcut))
    if btype[:4]=="band":
        res.attrs["filt_bandwidth_abs"] = fwidth * fcut
        res.attrs["filt_bandwidth_frac"] = fwidth 
    return res

#######################  - - -   complex demodulation   - - -  #######################

def complex_demod(da, fdemod=_freq_cpdmod, **kwargs):
    """ perform complex demodulation of a DataArray, running a lowpass filter on 
    the time series multiplied by the phase term at minus the targeted frequency.
    The result is multiplied by 2, so that the reconstructed signal is Re(A_cpdmod * exp(i*omega*(t-tref))
    
    Internally uses _dico_demod for optional parameters (plus default parameters for iir_filter)

    Parameters
    __________
    da: xr.DataArray
        time series on which to perform complex demodulation
    fdemod: float, optional; default=_freq_cpdmod
        frequency of complex demodulation in cycle/hour.

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


#######################  - - -   correlation & co.   - - -  #######################
def wrap_correlate(da1, da2=None, detrend=False, mode="same", maxlag=None):
    """ numpy-based, 1D wrapper of scipy.signal.correlate for computing the (cross)correlation

    if maxlag is specified and mode is "valid", compute positive lag cross-correlation with max-lag over fully overlapping windows
    no border effect.

    Result is not normalized (i.e. corr(lag=0) = mean(da1*da2.conj()) )

    Parameters
    __________
        da1: 1D array, size (N,), (M,)
        da2: 1D array, size N, optional
        detrend: bool, optional, default: False
            apply detrending before computing the correlation.
            WARNING: Won't work from within aply_ufunc
        mode: str {"valid", "same", "full"}
            mode for computing the convolution product. Default is "same"
        maxlag: int (default:None)
            maximum lag computed.

    Returns:
    ________
        1D array of size (maxlag + 1 if mode is 'valid', min(N,M) otherwise.
            cross-correlation between da1 and da2, or autocorrelation of da1
    """
    if detrend:
        da1 = sig.detrend(da1, axis=-1, type="linear")
        if da2 is not None:
            da2 = sig.detrend(da2, axis=-1, type="linear")
    if da2 is None:
        da2 = da1

    if mode == "valid" and maxlag is not None:
        da1 = da1[maxlag:]

    res = sig.correlate(da1, da2, mode=mode, method="auto")
    if mode != "valid":
        res = res[res.size//2:]
    return res


def correlation(v1, v2=None, dim="t", mode="same", **kwargs):
    """
    compute cross- (or auto-) correlation between two DataArrays v1 and v2.
    Wrapper of numpy.correlate for xarray.DataArrays using xarray.apply_ufunc.
    This implementation is not optimal since it vectorizes over every ther dimension that
    the one along which correlation is computed

    Parameters
    ----------

        v1, v2: ndarray, pd.Series
            Time series to correlate, the index must be time if coord is not provided

        dim: str (default: "t")
            name of coordinate along which correlation is performed

        mode: str, optional (default: "same")
            "same", "valid" or "full" -- see correlate or convolve docstring

        dt: float, optional
            value of increment along coord, used to compute lags
            Will use corresponding coordinate if not specified

        detrend: boolean, optional. Default: False
            Linear detrend data before computing correlation.

        maxlag: float, optional
            if mode=="valid", maximum timelag to compute.
            The first time series v1 will be cropped by the corresponding number of elements

        normalize: boolean, optional (default: False)
            normalize result by variance (otherwise by time series duration)

    Returns:
    ________
        da_corr: xr.Datarray containing the correlation function,
        with (only positive) lags as coordinate

    """

    assert dim in v1.dims
    indtype = [v1.dtype.kind]

    if v2 is not None:
        assert dim in v2.dims
        indtype.append(v2.dtype.kind)
    coord = v1[dim]

    letype = np.complex64 if "c" in indtype else np.float32

    dt = kwargs.pop("dt", None)
    if not dt:
        dt = coord.diff(dim).mean()
        if dt.dtype != "float":
            dt = float(dt.dt.seconds/3600.)
            units = "h"
        else:
            units = dt.attrs.pop("units", None)
            dt = float(dt)
    else:
        units = dt.attrs.pop("units", None)

    maxlag = kwargs.pop("maxlag", None)
    if maxlag is None:
        nlag = coord.size//2
    elif isinstance(maxlag, float):
        nlag = int(maxlag//dt) + 1
    elif isinstance(maxlag, int):
        nlag = maxlag + 1
    else:
        raise ValueError(f"unable to parse maxlag value '{maxlag}'")

    if kwargs.pop("detrend", False):
        v1 = detrend_dim(v1, dim)
        if v2 is not None:
            v2 = detrend_dim(v2, dim)

    args = (v1, v2) if v2 is not None else (v1, )
    kwgs = {"mode":mode, "maxlag":maxlag}

    res = xr.apply_ufunc(wrap_correlate, *args,
                        dask="parallelized",  vectorize=True,
                        input_core_dims=[[dim]]*len(args), output_core_dims=[["lag"]],
                        output_dtypes=[letype], kwargs=kwgs,
                        dask_gufunc_kwargs={"output_sizes":{"lag": nlag}}
                       )
    try:
        if v2 is None:
            namout = "corr_{0}".format(v1.name)
        else:
            namout = "corr_{0}-{1}".format(v1.name, v2.name)
    except:
        namout = "corr"

    res = res.assign_coords(lag = np.arange(nlag) * dt).rename(namout)
    res.lag.attrs["units"] = units

    if kwargs.pop("normalize", False):
        res /= (v1 * v2.conj()).real.sum()
    else:
        norm = coord.size
        if mode=="valid": norm -= nlag - 1
        res /= norm

    return res

#######################  - - -   harmonic fit   - - -  #######################
# for complex time series -- not tested on a real time series

### functions for performing the harmonic fit
def harmo(amp, dom, t):
    """ harmonic time series with complex amplitude 'amp',
    frequency (rad/units(h)) 'dom' and evaluated at time 't'
    returns amp * exp(i*dom*t)
    """
    return amp * np.exp(1.j*dom*t)

def _fitfunc(p, t, oms):
    res = harmo(p[0], oms[0], t)
    for i in range(1,len(p)):
        res += harmo(p[i], oms[i], t)
    return res

def _errfunc(p, t, y, oms):
    """ error between harmonic time series and input data
    work with 1D numpy arrays

    parameters
    __________
        p: iterable of complex numbers, fit parameters (complex amplitude)
        t: 1D array, time
        y: 1D array, data
        oms: iterable of float, harmonic frequencies
    """
    res = harmo(p[0], oms[0], t)
    for i in range(1,len(p)):
        res += harmo(p[i], oms[i], t)
    return res - y

def _wrap_err(p, t, y, oms):
    """ wrapper of _errfunc, going from complex numbers to 2 real time series """
    p = np.array([p[2*k] + 1.j*p[2*k+1] for k in range(len(p)//2)])
    res = _errfunc(p, t, y, oms)
    return np.r_[res.real, res.imag]

def _jac(p, t, y, oms):
    """ jacobian for the harmonic fit, casted in complex expanded to 2 real time series
    """
    Nt = t.size
    J = np.zeros((Nt*2, p.size))
    for k in range(p.size//2):
        res = harmo(1, oms[k], t)
        J[:,2*k] = np.r_[res.real, res.imag]
        J[:,2*k+1] = np.r_[-res.imag, res.real]
    return J

def _wrap_fit(yy, tt, oms, ferr, fjac):
    No = len(oms)
    if not np.isnan(yy).all():
        if False:
            p0 = np.c_[np.ones(No), np.zeros(No)].ravel()
        else:
            p0 = (yy[None,:] * np.exp(-1.j*np.array(oms)[:,None]*tt[None,:])
                 ).mean(axis=-1)
            p0 = np.c_[p0.real, p0.imag].ravel()
        try:
            res = least_squares(ferr, p0, jac=fjac, args=(tt,yy))
            pout = np.array([res.x[2*i] + 1.j*res.x[2*i+1]
                             for i in range(No)]
                           ).astype("complex64")
            rms_tot = (yy*yy.conj()).real.mean()**.5
            yr = yy - _fitfunc(pout, tt, oms)
            rms_res = (yr*yr.conj()).real.mean()**.5
        except:
            print("failed")
            pout = np.zeros(No)+0.j+np.nan
            rms_tot, rms_res = 0.j+np.nan, 0.j+np.nan
        return np.r_[pout, rms_tot, rms_res]
    else:
        return np.ones(No+2)*np.nan

def harmonic_fit(da, oms=_delom_dict, mask=None):
    """ compute harmonic fit at frequencies 'oms' on input DataArray 'da'

    wrap scipy.optimze.least_square using xr.apply_ufunc

    Parameters
    __________
        da: xr.DataArray
            input complex-value ddata time series
            must contain dimension 't' and coordinate 't_ellapse'
        oms: dict or list of float
            frequencies for harmonic fit (with labels as dict keys)

    Returns
    _______
        xr.Dataset containing the complex harmonic amplitudes,
            associated RMS and RMS of residual
    """

    if isinstance(oms, dict):
        fname = list(oms.keys())
        oms = [oms[k] for k in fname]
    else:
        fname = None

    n_out = 2+len(oms)

    da = da.chunk({"t":-1})

    ferr = lambda p,t,y: _wrap_err(p, t, y, oms)
    fjac = lambda p,t,y: _jac(p, t, y, oms)
    wrap_fit = lambda y, t: _wrap_fit(y, t, oms, ferr, fjac)

    res = xr.apply_ufunc(wrap_fit, da, da.t_ellapse,
                        input_core_dims=[["t"], ["t"]], output_core_dims=[["stack"]],
                         vectorize=True, dask="parallelized", output_dtypes="complex64",
                         dask_gufunc_kwargs=dict(output_sizes={"stack":n_out})
                        )
    if mask:
        if isinstance(mask, "str"):
            mask = da[mask]
        res = res.where(mask)

    ds = res.isel(stack=slice(0,n_out-2)).rename({"stack":"fcomp"}).to_dataset(name="cha")
    ds = ds.assign_coords(fcomp=oms)
    if fname:
        ds = ds.assign_coords(fcomp_name=xr.DataArray(fname, dims=("fcomp",)))
    ds["rms_tot"] = res.isel(stack=-2).real
    ds["rms_res"] = res.isel(stack=-1).real

    return ds

def reco_harmo(amp, t, fcomp=None, dim="fcomp"):
    """ reconstruct harmonic time series from DataArray 'amp' and time array

    Parameters
    __________
        amp: xr.DataArray with dimension "fcomp"
            harmonic amplitudes + frequency "fcomp" as coordinates
        t: xr.DataArray
            time array
        fcomp: xr.DataArray, optional
            use amp coordinate if not passed explicitly

    returns
    _______
        xr.DataArray with harmo(amp, fcomp, t).sum(dim)
    """
    if fcomp is None:
        fcomp = amp.fcomp
    return harmo(amp, fcomp, t).sum(dim)
