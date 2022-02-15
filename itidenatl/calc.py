import xarray as xr
from .tools.misc import _is_complex
from .gridop import _has_metrics
from xgcm import Grid

_flux_names = {"p":"p", "u":"u", "v":"v", "h":"hbot"}
_flux_dico = {"vavg": True, "sep_contrib": True, "var_names": _flux_names}

def _parse_calc_flux_args(ds_or_da, **kwargs):
    """ routine for parsing arguments of calc_flux and calc_div_flux routines """
    dico = _flux_dico.copy()
    dico.update(kwargs)
    if isinstance(ds_or_da, xr.Dataset):
        names = {k:dico["var_names"].get(k, _flux_names[k]) for k in _flux_names.keys()} # handle incomplete var_names dict
        dap = ds_or_da[names["p"]]
        dau = ds_or_da[names["u"]]
        dav = ds_or_da[names["v"]]
        hbot = dico["hbot"] if "hbot" in dico else ds_or_da[names["h"]]
        xgrid = dico["xgrid"] if "xgrid" in dico else Grid(ds_or_da)
    elif isinstance(ds_or_da, xr.DataArray):
        if ("hbot" not in dico and not dico["vavg"]) or "xgrid" not in dico:
            raise ValueError("please pass hbot and xgrid if using DataArrays")
        dap = ds_or_da
    return dap, dau, dav, hbot, xgrid

def calc_flux(ds_or_da, **kwargs):
    """ compute horizontal modal energy flux (H) p u
    
    Parameters:
    ___________
    ds_or_da: xarray.Dataset or xarray.DataArray
        input dataset containing p, u, v, hbot (use "var_names" dico for other naming convention")
        or input DataArray with pressure field, in which case u, v and hbot must be passed as separate datararrays
    hbot: xr.DataArray, optional
        bottom topography
    xgrid: xr.DataArray, optional
        xgcm Grid object associated with ds. Must be passed if using DataArrays
    vavg: bool, optional (default: False)
        wether the returned flux corresponds to vertically averaged flux (otherwise, vertically integrated flux is returned)
    dau: xr.DataArray, optional
        horizontal zonal velocity
    dav: xr.DataArray, optional
        horizontal meridional velocity
   
    Returns:
    ________
    xarray Dataset with the zonal and meridional of the vertically averaged/integrated energy flux Fx, Fy
    """
    
    _itp_kwg = dict(boundary="extrapolate")
    dap, dau, dav, hbot, xgrid = _parse_calc_flux_args(ds_or_da, **kwargs)
    vavg = kwargs.get("vavg", _flux_dico["vavg"])
    cpamp = _is_complex(dap)
    
    dr = xr.Dataset()
    if cpamp:
        dap = dap.conj()
    dr["Fx"] = xgrid.interp(dau, "X", **_itp_kwg) * dap
    dr["Fy"] = xgrid.interp(dav, "Y", **_itp_kwg) * dap
    if dap.dtype in ["complex64", "complex128"]:
        dr = 2 * dr.real
    if not vavg:
        dr = dr * hbot
    return dr

def calc_div_flux(ds_or_da, **kwargs):
    """ compute divergence of horizontal modal energy flux div(H*p*u) (/H)
    
    Parameters:
    ___________
    ds_or_da: xarray.Dataset or xarray.DataArray
        input dataset containing p, u, v, hbot (use "var_names" dico for other naming convention")
        or input DataArray with pressure field, in which case u, v and hbot must be passed as separate datararrays
    hbot: xr.DataArray, optional
        bottom topography
    xgrid: xr.DataArray, optional
        xgcm Grid object associated with ds. Must be passed if using DataArrays
    vavg: bool, optional (default: False)
        wether the returned flux corresponds to vertically averaged flux (otherwise, vertically integrated flux is returned)
    sep_contrib: bool, optional (default: True)
        compute contributions from divergence of p*u and gradient of topography separatly
    dau: xr.DataArray, optional
        horizontal zonal velocity
    dav: xr.DataArray, optional
        horizontal meridional velocity
   
    Returns:
    ________
    xarray Dataset with the divergence of the vertically averaged/integrated energy flux divF, 
        plus separate components div(pu) and p u\cdot\grad(H) if sep_contrib is True
    """
    _itp_kwg = dict(boundary="extrapolate")
    _dif_kwg = dict(boundary="extrapolate")
    dap, dau, dav, hbot, xgrid = _parse_calc_flux_args(ds_or_da, **kwargs)
    vavg = kwargs.get("vavg", _flux_dico["vavg"])
    sep_contrib = kwargs.get("sep_contrib", _flux_dico["sep_contrib"])
    cpamp = _is_complex(dap)
    diff = xgrid.derivative if _has_metrics(xgrid) else xgrid.diff
        
    dr = xr.Dataset()
    if cpamp:
        dap = dap.conj()
    if sep_contrib:
        dx_pu = diff(xgrid.interp(dap, "X", **_itp_kwg) * dau, "X", **_dif_kwg)
        dy_pv = diff(xgrid.interp(dap, "Y", **_itp_kwg) * dav, "Y", **_dif_kwg)
        pu_dh = xgrid.interp(dau * diff(hbot, "X", **_dif_kwg), "X", **_dif_kwg) * dap
        pv_dh = xgrid.interp(dav * diff(hbot, "Y", **_dif_kwg), "Y", **_dif_kwg) * dap
        if vavg:
            dr["dpuv"] = dx_pu + dy_pv
            dr["puvdh"] = (pu_dh + pv_dh) / hbot
        else:
            dr["dpuv"] = (dx_pu + dy_pv) * hbot
            dr["puvdh"] = pu_dh + pv_dh
        dr["divF"] = dr.dpuv + dr.puvdh
    else:
        divf = diff(xgrid.interp(dap * hbot, "X", **_itp_kwg) * dau, "X", **_dif_kwg) \
                + diff(xgrid.interp(dap * hbot, "Y", **_itp_kwg) * dav, "Y", **_dif_kwg)
        if vavg:
            dr["divF"] = divf / hbot
        else:
            dr["divF"] = divf
    if vavg:
        dr.attrs.update({"vert_kind":"averaged"})
    else:
        dr.attrs.update({"vert_kind":"integrated"})
    if cpamp:
        dr = 2 * dr.real
    return dr