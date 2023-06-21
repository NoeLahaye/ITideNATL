### routine for preparing plots
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams.update({"font.size":14, "font.family":"serif"})
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import copy
import warnings
import numpy as np
from .. import gridop as gop

_domain_extent = dict(AZ = [-37.5, -23.5, 28.3, 40.8],
                      GF = [-76, -52, 34, 44.5],
                      NAtl =[-95, 0, 6.5, 55],
                     )

_plt_kwgs = dict(nrow=1, ncol=1, projection="auto",
                 sharex=True, sharey=True,
                 central_lon=None, central_lat=None,
                 cbar_kwargs={"fraction":.025, "label":""}
                )

def _get_proj(proj, domain=None):
    """ from str or "auto" """
    if proj == "auto":
        proj = _get_proj(_def_proj(domain))
    elif proj.lower() == "orthographic":
        proj = ccrs.Orthographic
    elif proj.lower() in ["albersequalarea", "equalarea"]:
        proj = ccrs.AlbersEqualArea
    elif proj.lower() == "robinson":
        proj = ccrs.Robinson
    elif proj.lower() == "mollweide":
        proj = ccrs.Mollweide
    else:
        proj = ccrs.PlateCarree
    return proj

def _def_proj(domain):
    """ returns ccrs proj from name (str) """
    if domain is None:
        proj = "PlateCarree"
    elif domain.lower() in ["enatl60", "natl"]:
        proj = "Robinson"
    elif domain.lower() in ["global"]:
        proj = "Mollweide"
    elif domain.lower() in ["gf","az","local","regional"]:
        proj = "equalarea"
    else:
        proj = "PlateCarree"
    return proj

def plot(data, ax=None, domain=None, **kwargs):
    """ plot field using matplotlib and cartopy. Wraps pcolormesh (through xarray)

    Parameters
    __________
    data: xr.Datarray (2D)
        field to plot. Must have "llon" and "llat" coordinates attached to it
    ax: pyplot.axis instance (optional)
        plot will be put in ax if provided
    domain: str (optional, default: None)
        domain to plot, to enable automatic selection of domain extent, projection, etc.
    kwargs: TODO document this

    Returns
    _______
    matplotlib collection quadmesh or equivalent
    
    TODO: fix kwargs, "add_colorbar" seems to be ignored
    """
    kwgs = _plt_kwgs.copy()
    kwgs.update(kwargs)
    hgrid = "".join([gop._get_dim_from_d(data, d)[-1] for d in "xy"])
    if ax is None:
        extent = kwgs.pop("extent", _domain_extent.get(domain, None))
        if extent is None: # auto generation
            lon_l = data["llon_"+hgrid].isel({"x_"+hgrid[0]:0}).mean().values
            lon_r = data["llon_"+hgrid].isel({"x_"+hgrid[0]:-1}).mean().values
            lat_s = data["llat_"+hgrid].isel({"y_"+hgrid[1]:0}).mean().values
            lat_n = data["llat_"+hgrid].isel({"y_"+hgrid[1]:-1}).mean().values
            extent = [lon_l, lon_r, lat_s, lat_n]
        lon_0, lat_0 = kwgs.pop("central_lon"), kwgs.pop("central_lat")
        lon_0 = sum(extent[:2])/2 if lon_0 is None else lon_0
        lat_0 = sum(extent[2:])/2 if lat_0 is None else lat_0
        if kwgs["projection"] == "auto":
            projection = _def_proj(domain)
        else:
            projection = kwgs["projection"]
        ### Create figure axis
        fig_kwgs = ["sharex", "sharey", "figsize", "ncol", "nrow"]
        fig_kwgs = {k:kwgs[k] for k in kwgs.keys() if k in fig_kwgs}
        fig, ax = make_figax(central_lon=lon_0, central_lat=lat_0,
                            projection=projection, extent=extent, **fig_kwgs)
        #raise NotImplementedError("auto generation of ax not yet implemented"\
        #                          +"please generate one with make_figax")
    plt_kwgs = ["vmin", "vmax", "norm", "cmap", "add_colorbar"]
    plt_kwgs = {k:kwgs[k] for k in plt_kwgs if k in kwgs}
    if plt_kwgs.get("add_colorbar", True):
        plt_kwgs["cbar_kwargs"] = kwgs["cbar_kwargs"]
    hpc = data.plot(ax=ax, x="llon_"+hgrid, y="llat_"+hgrid, transform=ccrs.PlateCarree(),
              **plt_kwgs
             )
    return hpc

def make_figax(domain=None, **kwargs):
    """ make figure and axs ready for cartopy plotting on eNATL60 domain(s).
    basically a wrapper around pyplot.subplots and cartopy.ccrs
    subsequent plotting requires the following kwargs:
        ax=ax, transform=plot.ccrs.PlateCarree(), x="llon_xx", y="llat_xx"
    (x = c or r; llon_xx and llat_xx must be in the DataArray coordinates)

    Parameters (all optional)
    __________
    domain: str (default: None)
        which domain will be plotted, used to fix domain extents and
        automatically choosing the projection.
    ncol: int (default: 1)
        number of columns in the subplots
    nrow: int (default: 1)
        number of rows in the subplots
    projection: str (default: "auto")
        which projection to use. Default ("auto") triggers automatic selection based on "domain" value.
        Robinson will be used for basin-scale plots, "AlbersEqualArea" for regional plots and PlateCarree otherwise (see _def_proj).
    extent: list of 4 floats (default: None)
        domain extent in terms of lon/lat [left, right, bottom, top]
    cartopy kwargs "central_lon", "central_lat"
    subplots kwargs "sharex", "sharey", "figsize"

    Returns:
    ________
    A 2-element tuple containing fig, ax(s) handles.
    """

    kwgs = _plt_kwgs.copy()
    kwgs.update(kwargs)
    ### infer projection if not provided
    if kwgs["projection"] == "auto":
        proj = _def_proj(domain) # str
    else:
        proj = kwgs["projection"]
    map_proj = _get_proj(proj) # ccrs

    ### get or infer extent, lon_0, lat_0 and set map_proj
    extent = kwgs.pop("extent", _domain_extent.get(domain, None))
    central_lon = kwgs["central_lon"]
    if central_lon is None:
        central_lon = 0. if extent is None else sum(extent[:2])/2
    if proj.lower() in ["orthographic","albersequalarea","equalarea"]:
        central_lat = kwgs["central_lat"]
        if central_lat is None:
            central_lat = sum(extent[2:])/2
        map_proj = map_proj(central_longitude=central_lon,
                            central_latitude=central_lat
                            )
    else:

        map_proj = map_proj(central_longitude=central_lon)

    ### create fig subplots
    nrow, ncol = kwgs.pop("nrow"), kwgs.pop("ncol")
    subplot_kwg = {k:kwgs.pop(k,None) for k in ["sharex","sharey","figsize"]}
    subplot_kwg = {k:v for k,v in subplot_kwg.items() if v is not None}

    fig, axs = plt.subplots(nrow,ncol, subplot_kw={"projection":map_proj}, **subplot_kwg)
    # tweak labels & stuff # TODO FIX THIS for non-regional plots
    itax = axs.ravel() if nrow * ncol > 1 else [axs]
    for ax in itax:
        if extent is not None:
            ax.set_extent(extent)
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

    return fig, axs

### kept for backward compatibility
def prep_plot_local(**kwargs):
    warnings.warn("prep_plot_local is deprecated. Use make_figax instead.", 
                  DeprecationWarning
                  )
    _ploc_kwgs = {"figsize":(8,6), "nrow":1, "ncol":1, 
                  "central_lon":0, "central_lat":40, 
                  "extent":None, "sharex":True, "sharey":True,
                  "projection":"orthographic"
                  }
    kwgs = _ploc_kwgs.copy()
    kwgs.update(kwargs)
    if kwgs["projection"].lower() == "orthographic":
        map_proj = ccrs.Orthographic
    elif kwgs["projection"].lower() in ["albersequalarea", "equalarea"]:
        map_proj = ccrs.AlbersEqualArea
    if kwgs["projection"].lower() in ["orthographic","albersequalarea","equalarea"]:
        map_proj = map_proj(central_longitude=kwgs["central_lon"],
                            central_latitude = kwgs["central_lat"]
                            )
    else:
        map_proj = ccrs.PlateCarree(central_longitude=kwgs["central_lon"])

    nrow, ncol = kwgs.pop("nrow"), kwgs.pop("ncol")
    subplot_kwg = {k:kwgs.pop(k) for k in ["sharex","sharey","figsize"]}
    subplot_kwg = {k:v for k,v in subplot_kwg.items() if v is not None}

    fig, axs = plt.subplots(nrow, ncol, subplot_kw={"projection":map_proj}, **subplot_kwg) 
    itax = axs.ravel() if nrow * ncol > 1 else [axs]
    for ax in itax:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cfeature.LAND, facecolor="lightgray")

    if kwgs["extent"] is not None:
        for ax in itax:
            ax.set_extent(kwgs["extent"])

    return fig, axs

_plot_kwgs_old = {"figsize":(12,8),
                 "central_lon":-40, "extent":[-95, 0, 6.5, 55],
                 "sharex":True, "sharey":True
                }               
                 
def prep_one_plot(**kwargs):
    """ make figure and ax ready for cartopy plotting on eNATL60 domain.
    returns (fig, ax).
    subsequent plotting requires kwargs transform=plot.ccrs.PlateCarree() and "x" and "y" names in DataArray coordinates
    """
    warnings.warn("prep_one_plot is deprecated. Use make_figax instead.", 
                  DeprecationWarning
                  )
    kwgs = _plot_kwgs_old.copy()
    kwgs.update(kwargs)
    map_proj = ccrs.Robinson(central_longitude=kwgs["central_lon"])

    fig = plt.figure(figsize=kwgs["figsize"])
    ax = plt.axes(projection=map_proj)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.set_extent(kwgs["extent"])

    return fig, ax

def prep_subplots(nrow=1, ncol=1, **kwargs):
    """ make figure and axs ready for cartopy plotting on eNATL60 domain.
    returns (fig, axs).
    subsequent plotting requires kwargs transform=plot.ccrs.PlateCarree() and "x" and "y" names in DataArray coordinates
    wrapper around pyplot.subplots()
    """
    warnings.warn("prep_subplots is deprecated. Use make_figax instead.", 
                  DeprecationWarning
                  )
    kwgs = _plot_kwgs_old.copy()
    kwgs.update(kwargs)
    map_proj = ccrs.Robinson(central_longitude=kwgs["central_lon"])
    subplot_kwg = {k:v for k,v in kwgs.items() if k in ["sharex","sharey"]}

    fig, axs = plt.subplots(nrow, ncol, figsize=kwgs["figsize"], 
                            subplot_kw={"projection":map_proj}, 
                            **subplot_kwg)
    for ia,ax in enumerate(axs.ravel()):
        gl = ax.gridlines(draw_labels=False)
        ic, ir = ia%ncol, ia//ncol
        if ic==0 or not kwgs["sharex"]:
            gl.left_labels = True
        if ir==(nrow-1) or not kwgs["sharey"]:
            gl.bottom_labels = True
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.set_extent(kwgs["extent"])

    return fig, axs
### End of deprecated routines section

def get_cmap_setbad(cmap, color="LightGrey"):
    """ returns cmap with bad values (nan) set to color (default: "LightGrey")
    """
    cmap = copy.copy(plt.get_cmap(cmap))
    cmap.set_bad(color)
    return cmap

def get_discrete_cmap_norm(vmin=0, vmax=1, nlevs=10, cmap="Spectral", extend="neither"):
    """ discrete colormap and linear norm

    linear colorbar between vmin and vmax with "nlevs" levels using colormap "cmap".
    Depending of the value of "extend", the colormap will have nlevs-1 (extend="neither"),
    nlevs (extend="min" or "max") or nlevs+1 (extend="both") colors.

    internally uses colors.from_levels_and_colors and pyplot.get_cmap from matplotlib

    returns
    _______
    cmap, norm

    """
    cmap = copy.copy(plt.get_cmap(cmap))

    if extend=="both":
        idp = 1
    elif extend=="neither":
        idp = -1
    else:
        idp = 0

    colors = cmap(np.linspace(0, 1, nlevs+idp))
    levels = np.linspace(vmin, vmax, nlevs)

    cmap, norm = mpl.colors.from_levels_and_colors(levels, colors, extend=extend)
    return cmap, norm

def plot_mode_matrix(data, ax=None, x="mode", y=None, invert_y=True, **kwargs):
    """ plot a representation of a matrix
    data is a 2D xr.DataArray. 
    This routine wraps pcolormesh (through xarray plotting routines) and manually add lines.
    kwargs will be directly passed to pcolormesh and should contains items accordingly
    """
    dico = {k:v for k,v in kwargs.items()}
    if ax is not None:
        dico["ax"] = ax
    if y is not None:
        dico["y"] = y
        Ny = data[y].size
    else:
        Ny = data[x].size
    hpc = data.plot(**dico)
    if ax is None:
        ax = hpc.axes
    if invert_y:
        ax.invert_yaxis()
    # manually add lines
    ax.set_xticks(np.arange(data[x].size))
    ax.set_xticks(np.arange(data[x].size)-.5, minor=True)
    ax.set_yticks(np.arange(Ny))
    ax.set_yticks(np.arange(Ny)-.5, minor=True)
    ax.grid(which="minor")
    return hpc
