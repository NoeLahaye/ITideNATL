### routine for preparing plots
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams.update({"font.size":14, "font.family":"serif"})
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copy

import numpy as np

_plot_kwgs = {"figsize":(12,8),
                 "central_lon":-40, "extent":[-95, 0, 6.5, 55],
                 "sharex":True, "sharey":True
                }               
                 

def prep_one_plot(**kwargs):
    """ make figure and ax ready for cartopy plotting on eNATL60 domain.
    returns (fig, ax).
    subsequent plotting requires kwargs transform=plot.ccrs.PlateCarree() and "x" and "y" names in DataArray coordinates
    """
    kwgs = _plot_kwgs.copy()
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
    kwgs = _plot_kwgs.copy()
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
