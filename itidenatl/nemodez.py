""" nemodez.py python module
Defines class Vmodes and routines for computing vertical modes given a stratification profile on NEMO outputs. 
Adapted to z-level formulation, and outputs loaded as xarray Dataset through xorca library. Uses xgcm for grid manipulations.

The corresponding Sturm-Liouville problem is (phi'/N2)' + lam^2 phi = 0, with proper BCs (free surface or rigid lid, flat inpenetrable bottom)

TODO: 
    - this could receive some generalization of variable names.
    - check that handling type is as smart as possible. probably not.

NJAL May 2021 (noe.lahaye@inria.fr)
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.linalg import eig
import xarray as xr
from xgcm import Grid

_nmod_def = 10
_dico_def = {"g": 9.81, "free_surf": True, "eig_sigma":.1, "siz_sparse":30, 
        "corr_N":True, "N2name":"bvf", "first_order_formulation":False}
_core_variables = ['phi', 'phiw', 'c', 'norm']
_core_attrs = ["g", "nmodes", "free_surf"]

class Vmodes(object):
    """ Class for vertical modes computation and manipulation (projection and reconstruction)
    The vertical modes 'phi' are defined following the standard Sturm-Liouville problem 
    (e.g. Gill, 1982 or any standard GFD textbook), see below.
    Those are referred as 'pressure-like' modes. 
    Companion modes, referred as "w-like" modes and named 'phiw', obeye phiw'=phi

    Parameters:
    ___________
    ds: xarray.Dataset
        dataset containing Brunt-Vaisal frequency and vertical grid (levels or metrics, i.e. spacing)
    xgrid: xgcm.Grid 
        xgcm grid object associated with the dataset, required for grid manipulations
    nmodes: int, optional
        number of baroclinic modes to compute (barotropic mode will be added). Default: 10
    free_surf: bool, optional
        whether using a free-surface condition or rigid lid -- see warning below
    persist: bool, optional
        persist the dask dataset containing the modes or not
    grav: float, optional
        gravity constant
    sigma: float, optional
        parameter for shift-invert method in scipy.linalg.eig (default: _sig)
        
    Attributes:
    ___________
    xgrid: xgcm.Grid
    ds: xarray.Dataset
        dataset containing the vertical modes and associated data
    nmodes: int
    g: float
    free_surf: float
    dicopt: dict 
        containing various attributes

    Methods:
    ________
    project(data, vartype=None, **kwargs)
        project 'data' on vertical modes
    project_puv(data, z=False, sel=None, align=True)
        project 'data' on pressure-like vertical modes
    project_w(data, z=False, sel=None, align=True) 
        project 'data' on w-like vertical modes
    project_b(data, z=False, sel=None, align=True)
        project 'data' on b-like vertical modes
    reconstruct(projections, vartype=None, **kwargs)
        reconstruct field from projections coefficient (modal amplitudes)
    reconstruct_puv(projections, sel=None, align=True)
        reconstruct field from projections coefficient using pressure-like modes
    reconstruct_w(projections, sel=None, align=True)
        reconstruct field from projections coefficient using w-like modes
    reconstruct_b(projections, sel=None, align=True)
        reconstruct field from projections coefficient using w-like modes

    Warnings:
    _________
    When using rigid lid, no barotropic mode could be found and the modes will be shifted down by 1 along the "mode" dimension. Last elements will have "nan". I should fix this by adding an estimated barotropic mode with :math:`\phi=H^{-1/2}, c=\sqrt{gH}`.

    Notes:
    ______
    zc, zf, N2 should be ordered from bottom to top, and zf and N2 have the same z dimension
    The vertical modes are definied following the equation:
    .. math:: (\phi'/N^2)' + k^2\phi=0 
    with boundary conditions
        :math:`\phi'=0` at the bottom
        :math:`g\phi' + N^2\phi=0` at the surface (or :math:`\phi'=0` for a rigid lid condition). 
    Computation of the vertical modes is performed using second order finite difference (staggered grids)
    The eigenvalue used in this class is c=1/k.
    """
    dicopt = _dico_def.copy()

    def __init__(self, ds, xgrid,
                 nmodes=_nmod_def, persist=False,
                 **kwargs):
        self.dicopt.update(kwargs)
        self.xgrid = xgrid
        self.nmodes = nmodes
        self.g = self.dicopt["g"]
        self.free_surf = self.dicopt["free_surf"]
        self._z_dims = {"zc": "z_c", "zl": "z_l"}
        self._z_del = {"zc": "e3t", "zl": "e3w"}
        self._z_mask = {"zc": "tmask"}
        N2name = self.dicopt["N2name"] 
        self._N2name = N2name

        # create dataset
        if self.dicopt["corr_N"]:
            Nmin = 1e-10
            self.ds = xr.Dataset(coords={N2name:ds[N2name].where(ds[N2name]>=Nmin,Nmin)})
        else:
            self.ds = xr.Dataset(coords={N2name:ds[N2name]})
        coords = [v for v in self._z_del.values() if v not in self.ds]
        coords += [v for v in self._z_mask.values() if v not in self.ds]
        self.ds = self.ds.assign_coords({co:ds[co] for co in coords})
        
        self._compute_vmodes()        
       
        if persist:
            self.ds = self.ds.persist()
    
    def __getitem__(self, item):
        """ Enables calls such as vm['N2']
        """
        if item in ['zc', 'zl']:
            return self.ds[self._z_dims[item]]
        elif item in self.dicopt:
            return self.dicopt[item]
        elif item in self.ds:
            return self.ds[item]
        else:
            return None

    def __repr__(self):
        strout = 'Vmode object with dimensions {}\n'.format(tuple(self.ds.dims.keys()))
        strout += '  Number of modes = {}\n'.format(self.nmodes)
        strout += '  Corresponding Dataset in self.ds \n'
        strout += '  Options / parameters:\n' 
        strout += ', '.join(["{0}={1:.2e}".format(key,val) for key,val in self.dicopt.items() 
                             if not isinstance(val, (bool,str))])+"\n"
        strout += ', '.join(["{0}={1}".format(key,val) for key,val in self.dicopt.items() 
                             if isinstance(val, (bool,str))])+"\n"
        return strout
    
    def _compute_vmodes(self):
        """ compute vertical modes and store the results into the dataset 
        wrapper of external function get_vmodes """
        #dm = get_vmodes(self.ds, nmodes=self.nmodes,
        #                free_surf=self.free_surf, g=self.g,
        #                eig_sigma=self.dicopt["eig_sigma"], z_dims=self._z_dims, 
        #                z_dels=self._z_del, N2=self._N2name)
        dm = get_vmodes(self.ds, nmodes=self.nmodes,
                        z_dims=self._z_dims, z_dels=self._z_del, **self.dicopt)
        self.ds = xr.merge([self.ds, dm], compat="override")
        if "chunks" in self.dicopt:
            self.ds = self.ds.chunk(self.dicopt["chunks"])
        
    def project(self, data, vartype=None, **kwargs):
        """ Project a variable on vertical modes (p-modes or w-modes)
        Internally calls project_puv or project_w or project_b
        
        Parameters:
        ___________
        data: xarray.DataArray
            array containing the data to be projected. 
        vartype: str, optional 
            string specifying whether projection should be done w-modes ("w"), 
            buoyancy modes ("b") or pressure modes (any other value)
            If not specified, it is inferred from data.name. 
            If this fails, default is "p", i.e. pressure mode.
        sel: Dict, optional (default: None)
            indices applied to the vmodes dataset prior to projection
        align: bool, optional (default: True)
            whether alignment between the data and vmodes DataArray should be performed before projecting.
            Note that any mismatch between data and vertical modes horizontal grids will raise an error
            (horizontal interpolation needs to be performed prior to projection)
        
        Returns
        _______
        xarray.DataArray
            Projected array
            
        See also
        ________
        project_puv, project_w, project_b, reconstruct
        
        """
        ### try to infer vartype if not specified
        if vartype is None:
            if next(key in data.name for key in ["pres", "vomecrty", "vozocrtx"]) or next(data.name==key for key in ["p","u","v"]):
                vartype = "p"
            elif next(key in data.name for key in ["sigma", "dens", "buoy"]) or data.name=="b":
                vartype = "b"
            elif next(key in data.name for key in ["vovecrtz"]) or data.name=="w":
                vartype = "w"

        if vartype == "w":
            return self.project_w(data, **kwargs)
        elif vartype == "b":
            return self.project_b(data, **kwargs)
        else:
            return self.project_puv(data, **kwargs)

    def project_puv(self, data, sel=None, align=True):
        """ project on pressure modes, from data at T levels 
        Use (projection * self.ds.phi).sum("mode") to reconstruct
        See also: project, reconstruct_puv
        """
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
            
        if align:
            data, dm = xr.align(data, dm, join="inner")
            
        #_check_hdim_mismatch(data, dm)
            
        res = (dm[self._z_del["zc"]]*data*dm.phi).where(dm[self._z_mask["zc"]]).sum(self._z_dims['zc'])/dm.norm

        return res
    
    def project_w(self, data, sel=None, align=True): 
        """ project on w-modes, from data at T or w levels 
        use (projection * self.ds.phiw).sum("mode") to reconstruct
        See also: project, reconstruct_w
        this needs some more tests """
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        zc, zl = self._z_dims["zc"], self._z_dims["zl"]

        if align:
            data, dm = xr.align(data, dm, join="inner")
            
        #_check_hdim_mismatch(data, dm)

        if zl in data.dims:
            val_surf = data.isel({zl:0}).drop(zl)
            data = self.xgrid.interp(data, "Z", boundary="extrapolate")
        else:
            val_surf = data.isel({zc:0}).drop(zc) ### warning: nearest interpolation at surface
        prov = (data * self.xgrid.interp(dm.phiw*dm[self._N2name], "Z", boundary="fill", 
                                        fill_value=0
                                        ) * dm[self._z_del["zc"]]
               ).where(dm[self._z_mask["zc"]]).sum(zc)
        
        if self.free_surf:
            prov += self.g * dm.phiw.isel({zl:0}) * val_surf
       
        return prov /dm.norm/dm.c**2

    def project_b(self, data, sel=None, align=True): 
        """ project on b-modes, from data at T or w levels 
        Use (projections * self.ds.phiw * self.ds.Nsqr).sum("mode") to reconstruct
        See also: project, reconstruct, reconstruct_b
        this needs some more tests"""
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        zc, zl = self._z_dims["zc"], self._z_dims["zl"]

        if align:
            data, dm = xr.align(data, dm, join="inner")
            
        #_check_hdim_mismatch(data, dm)

        if zl in data.dims:
            val_surf = data.isel({zl:0}).drop(zl)
            data = self.xgrid.interp(data, "Z", boundary="extrapolate")
        else:
            val_surf = data.isel({zc:0}).drop(zc) ### warning: nearest interpolation at surface
        prov = (data * self.xgrid.interp(dm.phiw, "Z", boundary="fill", fill_value=0) 
                * dm[self._z_del["zc"]]
               ).where(dm[self._z_mask["zc"]]).sum(zc)
        
        if self.free_surf:
            prov += self.g * val_surf * \
                    (dm.phiw/dm[self._N2name]).isel({zl:0}).where(dm[self._N2name].isel({zl:0})!=0., 0.)
       
        return prov /dm.norm/dm.c**2

    def reconstruct(self, projections, vartype=None, **kwargs):
        """ Reconstruct a variable from modal amplitudes
        Internally calls reconstruct_puv or reconstruct w or reconstruct_b
        
        Parameters:
        ___________
        projections: xarray.DataArray
            array containing the modal projection coefficients (modal amplitudes)
        vartype: {"p", "u", "v", "w", "b"}, optional 
            string specifying whether reconstruction should be done using w-modes ("w"), buoyancy modes ("b") or pressure modes ("p", "u" or "v"). 
            Default is will be inferred from projections.name, or use "p".
        sel: Dict, optional (default: None)
            indices applied to the vmodes dataset prior to reconstruction
        
        Returns
        _______
        xarray.DataArray
            Reconstructed field array
            
        See also
        ________
        reconstruct_puv, reconstruct_w, reconstruct_b, project
        
        """
        
        if vartype is None:
            vartyps = "puvbw"
            if sum(s in projections.name.lower() for s in vartyps)==1:
                vartype = next((s for s in vartyps if s in projections.name.lower()))
            else: 
                raise ValueError("unable to find what kind of basis to use for reconstruction")
                
        if vartype in "puv":
            return self.reconstruct_puv(projections, **kwargs)
        elif vartype == "w":
            return self.reconstruct_w(projections, **kwargs)
        elif vartype == "b":
            return self.reconstruct_b(projections, **kwargs)
        
    def reconstruct_puv(self, projections, sel=None, align=True):
        """ Reconstruct a variable from modal amplitudes, using pressure-like modes
            
        See also
        ________
        reconstruct, project, project_puv
        
        """

        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        if align:
            projections, dm = xr.align(projections, dm, join="inner")    
        return (projections * dm.phi).sum("mode")

    def reconstruct_w(self, projections, sel=None, align=True, on_t_lev=True):
        """ Reconstruct a variable from modal amplitudes, using w-like modes 
            
        See also
        ________
        reconstruct, project, project_w
        
        """
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        if align:
            projections, dm = xr.align(projections, dm, join="inner")    
        if on_t_lev:
            phiw = self.xgrid.interp(dm.phiw, "Z", boundary="fill", fill_value=0)
        else:
            phiw = dm.phiw
        #return (-dm.c**2 / dm.N2 * dm.dphidz * projections).sum("mode")
        return (projections * phiw).sum("mode")
    
    def reconstruct_b(self, projections, sel=None, align=True, on_t_lev=True):
        """ Reconstruct a variable from modal amplitudes, using w-like modes and b-like normalization
            
        See also
        ________
        reconstruct, project, project_b
        
        """
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        if align:
            projections, dm = xr.align(projections, dm, join="inner")    
        if on_t_lev:
            Nsq_phiw = self.xgrid.interp(dm.phiw * dm[self._N2name], "Z", 
                                    boundary="fill", fill_value=0)
        else:
            Nsq_phiw = dm.phiw * dm[self._N2name]
        return (projections * Nsq_phiw).sum("mode")

    def store(self, file_path, projections=None, coords=True, **kwargs):
        """ Store Vmodes object along with projections into zarr archives
        
        Parameters
        ----------
        file_path: str
            File path where data will be stored, e.g. '/home/roger/vmodes.zarr'
        projections: xarray.Dataset
            Dataset containing projections
        **kwargs:
            kwargs passed to `to_zarr`
        """
        ds = self._wrap_in_dataset()
        if isinstance(file_path, str):
            _file = Path(file_path).with_suffix(".zarr")
        else:
            _file = file_path.with_suffix(".zarr")
        # undesired singleton time coordinates
        #ds = _move_singletons_as_attrs(ds)
        #
        if coords:
            sds = ds    
        else:
            sds = ds.reset_coords(drop=True)
        sds.to_zarr(_file, **kwargs)
        print('Store vertical modes in {}'.format(_file.resolve()))
        #
        if projections:
            _file = Path(_file.stem+'_projections.zarr')
            ds = _move_singletons_as_attrs(projections)
            ds.to_zarr(_file, **kwargs)
            print('Store projections in {}'.format(_file.resolve()))

    
    def _wrap_in_dataset(self):
        """ wrap critical data in a xarray dataset
        """
        # we won't store xgrid as the intend is to store this in a zarr archive
        #
        # One should try to figure out whether minimal information can be extracted
        # from the xgrid object and store as attribute to the xarray dataset
        attrs = {key:val for key,val in self.dicopt.items() if not isinstance(val,dict)}
        ds = self.ds.assign_attrs(attrs) 
        ds = ds.assign_attrs(**{a: getattr(self, a) for a in _core_attrs})
        ds = ds.assign_attrs(zc_name=self._z_dims['zc'],
                             zl_name=self._z_dims['zl'])    
        return ds
    
def get_vmodes(ds, nmodes=_nmod_def, **kwargs):
    """ compute vertical modes
    Wrapper for calling `compute_vmodes` with DataArrays through apply_ufunc. 
    z levels must be in descending order (first element is at surface, last element is at bottom) with algebraic depth (i.e. negative)
    Normalization is performed here (int_z \phi^2 \dz = Hbot)
    
    Parameters:
    ___________
    ds: xarray.Dataset
        contains brunt-vaisala frequency and vertical grid information (levels of metrics, i.e. spacing)
    nmodes: int, optional
        number of vertical baroclinic modes (barotropic is added)
    free_surf: bool, optional
        whether to use free surface boundary condition or not
    sigma: scalar or None, optional
        parameter for shift-invert method in scipy.linalg.eig (default: _sig)
    g: scalar, optional
        gravity constant
    z_dims: list of str, optional
        vertical dimension names in zc, zl (default: "z_c", "z_l")
    N2: str, optional
        name of BVF squared variable in ds

    Returns:
    ________
    xarray.DataSet: vertical modes (p and w) and eigenvalues
    
    See Also:
    _________
    compute_vmodes: routine for computing vertical modes from numpy arrays
    Vmodes: class for computing and manipulating vertical modes
   
    """
    """ normalization is performed here """
    kworg = kwargs.copy()
    kworg.update({"nmodes":nmodes})
    zc, zl = kwargs["z_dims"]["zc"], kwargs["z_dims"]["zl"]
    N2 = kworg.get("N2name", _dico_def["N2name"])
    
    N = ds[zc].size
    res = xr.apply_ufunc(_compute_vmodes_1D_stack, 
                         ds[N2].chunk({zl:-1}), 
                         (ds.e3t.where(ds.tmask)).chunk({zc:-1}),
                         ds.e3w.chunk({zl:-1}),
                         kwargs=kworg, 
                         input_core_dims=[[zl],[zc],[zl]],
                         dask='parallelized', vectorize=True,
                         output_dtypes=[ds[N2].dtype],
                         output_core_dims=[["s_stack","mode"]],
                         dask_gufunc_kwargs={"output_sizes":{"mode":nmodes+1,"s_stack":2*N+1}}
                        )
    res['mode'] = np.arange(nmodes+1)
    # unstack variables
    c = res.isel(s_stack=0)
    phi = (res.isel(s_stack=slice(1,N+1))
           .rename('phi')
           .rename({'s_stack': zc})
           #.assign_coords(z_rho=zc)
          )
    if "z_del" in kwargs:
        dzc = ds[kwargs["z_del"]["zc"]]
    else:
        dzc = ds["e3t"] # use default value for NEMO    
    norm_tg = dzc.where(ds.tmask).sum(zc)
    norm = (phi**2*dzc).where(ds.tmask).sum(zc) 
    phi /= (norm/norm_tg)**.5 # 1/H \int(phi^2 dz) = 1
    phiw = (res.isel(s_stack=slice(N+1,2*N+1))
              .rename('phiw')
              .rename({'s_stack': zl})
            #  .assign_coords(z_w=zf)
             ) / norm**.5
    norm = norm_tg # norm = int(phi^2 dz)
    # merge data into a single dataset
    dm = xr.merge([c.rename("c"), phi.rename("phi"), 
                   phiw.rename("phiw"), norm.rename("norm")
                 ])
    return dm  ### hard-coded norm = H 

def load_vmodes(file_path, xgrid=None, persist=False):
    """ load vertical modes from a file
    
    Parameters
    ----------
    file_path: str
        Path to vmode datafile (zarr archive)
    xgrid: xgcm.Grid object
        Required for grid manipulations
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    ds = xr.open_zarr(file_path.with_suffix(".zarr"))
    if xgrid is None:
        metrics = {key:val for key,val in ds.attrs.items() if key.startswith("metrics_")}
        metrics = _unwrap_xgcm_grid_metrics(metrics)
        xgrid = Grid(ds, metrics=metrics, periodic=False)
    vm = Vmodes(ds, xgrid,
                 nmodes=ds.nmodes, 
                free_surf=ds.free_surf,
                persist=False,
                g=ds.g, eig_sigma=ds.eig_sigma)
    # transfer mode from ds to vm
    for v in _core_variables:
        if v in ds:
            vm.ds[v] = ds[v]
        elif v in ds.attrs:
            # singleton cases (norm)
            vm.ds[v] = ds.attrs[v]
    if persist:
        vm.ds = vm.ds.persist()
    # search for projections:
    _pfile = Path(file_path.stem+'_projections.zarr')
    if _pfile.is_dir():
        projections = xr.open_zarr(_pfile)
        if persist:
            projections = projections.persist()
        return vm, projections
    else:
        return vm
    
def compute_vmodes_1D(Nsqr, dzc=None, dzf=None, zc=None, zf=None, nmodes=_nmod_def, **kwargs): 
    """
    Compute vertical modes from stratification. Assume grid is sorted downoward (first point at surface, last point at bottom) and depth is algebraic (i.e. negative)
    Take either vertical grid metrics (spacing) or levels as inputs. 
    Need 2 staggered grid (center and left or outer), with Nsqr specified on left/outer grid
    No normalization. Pressure mode is positive at the surface.

    Parameters:
    ___________
    N2f: (N,) ndarray
        Brunt-Vaisala frequency at cell left points
    dzc: (N) ndarray, optional
        vertical grid spacing at cell centers. Either dzc, dzf or zc, zf must be passed
    dzf: (N) ndarray
        vertical grid spacing at cell left points
    zc: (N) ndarray
        vertical grid at cell centers
    zf: (N+1,) or (N,) ndarray
        vertical grid at cell left points or interfaces (outer points)
    nmodes: int, optional
        number of baroclinic modes to compute (barotropic mode will be added)
    free_surf: bool, optional
        whether using a free-surface condition or rigid lid
    g: float, optional
        gravity constant
    eig_sigma: float, optional
        parameter for shift-invert method in scipy.linalg.eig (default: _sig)
    Hbot: float, optional
        bottom depth. Must be specified if zf is passed instead of dzf, and bottom point is not included in zf (i.e. zf.size=N)

    Returns:
    ________
    c: (nmodes) ndarray
        eigenvalues (pseudo phase speed, c=1/sqrt(k))
    phi: (N,nmodes) ndarray
        p-like modes at cell centers
    phiw: (N,nmodes) ndarray
        w-like modes at cell interfaces. phiw' = phi

    Notes:
    ______
    The vertical modes are definied following the equation:
    .. math:: (\phi'/N^2)' + k^2\phi=0 
    with boundary condition :math:`\phi'=0` at the bottom and :math:`g\phi' + N^2\phi=0` at the surface (or :math:`\phi'=0` for a rigid lid condition). 
    Computation of the vertical modes is performed using second order finite difference with staggered grid

    """ 
    ### extract keywords
    kwgs = _dico_def.copy()
    kwgs.update(kwargs)
    free_surf, g, sigma = kwgs["free_surf"], kwgs["g"], kwgs["eig_sigma"]
    first_ord = kwgs["first_order_formulation"]

    ### deal with vertical grids
    Nz = Nsqr.size
    if dzc is not None and dzf is not None:
        if zc is not None:
            dz_surf = abs(zc[0])
        else:
            dz_surf = .25*(dzc[0] + dzf[0]) ### this is approx for NEMO grid
        dzc, dzf = dzc, dzf
    elif zc is not None and zf is not None:
        if zf.size == zc.size:
            if "Hbot" not in kwgs["Hbot"]:
                raise ValueError("must specify w-grid at outer points or Hbot")
            else:
                zf = np.r_[zf, -abs(kwgs["Hbot"])]
        elif zf.size != zc.size+1:
            raise ValueError("incompatible sizes between zc grid and zf grid")
        dzc = np.diff(zf)
        dzf = np.diff(zc)
    else:
        raise ValueError("must specify either grid increments dzc, dzf or z grids zc, zf") 

    if free_surf and g>0:
        invg = np.ones(1)/g
    else:
        invg = np.zeros(1)
    Nsqog = Nsqr[:1]*invg

    if first_ord:
        ### construct sparse matrices for differentiation
        # vertical derivative matrix, w-to-p grids, taking left w-points (assume w(N+1)=0)
        v12 =  np.stack([-1./np.r_[np.ones(1),dzc], 1./np.r_[dzc, np.ones(1)]])
        Dw2p = sp.spdiags(v12,[1, 0],Nz,Nz,format="lil")
    
        # vertical derivative matrix, p-to-w grids, targetting inner w points only
        v12 =  np.stack([-1./np.r_[np.ones(1),dzf], 1./np.r_[dzf, np.ones(1)]])
        Dp2w = sp.spdiags(v12,[1, 0],Nz-1,Nz,format="lil")
        
        ### construct eigenproblem. State vector is (p=-w', w) (w is not true vertical velocity)
        # eigen problem is 
        # (1 Dz) * psi = lam * (0 0 ) * psi
        # (Dz 0)               (0 N2)
        # plus corrections for boundary condition at the surface
        # this free surface condition implementation is uncertain...
    
        B = sp.diags(np.r_[np.zeros(Nz), 1.+Nsqog*dz_surf/2., Nsqr[1:]], 0, (2*Nz,2*Nz), format="lil")
        Awp = sp.vstack([np.r_[-invg, np.zeros(Nz-1)], Dp2w])
        Aww = None #sp.diags(np.r_[np.ones(1)+Nsqog*dz_surf, np.zeros(Nz-1)], 0, (Nz,Nz))
        A = sp.bmat([ [ sp.identity(Nz), Dw2p ], 
                      [ Awp,             Aww  ]
                    ])
        
        ### compute numerical solution
        if Nz >= kwgs["siz_sparse"]:
            ev,ef = la.eigs(A.tocsc(), k=nmodes+1, M=B.tocsc(), sigma=sigma)
        else:
            Adens = A.toarray()
            Bdens = B.toarray()
            ev, ef = eig(Adens, Bdens)
    else:
        v12 =  np.stack([1./np.r_[dzc, np.ones(1),], -1./np.r_[np.ones(1), dzc]])
        Dw2p = sp.spdiags(v12,[0, 1],Nz,Nz,format="lil")

        ### vertical derivative matrix, p-to-w grids, targetting inner w points only
        v12 =  np.stack([1./np.r_[dzf[1:], np.ones(1)], -1./dzf])
        Dp2w = sp.spdiags(v12,[-1, 0],Nz,Nz,format="lil")
        
        ### second order diff matrix
        D2z = Dw2p*Dp2w
        Dp2w[0,0] = -Nsqog*(1-Nsqog*dz_surf) # surface boundary condition (free or rigid lid)

        ### formulation of the problem : -dz(dz(p)/N^2) = lambda * p
        A = - Dw2p * sp.diags(1./Nsqr) * Dp2w

        ### compute numerical solution
        if Nz >= kwgs["siz_sparse"]:
            ev,ef = la.eigs(A.tocsc(), k=nmodes+1, sigma=sigma)
        else:
            ev, ef = eig(A.toarray())

    #### select and arrange modes
    inds = np.isfinite(ev)
    ev, ef = ev[inds].real, ef[:,inds].real
    isort = np.argsort(ev)[:nmodes+1]
    ev, ef = ev[isort], ef[:,isort]
    ef *= np.sign(ef[0,:])[None,:] # positive pressure at the surface
    if first_ord:
        pmod, wmod = ef[:Nz,:], -ef[Nz:,:]
    else:
        pmod = ef[:Nz,:]
        wmod = -(Dp2w * pmod) / (Nsqr[:,None] * ev[None,:])
        if not (free_surf and g>0):
            wmod[:,0] = 0.
    
    return 1./ev**.5, pmod, wmod

def _compute_vmodes_1D_stack(N2l, dzc, dzf, **kwargs):
    """ wrapper to stack results from compute_vmodes_1D vertically
    Order is c, p, w. Also fills missing vertical levels with nans 
    returns a numpy array of shape (nmode+1, 2*NZ+1) """
    assert N2l.ndim==dzc.ndim==dzf.ndim==1
    lemask = ~np.isnan(dzc)
    nmodes = kwargs["nmodes"]
    if lemask.sum() > nmodes:
        c, p, w = compute_vmodes_1D(N2l[lemask], dzc=dzc[lemask], dzf=dzf[lemask], **kwargs)
        nans = np.full((dzc.size-lemask.sum(),c.size), np.nan, dtype=N2l.dtype)
        res = np.vstack([c, p, nans, w, nans]).astype(N2l.dtype)
    else:
        res = np.full((dzc.size * 2 + 1,nmodes+1), np.nan, dtype=N2l.dtype)
    return res

def _move_singletons_as_attrs(ds, ignore=[]):
    """ change singleton variables and coords to attrs
    This seems to be required for zarr archiving
    """
    for c,co in ds.coords.items():
        if co.size==1 and ( len(co.dims)==1 and co.dims[0] not in ignore or len(co.dims)==0 ):
            ds = ds.drop_vars(c).assign_attrs({c: ds[c].values})
    for v in ds.data_vars:
        if ds[v].size==1 and ( len(v.dims)==1 and v.dims[0] not in ignore or len(v.dims)==0 ):
            ds = ds.drop_vars(v).assign_attrs({v: ds[v].values})
    return ds

def _get_xgcm_grid_metrics(grid, ds=None):
    """ get metrics from xgcm grid object, associated with a ds or all available """
    metrics = {key: [va.name for va in val if va.name in ds.coords] for key,val in grid._metrics.items()}
    if ds is not None:
        metrics = {key: [va for va in val if va in ds.coords] for key,val in metrics.items()}
    return metrics

def _wrap_xgcm_grid_metrics(grid, ds=None):
    """ wrap xgcm grid metrics into a dict """
    metrics = _get_xgcm_grid_metrics(grid, ds=ds)
    return {"metrics_"+''.join(key): list(key)+val for key,val in metrics.items()}

def _unwrap_xgcm_grid_metrics(metrics):
    """ unwrap gxcm grid metrics from a dict """
    return {tuple(val[0]): val[1:] for val in metrics.values()}
