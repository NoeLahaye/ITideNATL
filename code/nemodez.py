import numpy as np
from pathlib import Path# change this to pathlib maybe

import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.linalg import eig
import xarray as xr
from xgcm import Grid

_nmod_def = 10
_dico_def = {"g": 9.81, "free_surf": True, "eig_sigma":.1, "siz_sparse":30, "corr_N":True}
_core_variables = ['phi', 'phiw', 'c', 'norm']
_core_attrs = ["g", "nmodes", "free_surf"]

class Vmodes(object):
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
        N2name = "Nsqr"

        # create dataset
        if self.dicopt["corr_N"]:
            self.ds = xr.Dataset({N2name:ds[N2name].where(ds[N2name]>=0,0.)})
        else:
            self.ds = xr.Dataset({N2name:ds[N2name]})
        self.ds = self.ds.assign_coords({val:ds[val] for val in self._z_del.values()})
        self.ds = self.ds.assign_coords({val:ds[val] for val in self._z_mask.values()})
        
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
        dm = get_vmodes(self.ds, nmodes=self.nmodes,
                        free_surf=self.free_surf, g=self.g,
                        eig_sigma=self.dicopt["eig_sigma"], z_dims=self._z_dims, z_dels=self._z_del)
        self.ds = xr.merge([self.ds, dm], compat="override")
        
    
    def project_puv(self, data, sel=None, align=True):
        """ project on pressure modes, from data at T levels """
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
        this needs some more work """
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
        prov = (data * self.xgrid.interp(dm.phiw*dm.Nsqr, "Z", boundary="fill", fill_value=0) 
                * dm[self._z_del["zc"]]
               ).sum(zc)
        
        if self.free_surf:
            prov += self.g * dm.phiw.isel({zl:0}) * val_surf
       
        return prov /dm.norm/dm.c**2

    def reconstruct(self, projections, vartype=None, **kwargs):
        """ Reconstruct a variable from modal amplitudes
        Internally calls reconstruct_puv or reconstruct w or reconstruct_b
        
        Parameters:
        ___________
        projections: xarray.DataArray
            array containing the modal projection coefficients (modal amplitudes)
        vartype: {"p", "u", "v", "w", "b"}, optional 
            string specifying whether reconstruction should be done using w-modes ("w"), buoyancy modes ("b") or pressure modes ("p", "u" or "v"). Default is "p".
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
        reconstruct
        
        """

        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        if align:
            projections, dm = xr.align(projections, dm, join="inner")    
        return (projections * dm.phi).sum("mode")

    def reconstruct_w(self, projections, sel=None, align=True):
        """ Reconstruct a variable from modal amplitudes, using w-like modes and w-like normalization
            
        See also
        ________
        reconstruct
        
        """
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        if align:
            projections, dm = xr.align(projections, dm, join="inner")    
        #return (-dm.c**2 / dm.N2 * dm.dphidz * projections).sum("mode")
        return (projections * dm.phiw).sum("mode")
    
    def reconstruct_b(self, projections, sel=None, align=True):
        """ Reconstruct a variable from modal amplitudes, using w-like modes and b-like normalization
            
        See also
        ________
        reconstruct
        
        """
        raise NotImplementedError("reconstruct b not implemented")
        
        if sel is None:
            dm = self.ds
        else:
            dm = self.ds.sel(sel)
        if align:
            projections, dm = xr.align(projections, dm, join="inner")    
        return (-dm.c**2 * dm.dphidz * projections).sum("mode")

    def store(self, file_path, projections=None, **kwargs):
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
        _file = Path(file_path.rstrip(".zarr")+".zarr")
        # undesired singleton time coordinates
        #ds = _move_singletons_as_attrs(ds)
        #
        ds.to_zarr(_file, **kwargs)
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
        ds = self.ds.assign_attrs(**self.dicopt) 
        ds = ds.assign_attrs(**{a: getattr(self, a) for a in _core_attrs})
        ds = ds.assign_attrs(zc_name=self._z_dims['zc'],
                             zl_name=self._z_dims['zl'])    
        return ds
    
def get_vmodes(ds, nmodes=_nmod_def, **kwargs):
    """ normalization is performed here """
    kworg = kwargs.copy()
    kworg.update({"nmodes":nmodes})
    zc, zl = kwargs["z_dims"]["zc"], kwargs["z_dims"]["zl"]
    
    N = ds[zc].size
    res = xr.apply_ufunc(_compute_vmodes_1D_stack, 
                         ds.Nsqr.chunk({zl:-1}), 
                         (ds.e3t.where(ds.tmask)).chunk({zc:-1}),
                         ds.e3w.chunk({zl:-1}),
                         kwargs=kworg, 
                         input_core_dims=[[zl],[zc],[zl]],
                         dask='parallelized', vectorize=True,
                         output_dtypes=[np.float64],
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
    norm = (phi**2*dzc).where(ds.tmask).mean(zc) # 1/H \int(phi^2 dz) = 1
    phi /= norm**.5
    phiw = (res.isel(s_stack=slice(N+1,2*N+1))
              .rename('phiw')
              .rename({'s_stack': zl})
            #  .assign_coords(z_w=zf)
             ) / norm**.5
    norm = (phi**2*dzc).where(ds.tmask).sum(zc) # norm = int(phi^2 dz)
    # merge data into a single dataset
    #other_dims = tuple([dim for dim in ds.Nsqr.dims if dim!=zc]) # extra dims
    #dm = (xr.merge([c, phi, dphidz, norm])
    #      .transpose(*('mode',s_rho,s_w)+other_dims)
    #     )
    dm = xr.merge([c.rename("c"), phi.rename("phi"), phiw.rename("phiw"), norm.rename("norm")])
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
    ds = xr.open_zarr(file_path.rstrip(".zarr")+".zarr")
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
    _pfile = Path(file_path.rstrip('.zarr')+'_projections.zarr')
    if _pfile.is_dir():
        projections = xr.open_zarr(_pfile)
        if persist:
            projections = projections.persist()
        return vm, projections
    else:
        return vm
    
def compute_vmodes_1D(Nsqr, dzc=None, dzf=None, zc=None, zf=None, nmodes=_nmod_def, **kwargs): 
    """
    Nsqr is brunt-vaisala frequency at left z points (outer wthout bottom)
    """
    ### extract keywords
    kwgs = _dico_def.copy()
    kwgs.update(kwargs)
    free_surf, g, sigma = kwgs["free_surf"], kwgs["g"], kwgs["eig_sigma"]

    ### deal with vertical grids
    Nz = Nsqr.size
    if dzc is not None and dzf is not None:
        if zc is not None:
            dz_surf = abs(zc[0])
        else:
            dz_surf = .25*(dzc[0] + dzf[0]) ### this is NEMO grid
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
    if free_surf and g>0:
        invg = np.ones(1)/g
    else:
        invg = np.zeros(1)
    Nsqog = Nsqr[:1]*invg

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

    inds = np.isfinite(ev)
    ev, ef = ev[inds].real, ef[:,inds].real
    isort = np.argsort(ev)[:nmodes+1]
    ev, ef = ev[isort], ef[:,isort]
    ef *= np.sign(ef[0,:])[None,:] # positive pressure at the surface
    pmod, wmod = ef[:Nz,:], -ef[Nz:,:]
    
    return np.float64(1)/ev**.5, pmod, wmod

def _compute_vmodes_1D_stack(N2l, dzc, dzf, **kwargs):
    assert N2l.ndim==dzc.ndim==dzf.ndim==1
    lemask = ~np.isnan(dzc)
    c, p, w = compute_vmodes_1D(N2l[lemask], dzc=dzc[lemask], dzf=dzf[lemask], **kwargs)
    nans = np.full((dzc.size-lemask.sum(),c.size), np.nan)
    return np.vstack([c, p, nans, w, nans])

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
    metrics = {key: [va.name for va in val if va.name in ds.coords] for key,val in grid._metrics.items()}
    if ds is not None:
        metrics = {key: [va for va in val if va in ds.coords] for key,val in metrics.items()}
    return metrics

def _wrap_xgcm_grid_metrics(grid, ds=None):
    metrics = _get_xgcm_grid_metrics(grid, ds=ds)
    return {"metrics_"+''.join(key): list(key)+val for key,val in metrics.items()}

def _unwrap_xgcm_grid_metrics(metrics):
    return {tuple(val[0]): val[1:] for val in metrics.values()}
