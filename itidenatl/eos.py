""" eos.py
Equation Of State powered for xarray and xgcm.Grid and adapted to NEMO output through xorca
This package merely contains a few routines borrowed from CDFTOOLS (https://github.com/meom-group/CDFTOOLS)
to compute basic things such as the density and the Brunt-Vaisala frequency

TODO:
    - use xarray wrapper for gsw library
    - test new implementation of bvf2 and compare with gsw library
"""
import xgcm
try:
    import gsw_xarray as gsw
except:
    print("eos.py: package gsw_xarray not found. Using standard gsw instead")
    import gsw
import xarray as xr

### these definition could re-use xorca defs
#_grav = 9.81
#_zmet = "e3w"
#_temp = "votemper"
#_salt = "vosaline"
#_zdim = "Z"
#_zcoord = "depth_l"
#_var_names = {"temp": "votemper", "salt": "vosaline", "pref": "depth_c"}

_defo_dico = {"grav": 9.81, "z_inv": True,
             "zmet": "e3w", "zdim":"Z", "zcoord":"depth_l",
             "temp": "votemper", "salt": "vosaline", "pref":"depth_c", 
             "alpha":"alpha", "beta":"beta"}

from itidenatl.utils import _parse_inp_dict
from .eos_coefs import *


################################################################################
########## - - - Taken from pre-processed fortran files of eNATL60 - - - #######
################################################################################

### taken from pre-processed eNATL60 NEMO 3.6 file eosbn2.f90
def rho_insitu(ds, inv_p=True, **kwargs):
    """ compute in-situ density field following TEOS-10 implementation [Roquet et al 2015]
    A constant vertical profile is substracted, following NEMO implementation.
    Work from conservative temperature and absolute salinity.
    """
    var_names = _defo_dico.copy()
    var_names.update(kwargs)

    pdep = ds[var_names["pref"]]
    if inv_p:
        pdep = -pdep
    res = rho_insitu_tsp(ds[var_names["temp"]], ds[var_names["salt"]], pdep)
    if "tmask" in ds:
        res = res.where(ds.tmask)
    return res

def rho_insitu_tsp(temp, salt, pdep):
    # eos_insitu(pts, prd, pdep)
    #!----------------------------------------------------------------------
    #!                   ***  ROUTINE eos_insitu  ***
    #!
    #! ** Purpose :   Compute the in situ density (ratio rho/rau0) from
    #!       potential temperature and salinity using an equation of state
    #!       defined through the namelist parameter nn_eos.
    #!
    #! ** Method  :   prd(t,s,z) = ( rho(t,s,z) - rau0 ) / rau0
    #!         with   prd    in situ density anomaly      no units
    #!                t      TEOS10: CT or EOS80: PT      Celsius
    #!                s      TEOS10: SA or EOS80: SP      TEOS10: g/kg or EOS80: psu
    #!                z      depth                        meters
    #!                rho    in situ density              kg/m^3
    #!                rau0   reference density            kg/m^3
    #!
    #!     nn_eos = -1 : polynomial TEOS-10 equation of state is used for rho(t,s,z).
    #!         Check value: rho = 1028.21993233072 kg/m^3 for z=3000 dbar, ct=3 Celcius, sa=35.5 g/kg
    #!
    #!     nn_eos =  0 : polynomial EOS-80 equation of state is used for rho(t,s,z).
    #!         Check value: rho = 1028.35011066567 kg/m^3 for z=3000 dbar, pt=3 Celcius, sp=35.5 psu
    #!
    #!     nn_eos =  1 : simplified equation of state
    #!              prd(t,s,z) = ( -a0*(1+lambda/2*(T-T0)+mu*z+nu*(S-S0))*(T-T0) + b0*(S-S0) ) / rau0
    #!              linear case function of T only: rn_alpha<>0, other coefficients = 0
    #!              linear eos function of T and S: rn_alpha and rn_beta<>0, other coefficients=0
    #!              Vallis like equation: use default values of coefficients
    #!
    #! ** Action  :   compute prd , the in situ density (no units)
    #!
    #! References :   Roquet et al, Ocean Modelling, in preparation (2014)
    #!                Vallis, Atmospheric and Oceanic Fluid Dynamics, 2006
    #!                TEOS-10 Manual, 2010
    #!----------------------------------------------------------------------
    #     REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::   pts   ! 1 : potential temperature  [Celcius]
    #                                                               ! 2 : salinity               [psu]
    #     REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(  out) ::   prd   ! in situ density            [-]
    #     REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(in   ) ::   pdep  ! depth                      [m]
    #
    #     INTEGER  ::   ji, jj, jk                ! dummy loop indices
    #     REAL(wp) ::   zt , zh , zs , ztm        ! local scalars
    #     REAL(wp) ::   zn , zn0, zn1, zn2, zn3   !   -      -
    #!----------------------------------------------------------------------
    zh  = pdep * r1_Z0                                  # depth
    zt  = temp * r1_T0                           # temperature
    zs  = ( abs( salt + rdeltaS ) * r1_S0 )**.5   # square root salinity
    
    zn3 = EOS013 * zt + EOS103 * zs + EOS003
    
    zn2 = (EOS022 * zt + EOS112 * zs + EOS012) * zt \
           + (EOS202 * zs + EOS102) * zs + EOS002
    
    zn1 = (((EOS041 * zt + EOS131 * zs + EOS031) * zt   \
           + (EOS221 * zs + EOS121) * zs + EOS021) * zt   \
           + ((EOS311 * zs + EOS211) * zs + EOS111) * zs + EOS011) * zt   \
           + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs + EOS001
    
    zn0 = (((((EOS060 * zt  + EOS150 * zs + EOS050) * zt   \
           + (EOS240 * zs + EOS140) * zs + EOS040) * zt   \
           + ((EOS330 * zs + EOS230) * zs + EOS130) * zs + EOS030) * zt   \
           + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs + EOS020) * zt   \
           + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110) * zs + EOS010) * zt   \
           + (((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs + EOS100) * zs + EOS000
    
    zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0
    
    prd = (  zn / rau0 - 1. ) # * ztm  # density anomaly (masked) # NJAL: tmask (ztm) removed
    
    return prd.astype(temp.dtype)

def ts_expan_ratio(ds_or_da, salt=None, pdep=None, inv_p=True, **kwargs):
    """  Calculates thermal/haline expansion ratio; from conservative temperature and absolute salinity
    
    Parameters
    __________
    Two xr.DataArray or one xr.Dataset for temperature and sallinity 

    Returns
    _______
    Two xr.DataArray for thermal and haline expansion ratio

    Taken from CDFtools: SUBROUTINE rab_3d( pts, pab )
    !!----------------------------------------------------------------------
    !!                 ***  ROUTINE rab_3d  ***
    !!
    !! ** Purpose :   Calculates thermal/haline expansion ratio at T-points
    !!
    !! ** Method  :   calculates alpha / beta at T-points
    !!
    !! ** Action  : - pab     : thermal/haline expansion ratio at T-points
    !!----------------------------------------------------------------------
          REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::   pts   ! pot. temperature & salinity
          REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(  out) ::   pab   ! thermal/haline expansion ratio
    !
          INTEGER  ::   ji, jj, jk                ! dummy loop indices
          REAL(wp) ::   zt , zh , zs , ztm        ! local scalars
          REAL(wp) ::   zn , zn0, zn1, zn2, zn3   !   -      -
    !!----------------------------------------------------------------------
    """

    if isinstance(ds_or_da, xr.Dataset):
        var_names = _defo_dico.copy()
        var_names.update(kwargs)
        temp = ds_or_da[var_names["temp"]]
        salt = ds_or_da[var_names["salt"]]
        pdep = ds_or_da[var_names["pref"]]
    else:
        assert (salt is not None) and (pdep is not None)
        temp = ds_or_da
    if inv_p:
        pdep = -pdep

    ### Prep
    zh  = pdep * r1_Z0                                # depth
    zt  = temp * r1_T0                           # temperature
    zs  = ( abs( salt + rdeltaS ) * r1_S0 )**.5   # square root salinity

    ### alpha (thermal expansion ratio)
    zn3 = ALP003

    zn2 = ALP012 * zt + ALP102 * zs + ALP002

    zn1 = ( ( ALP031 * zt + ALP121 * zs + ALP021 ) * zt \
            + ( ALP211 * zs + ALP111 ) * zs + ALP011 ) * zt \
          + ( ( ALP301 * zs + ALP201 ) * zs + ALP101 ) * zs + ALP001

    zn0 = ( ( ( ( ALP050 * zt + ALP140 * zs + ALP040 ) * zt \
               + ( ALP230 * zs + ALP130 ) * zs + ALP030 ) * zt \
             + ( ( ALP320 * zs + ALP220 ) * zs + ALP120 ) * zs + ALP020 ) * zt \
          + ( ( ( ALP410 * zs + ALP310 ) * zs + ALP210 ) * zs + ALP110 ) * zs + ALP010 ) * zt \
          + ( ( ( ( ALP500 * zs + ALP400 ) * zs + ALP300 ) * zs + ALP200 ) * zs + ALP100 ) * zs + ALP000

    zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0
    alt = ( zn * r1_rau0 ).rename("alpha")

    ### beta (haline expansion ratio)
    zn3 = BET003

    zn2 = BET012 * zt + BET102 * zs + BET002

    zn1 = ( ( BET031 * zt + BET121 * zs + BET021 ) * zt \
           + ( BET211 * zs + BET111 ) * zs + BET011 ) * zt \
           + ( ( BET301 * zs + BET201 ) * zs + BET101 ) * zs + BET001

    zn0 = ( ( ( ( BET050 * zt + BET140 * zs + BET040 ) * zt \
               + ( BET230 * zs + BET130 ) * zs + BET030 ) * zt \
             + ( ( BET320 * zs + BET220 ) * zs + BET120 ) * zs + BET020 ) * zt \
           + ( ( ( BET410 * zs + BET310 ) * zs + BET210 ) * zs + BET110 ) * zs + BET010 ) * zt \
           + ( ( ( ( BET500 * zs + BET400 ) * zs + BET300 ) * zs + BET200 ) * zs + BET100 ) * zs + BET000

    zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0
    bas = ( zn / zs * r1_rau0 ).rename("beta")

    ### Finalize and return result
    if "tmask" in ds_or_da:
        alt, bas = alt.where(ds_or_da.tmask), bas.where(ds_or_da.tmask)
    return alt, bas

def bvf2(ds, grid=None, boundary="extrapolate", **kwargs):
    """ compute Brunt-Vasiala Frequency. Implementation followint NEMO, TEOS-10-based routine
    taken from SUBROUTINE bn2( pts, pab, pn2 ).
    From conservative temperature and absolute salinity.

    !!----------------------------------------------------------------------
    !!                  ***  ROUTINE bn2  ***
    !!
    !! ** Purpose :   Compute the local Brunt-Vaisala frequency at the
    !!                time-step of the input arguments
    !!
    !! ** Method  :   pn2 = grav * (alpha dk[T] + beta dk[S] ) / e3w
    !!      where alpha and beta are given in pab, and computed on T-points.
    !!      N.B. N^2 is set one for all to zero at jk=1 in istate module.
    !!
    !! ** Action  :   pn2 : square of the brunt-vaisala frequency at w-point
    !!
    !!----------------------------------------------------------------------
          REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::  pts   ! pot. temperature and salinity   [Celcius,psu]
          REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::  pab   ! thermal/haline expansion coef.  [Celcius-1,psu-1]
          REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(  out) ::  pn2   ! Brunt-Vaisala frequency squared [1/s^2]
    !
          INTEGER  ::   ji, jj, jk      ! dummy loop indices
          REAL(wp) ::   zaw, zbw, zrw   ! local scalars
    !!----------------------------------------------------------------------
    """
    dico = _parse_inp_dict(kwargs, _defo_dico)
    temp, salt = ds[dico["temp"]], ds[dico["salt"]]
    zdim, zmetric, zcoord = dico["zdim"], dico["zmet"], dico["zcoord"]
    grav, z_inv = dico["grav"], dico["z_inv"]
    pdep = ds[dico["pref"]]
    if z_inv:
        pdep = -pdep
    alt, bes = dico["alpha"], dico["beta"]
    if alt in ds and bes in ds:
        alt, bes = ds[alt], ds[bes]
    else:
        alt, bes = ts_expan_ratio(temp, salt, pdep, inv_p=False)

    if grid is None: # no xgcm grid passed: use one without metrics (assume mid-depth points = w points)
        grid = xgcm.Grid(ds, periodic=False)

    z_sign = -1. if z_inv else 1. # depth increases with index

    if True:
        zaw = grid.interp(alt, zdim, boundary="extrapolate")
        zbw = grid.interp(bes, zdim, boundary="extrapolate")
    else:
        raise NotImplementedError("True interpolation not implemented in bnsq")
        # this could be achieved calling itidenatl.gridop.interp_z, but requires depth at T-levels and w-levels
               #zrw =   ( gdepw_n(ji,jj,jk) - gdept_n(ji,jj,jk) )   &
                  #&  / ( gdept_n(ji,jj,jk-1) - gdept_n(ji,jj,jk) ) 

               #zaw = pab(ji,jj,jk,jp_tem) * (1. - zrw) + pab(ji,jj,jk-1,jp_tem) * zrw 
               #zbw = pab(ji,jj,jk,jp_sal) * (1. - zrw) + pab(ji,jj,jk-1,jp_sal) * zrw

               #pn2(ji,jj,jk) = grav * (  zaw * ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) )     &
                  #&                    - zbw * ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )  )  &

    if "_metrics" in dir(grid):
        try:
            eosbn2 = z_sign * grav * ( zaw * grid.derivative(temp, zdim, boundary=boundary) 
                             - zbw * grid.derivative(salt, zdim, boundary=boundary) )
        except:
            if zmetric not in ds:
                raise ValueError("no metric found in dataset")
            eosbn2 = z_sign * grav * ( zaw * grid.diff(temp, zdim, boundary=boundary) 
                             - zbw * grid.diff(salt, zdim, boundary=boundary) ) / ds[zmetric]
    else:
        eosbn2 = z_sign * grav * ( zaw * grid.diff(temp, zdim, boundary=boundary) 
                             - zbw * grid.diff(salt, zdim, boundary=boundary) ) / ds[zmetric]
    
    return eosbn2.rename("bvf")


################################################################################
############### - - - Wrappers of GSW library - - - ############################
################################################################################

def rho_gsw(ds, inv_p=True, **kwargs):
    """ returns reduced in-situ density anomaly (with respect ro background profile)
    i.e. r/rho0, whith rho = rho0 + r(x,y,z,t) + r0(z)
    wapper around gsw routine based on TEOS-10 [Roquet elt al 2015]. 
    Result is the same as eNATL60-derived rho_insitu routine above.
    """
    var_names = _defo_dico.copy()
    var_names.update(kwargs)

    pdep = ds[var_names["pref"]]
    if inv_p:
        pdep = -pdep
    res = rho_gsw_tsp(ds[var_names["temp"]], ds[var_names["salt"]], pdep)
    if "tmask" in ds:
        res = res.where(ds.tmask)
    return res

def rho_gsw_tsp(temp, salt, pdep):
    """ wrapper around gsw.rho to compute the in-situ density from conservative temperature and absolute salinity (TEOS-10). 
    A mean vertical profile is subtracted to recoevr the same behaviour as NEMO implementation.
    Result is the same as rho_insitu routine above.
    
    See also
    ________
    eos.rho_gsw
    """
    # this is not optimal : I should be able to get directly rho - r0 from gsw, since it should be how it is computed. 
    # But this is still faster than rho_insitu above
    r0 = gsw.rho(35.16504, 4, pdep) - gsw.rho(35.16504, 4, 0.)
    res = (gsw.rho(salt, temp, pdep) - r0)/rau0 - 1.
    return res.astype(temp.dtype)


################################################################################
##################### Old routines from CDFtools.f90 ###########################
################################################################################
# note that most of (if not all) these routines work with potential temperature and practical salinity

def bvf2_cdftools(ds, grid=None, boundary="extrapolate", **kwargs):
    """ compute Brunt-Vaisala frequency squared
    taken from eosbn2 (from CDFTOOLS function eosbn2 in eos.f90)
    uses potential temperature and practical salinity
    if z_inv: assume that depth is negative and z-grid oriented downwards while weights are positive
    """
    dico = _parse_inp_dict(kwargs, _defo_dico)
    temp, salt = dico["temp"], dico["salt"]
    zdim, zmetric, zcoord = dico["zdim"], dico["zmet"], dico["zcoord"]
    grav, z_inv = dico["grav"], dico["z_inv"]
    
    if grid is None: # no xgcm grid passed: use one without metrics (assume mid-depth points = w points)
        grid = xgcm.Grid(ds, periodic=False)

    z_sign = -1. if z_inv else 1.

    zt = grid.interp(ds[temp], zdim, boundary=boundary)
    zs = grid.interp(ds[salt], zdim, boundary=boundary) - 35.0
    zh = z_sign * ds[zcoord] # take minus (i.e. positive depth)

    zalbet = ( ( ( - 0.255019e-07 * zt + 0.298357e-05 ) * zt     #   ! ratio alpha/beta
                                       - 0.203814e-03 ) * zt   
                                       + 0.170907e-01 ) * zt \
           + 0.665157e-01                                    \
           +     ( - 0.678662e-05 * zs                         
                   - 0.846960e-04 * zt + 0.378110e-02 ) * zs \
           +   ( ( - 0.302285e-13 * zh                         
                   - 0.251520e-11 * zs                         
                   + 0.512857e-12 * zt * zt           ) * zh   
                   - 0.164759e-06 * zs                         
                +(   0.791325e-08 * zt - 0.933746e-06 ) * zt   
                                       + 0.380374e-04 ) * zh

    zbeta  = ( ( -0.415613e-09 * zt + 0.555579e-07 ) * zt        #   ! beta
                                    - 0.301985e-05 ) * zt     \
           + 0.785567e-03                                     \
           + (     0.515032e-08 * zs                           
                 + 0.788212e-08 * zt - 0.356603e-06 ) * zs    \
           +(  (   0.121551e-17 * zh                           
                 - 0.602281e-15 * zs                           
                 - 0.175379e-14 * zt + 0.176621e-12 ) * zh     
                                     + 0.408195e-10   * zs     
             + ( - 0.213127e-11 * zt + 0.192867e-09 ) * zt     
                                     - 0.121555e-07 ) * zh
        
    if "_metrics" in dir(grid):
        try:
            eosbn2 = z_sign * grav * zbeta * ( zalbet * grid.derivative(ds[temp], zdim, boundary=boundary) 
                             - grid.derivative(ds[salt], zdim, boundary=boundary) )
        except:
            if zmetric not in ds:
                raise ValueError("no metric found in dataset")
            eosbn2 = z_sign * grav * zbeta * ( zalbet * grid.diff(ds[temp], zdim, boundary=boundary) 
                             - grid.diff(ds[salt], zdim, boundary=boundary) ) / ds[zmetric]
    else:
        eosbn2 = z_sign * grav * zbeta * ( zalbet * grid.diff(ds[temp], zdim, boundary=boundary) 
                             - grid.diff(ds[salt], zdim, boundary=boundary) ) / ds[zmetric]

    return eosbn2.rename("bvf")

def sigmai(ds, inv_p=True, **kwargs):
    """ sigmai_tsp: potential density referenced at depth pref
    adapted from CDFTOOLS sigmai_dep2d (in eos.f90)
    Purpose : Compute the  density referenced to pref (ratio rho/rau0) 
            from potential temperature and salinity fields
            using an equation of state defined through the amelist parameter neos.
    Internally calls sigmai_tps

    Parameters
    __________
    ds: xarray Dataset
        contains potential temperature, practical salinity and depth
    var_names: dict, optional
        names of variables. default: {"temp": "votemper", "salt": "vosaline", "pref": "depth_c"}
    inv_p: bool, optional (default: True)
        whether sign of depth/pressure must be changed

    Outputs
    ______
    sigmai: xarray.DatarArray
        potential density referenced at depth pref

     Notes
      ____
    #!!       Jackett and McDougall (1994) equation of state.
    #!!         the in situ density is computed directly as a function of
    #!!         potential temperature relative to the surface (the opa t
    #!!         variable), salt and pressure (assuming no pressure variation
    #!!         along geopotential surfaces, i.e. the pressure p in decibars
    #!!         is approximated by the depth in meters.
    #!!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
    #!!              rhop(t,s)  = rho(t,s,0)
    #!!         with pressure                      p        decibars
    #!!              potential temperature         t        deg celsius
    #!!              salinity                      s        psu
    #!!              reference volumic mass        rau0     kg/m**3
    #!!              in situ volumic mass          rho      kg/m**3
    #!!              in situ density anomalie      prd      no units
    #!! --------------------------------------------------------------------
    """
    var_names = _defo_dico.copy()
    var_names.update(kwargs)

    pref = ds[var_names["pref"]]
    if inv_p:
        pref = -pref
    sigmai = sigmai_tsp(ds[var_names["temp"]], ds[var_names["salt"]], pref)
    return sigmai
    
def sigmai_tsp( temp, salt, pref):
    """ sigmai_tsp: potential density referenced at depth pref
    adapted from CDFTOOLS sigmai_dep2d (in eos.f90)
    Purpose : Compute the  density referenced to pref (ratio rho/rau0) 
            from potential temperature and salinity fields 
            using an equation of state defined through the amelist parameter neos.
    Automatic broadcasting of input arrays will be performed

    Parameters
    __________
    temp: numpy array or xarray.datarray
        potential temperature
    salt: numpy array or xarray.datarray
        practical salinity
    pref: numpy array or xarray.datarray
        reference pressure in decibar (depth)

    Outputs
    ______
    sigmai: numpy array or xarray.DatarAray
        potential density referenced at depth pref

    Notes
    ____
  #!!       Jackett and McDougall (1994) equation of state.
  #!!         the in situ density is computed directly as a function of
  #!!         potential temperature relative to the surface (the opa t
  #!!         variable), salt and pressure (assuming no pressure variation
  #!!         along geopotential surfaces, i.e. the pressure p in decibars
  #!!         is approximated by the depth in meters.
  #!!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
  #!!              rhop(t,s)  = rho(t,s,0)
  #!!         with pressure                      p        decibars
  #!!              potential temperature         t        deg celsius
  #!!              salinity                      s        psu
  #!!              reference volumic mass        rau0     kg/m**3
  #!!              in situ volumic mass          rho      kg/m**3
  #!!              in situ density anomalie      prd      no units
  #!! --------------------------------------------------------------------
  """

    #if kwargs.get("with_persist", False):
        ## this feature is very likely useless, if not dangerous
        #temp = temp.persist()
        #salt = salt.persist()

    dpr4, dpd, dprau0 = 4.8314e-4, -2.042967e-2, 1000.
    dlrs = abs(salt)**.5

    # Compute the volumic mass of pure water at atmospheric pressure.
    dlr1 = (((( 6.536332e-9 * temp - 1.120083e-6) *
             temp + 1.001685e-4) *
             temp - 9.095290e-3) *
             temp + 6.793952e-2) * temp + 999.842594
    # Compute the seawater volumic mass at atmospheric pressure.
    dlr2 = ((( 5.3875e-9 * temp - 8.2467e-7) *
            temp + 7.6438e-5) *
            temp - 4.0899e-3) * temp + 0.824493

    dlr3 = ( -1.6546e-6 * temp + 1.0227e-4) * temp - 5.72466e-3

    # Compute the potential volumic mass (referenced to the surface).
    dlrhop = ( dpr4 * salt + dlr3 * dlrs + dlr2 ) * salt + dlr1

    # Compute the compression terms.
    dle = ( -3.508914e-8 * temp - 1.248266e-8 ) * temp - 2.595994e-6

    dlbw = ( 1.296821e-6 * temp - 5.782165e-9 ) * temp + 1.045941e-4

    dlb = dlbw + dle * salt

    dlc = ( -7.267926e-5 * temp + 2.598241e-3 ) * temp + 0.1571896

    dlaw = (( 5.939910e-6 * temp + 2.512549e-3 ) * temp - 0.1028859 ) * \
                temp - 4.721788

    dla = ( dpd * dlrs + dlc ) * salt + dlaw

    dlb1 = ( -0.1909078 * temp + 7.390729 ) * temp - 55.87545

    dla1 = (( 2.326469e-3 * temp + 1.553190 ) * temp - 65.00517) * temp + 1044.077

    dlkw = ((( -1.361629e-4 * temp - 1.852732e-2) * temp - 30.41638 ) * \
                temp + 2098.925 ) * temp + 190925.6

    dlk0 = ( dlb1 * dlrs + dla1 ) * salt + dlkw

    # Compute the potential density anomaly.
    sigmai = dlrhop / (1. - pref / (dlk0 - pref * (dla - pref * dlb))) - dprau0

    return sigmai.astype(temp.dtype)


