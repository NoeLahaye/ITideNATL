""" eos.py
Equation Of State powered for xarray and xgcm.Grid and adapted to NEMO output through xorca
This package merely contains a few routines borrowed from CDFTOOLS (https://github.com/meom-group/CDFTOOLS)
to compute basic things such as the density and the Brunt-Vaisala frequency
"""
import xgcm
import gsw

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
             "temp": "votemper", "salt": "vosaline", "pref":"depth_c"}

from itidenatl.utils import _parse_inp_dict

rdeltaS = 32.
r1_S0  = 0.875/35.16504
r1_T0  = 1./40.
r1_Z0  = 1.e-4
rau0 = 1026.

EOS000 = 8.0189615746e+02
EOS100 = 8.6672408165e+02
EOS200 = -1.7864682637e+03
EOS300 = 2.0375295546e+03
EOS400 = -1.2849161071e+03
EOS500 = 4.3227585684e+02
EOS600 = -6.0579916612e+01
EOS010 = 2.6010145068e+01
EOS110 = -6.5281885265e+01
EOS210 = 8.1770425108e+01
EOS310 = -5.6888046321e+01
EOS410 = 1.7681814114e+01
EOS510 = -1.9193502195
EOS020 = -3.7074170417e+01
EOS120 = 6.1548258127e+01
EOS220 = -6.0362551501e+01
EOS320 = 2.9130021253e+01
EOS420 = -5.4723692739
EOS030 = 2.1661789529e+01
EOS130 = -3.3449108469e+01
EOS230 = 1.9717078466e+01
EOS330 = -3.1742946532
EOS040 = -8.3627885467
EOS140 = 1.1311538584e+01
EOS240 = -5.3563304045
EOS050 = 5.4048723791e-01
EOS150 = 4.8169980163e-01
EOS060 = -1.9083568888e-01
EOS001 = 1.9681925209e+01
EOS101 = -4.2549998214e+01
EOS201 = 5.0774768218e+01
EOS301 = -3.0938076334e+01
EOS401 = 6.6051753097
EOS011 = -1.3336301113e+01
EOS111 = -4.4870114575
EOS211 = 5.0042598061
EOS311 = -6.5399043664e-01
EOS021 = 6.7080479603
EOS121 = 3.5063081279
EOS221 = -1.8795372996
EOS031 = -2.4649669534
EOS131 = -5.5077101279e-01
EOS041 = 5.5927935970e-01
EOS002 = 2.0660924175
EOS102 = -4.9527603989
EOS202 = 2.5019633244
EOS012 = 2.0564311499
EOS112 = -2.1311365518e-01
EOS022 = -1.2419983026
EOS003 = -2.3342758797e-02
EOS103 = -1.8507636718e-02
EOS013 = 3.7969820455e-01


def bvf2(ds, grid=None, boundary="extrapolate", **kwargs):
    """ compute Brunt-Vaisala frequency squared
    taken from eosbn2 (from CDFTOOLS function eosbn2 in eos.f90)
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

### taken from pre-processed eNATL60 NEMO 3.6 file eosbn2.f90
def rho_insitu(ds, inv_p=True, **kwargs):
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

def rho_gsw(ds, **kwargs):
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
    # this is not optimal : I should be able to get directly rho - r0 from gsw, since it should be how it is computed. 
    # But this is still faster than rho_insitu above
    r0 = gsw.rho(35.16504, 4, pdep) - gsw.rho(35.16504, 4, 0.)
    res = gsw.rho(salt, temp, pdep) - r0 - rau0
    return (res/rau0).astype(temp.dtype)

