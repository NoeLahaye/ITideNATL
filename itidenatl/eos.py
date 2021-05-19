""" eos.py
Equation Of State powered for xarray and xgcm.Grid and adapted to NEMO output through xorca
This package merely contains a few routines borrowed from CDFTOOLS (https://github.com/meom-group/CDFTOOLS)
to compute basic things such as the density and the Brunt-Vaisala frequency
"""
import xgcm

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

def sigmai(ds, var_names=_defo_dico, inv_p=True):
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
    pref = ds[var_names["pref"]]
    if inv_p:
        pref = -pref
    sigmai = sigmai_tsp(ds[var_names["temp"]], ds[var_names["salt"]], pref)
    return sigmai
    
def sigmai_tsp( temp, salt, pref ):
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
