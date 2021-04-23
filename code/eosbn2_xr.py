import xgcm

### these definition could re-use xorca defs
_grav = 9.81
_zmet = "e3w"
_temp = "votemper"
_salt = "vosaline"
_zdim = "Z"
_zcoord = "depth_l"

def eosbn2(ds, grid=None, boundary="extend", z_reverse=True):
    """ eosbn2 (from CDFTOOLS function eosbn2 in eos.f90)
    if z_reverse: assume that depth is negative and z-grid oriented downwards while weights are positive
    """
    temp = _temp
    salt = _salt
    zdim = _zdim
    zmetric = _zmet
    grav = _grav
    zcoord = _zcoord
    
    if grid is None: # no xgcm grid passed: use one without metrics (assume mid-depth points = w points)
        grid = xgcm.Grid(ds)

    z_sign = -1. if z_reverse else 1.

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

    return eosbn2
