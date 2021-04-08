# Notes eNATL60 on occigen

Files physical structure:

```
dimensions:
	axis_nbounds = 2 ;
	x = 8354 ;
	y = 4729 ;
	deptht = 300 ;
	time_counter = UNLIMITED ; // (24 currently)

	float votemper(time_counter, deptht, y, x) ;
		votemper:standard_name = "sea_water_potential_temperature" ;

		votemper:_Storage = "chunked" ;
		votemper:_ChunkSizes = 1, 1, 10, 8354 ;
		votemper:_DeflateLevel = 2 ;
		votemper:_Endianness = "little" ;

```

[Notes Aurélie](https://github.com/auraoupa/scripts-occigen-for-Arne)

[eNATL60](https://github.com/ocean-next/eNATL60/blob/master/02_experiment-setup.md)

[xarray netcdf chunking](https://github.com/pydata/xarray/issues/1440)

[CINES stockage](https://www.cines.fr/calcul/organisation-des-espaces-de-donnees/espaces-de-donnees-quotas-disques-restaurations-de-fichiers/)

Create vnc server on dunree:

- on dunree: `vncserver :2 -localhost -geometry 1400x1000`
- to kill server: `vncserver -kill :2`
- on baliste: `ssh -L 1234:localhost:5902 -C -N -l aponte dunree`
- on baliste: launch tigervnc with the address `localhost:1234` and appropriate password
