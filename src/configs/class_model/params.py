from abcmodel.clouds import StandardCumulusParams
from abcmodel.radiation import StandardRadiationParams
from abcmodel.surface_layer import StandardSurfaceLayerParams

radiation = StandardRadiationParams(
    # latitude [deg]
    lat=51.97,
    # longitude [deg]
    lon=-4.93,
    # day of the year [-]
    doy=268.0,
    # time of the day [h UTC]
    tstart=6.8,
    # cloud cover fraction [-]
    cc=0.0,
    # cloud top radiative divergence [W m-2]
    dFz=0.0,
)

surface_layer = StandardSurfaceLayerParams()

clouds = StandardCumulusParams()
