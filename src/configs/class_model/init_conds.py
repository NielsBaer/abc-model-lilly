from abcmodel.clouds import StandardCumulusInitConds
from abcmodel.radiation import StandardRadiationInitConds
from abcmodel.surface_layer import StandardSurfaceLayerInitConds

THETA = 288.0

radiation = StandardRadiationInitConds(
    # net surface radiation [W/m²]
    net_rad=400,
)

surface_layer = StandardSurfaceLayerInitConds(
    # surface friction velocity [m s-1]
    ustar=0.3,
    # roughness length for momentum [m]
    z0m=0.02,
    # roughness length for scalars [m]
    z0h=0.002,
    # initial mixed-layer potential temperature [K]
    theta=THETA,
)

clouds = StandardCumulusInitConds()
