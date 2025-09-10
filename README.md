# ABC Model
A simple column model coupling biosphere and atmosphere.

## Installation
Install with
```
pip install git+https://git.bgc-jena.mpg.de/abc3/abc-model
```

or clone the repo and make an editable install inside your local repo using
```
pip install -e .
```

## Quick example
To setup the coupler we will always use 5 models:
1. Radiation model
2. Land surface model
3. Surface layer model
4. Mixed layer model
5. Cloud model

Every model needs two main arguments: `params`, `init_conds`, which can sometimes take a lot of arguments...
But worry not! We provide a config example (which we take from the [CLASS model](https://github.com/classmodel/modelpy)).
This can be loaded through the `abcconfigs` module:
```python
import abcconfigs.class_model as cm
```

Now we are ready to set up our models with ease...
We will do this using the `abcmodel` module, which is the _de facto_ module in this repository.
The first one is the radiation model, and we can use the standard model:
```python
from abcmodel.radiation import StandardRadiationModel

radiation_model = StandardRadiationModel(
        cm.standard_radiation.params,
        cm.standard_radiation.init_conds,
    )
```

Next on the list, the land surface. Let's use the simple Jarvis Stewart model:
```python
from abcmodel.land_surface import JarvisStewartModel

land_surface_model = JarvisStewartModel(
    cm.jarvis_stewart.params,
    cm.jarvis_stewart.init_conds,
)
```

Now the atmosphere. We will use the standard model for the surface layer,
the bulk model for the mixed layer and the cumulus model for the clouds. We set them
up just like before:
```python
from abcmodel.surface_layer import StandardSurfaceLayerModel
from abcmodel.mixed_layer import BulkMixedLayerModel
from abcmodel.clouds import StandardCumulusModel

# define surface layer model
surface_layer_model = StandardSurfaceLayerModel(
    cm.standard_surface_layer.params,
    cm.standard_surface_layer.init_conds,
)

# define mixed layer model
mixed_layer_model = BulkMixedLayerModel(
    cm.bulk_mixed_layer.params,
    cm.bulk_mixed_layer.init_conds,
)

# define cloud model
cloud_model = StandardCumulusModel(
    cm.standard_cumulus.params,
    cm.standard_cumulus.init_conds,
)
```

We have everything we need for our vertical column structure. Now we can setup the coupler
with these components together with the running time configuration. It typically goes as follows:

```python
from abcmodel import ABCoupler

# time step [s]
dt = 60.0
# total run time [s]
runtime = 12 * 3600.0

abc = ABCoupler(
    dt=dt,
    runtime=runtime,
    radiation=radiation_model,
    land_surface=land_surface_model,
    surface_layer=surface_layer_model,
    mixed_layer=mixed_layer_model,
    clouds=cloud_model,
)
```

All set - let's run it!
```python
abc.run()
```

All variables are stored under `diagnostics` (which are set by default, but you could also adjust that as you like).
Now we can make beautiful plots. Here's one example:
```python
import matplotlib.pyplot as plt

time = abc.get_t()
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.plot(time, abc.mixed_layer.diagnostics.get("abl_height"))
plt.xlabel("time [h]")
plt.ylabel("h [m]")

plt.subplot(234)
plt.plot(time, abc.mixed_layer.diagnostics.get("theta"))
plt.xlabel("time [h]")
plt.ylabel("theta [K]")

plt.subplot(232)
plt.plot(time, abc.mixed_layer.diagnostics.get("q") * 1000.0)
plt.xlabel("time [h]")
plt.ylabel("q [g kg-1]")

plt.subplot(235)
plt.plot(time, abc.clouds.diagnostics.get("cc_frac"))
plt.xlabel("time [h]")
plt.ylabel("cloud fraction [-]")

plt.subplot(233)
plt.plot(time, abc.land_surface.diagnostics.get("gf"))
plt.xlabel("time [h]")
plt.ylabel("ground heat flux [W m-2]")

plt.subplot(236)
plt.plot(time, abc.land_surface.diagnostics.get("le_veg"))
plt.xlabel("time [h]")
plt.ylabel("transpiration [W m-2]")

plt.tight_layout()
plt.show()
```

Which should give us something like the figure below.
![readme_example](figs/readme-example.png "readme example")

## Changing models, parameters and initial conditions
Now let's say you want to use a different model for the land surface.
Instead of the Jarvis Stewart model, you may choose AquaCrop.
We also provide a configuration for that, which you can load using the `abcconfigs` module, as previously done.
You can take a look at the config [here](https://git.bgc-jena.mpg.de/abc3/abc-model/-/blob/main/src/abcconfigs/class_model/aquacrop.py?ref_type=heads).

Now you may change from C3 to C4 with something like
```python
my_config = cm.land_surface.jarvis_stewart.params
my_config.c3c4 = "c4"
```

and define your new land surface model as
```python
from abcmodel.land_surface import AquaCropModel

land_surface_model = AquaCropModel(
    my_config,
    cm.aquacrop.init_conds,
)
```

then you can redefine the `ABCoupler` using this new model and run it to see different outcomes.


## See also
The model was constructed from the [CLASS model](https://github.com/classmodel/modelpy).
For a more advanced model, see [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl).
