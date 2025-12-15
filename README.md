# ABC Model
A simple model coupling biosphere and atmos, but made fully differentiable using JAX.

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

Each model is a class that is initialized with model-specific parameters. We provide a config example (which we take from the [CLASS model](https://github.com/classmodel/modelpy)).
This can be loaded through the `abcconfigs` module:
```python
import abcconfigs.class_model as cm
```

Now we are ready to set up our models with ease...
We will do this using the `abcmodel` module, which is the _de facto_ module in this repository.
```python
import abcmodel

# setup models
rad_model = abcmodel.rad.StandardRadiationModel(**cm.standard_rad.model_kwargs)
land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)
surface_layer_model = abcmodel.surface_layer.StandardSurfaceLayerModel() # no parameters
mixed_layer_model = abcmodel.mixed_layer.BulkMixedLayerModel(**cm.bulk_mixed_layer.model_kwargs)
cloud_model = abcmodel.clouds.StandardCumulusModel() # no parameters
```

We have everything we need for our vertical column structure. Now we can setup the coupler
with these components together with the running time configuration.
```python
abcoupler = abcmodel.ABCoupler(
    rad=rad_model,
    land=land_model,
    surface_layer=surface_layer_model,
    mixed_layer=mixed_layer_model,
    clouds=cloud_model,
)
```

To setup the initial condition, we also take advantage of a ready-made config, and we can setup initial conditions for each model.
```python
# setup initial condition
rad_init_conds = abcmodel.rad.StandardRadiationInitConds(
    **cm.standard_rad.init_conds_kwargs
)
land_init_conds = abcmodel.land.JarvisStewartInitConds(
    **cm.jarvis_stewart.init_conds_kwargs,
)
surface_layer_init_conds = abcmodel.surface_layer.StandardSurfaceLayerInitConds(
    **cm.obukhov_surface_layer.init_conds_kwargs
)
mixed_layer_init_conds = abcmodel.mixed_layer.BulkMixedLayerInitConds(
    **cm.bulk_mixed_layer.init_conds_kwargs,
)
cloud_init_conds = abcmodel.clouds.StandardCumulusInitConds()
```

Finally we can use the coupler to bound everything in a initial state that will be carried by our model.
```python
state = abcoupler.init_state(
    rad_init_conds,
    land_init_conds,
    surface_layer_init_conds,
    mixed_layer_init_conds,
    cloud_init_conds,
)
```


All set - let's integrate our model by defining the timestepping and the run time.
```python
# time step [s]
dt = 60.0
# total run time [s]
runtime = 12 * 3600.0

time, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)
```

To plot the results, we will typically follow something like the code below.
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.plot(time, trajectory.abl_height)
plt.xlabel("time [h]")
plt.ylabel("h [m]")

plt.subplot(234)
plt.plot(time, trajectory.theta)
plt.xlabel("time [h]")
plt.ylabel("theta [K]")

plt.subplot(232)
plt.plot(time, trajectory.q * 1000.0)
plt.xlabel("time [h]")
plt.ylabel("q [g kg-1]")

plt.subplot(235)
plt.plot(time, trajectory.cc_frac)
plt.xlabel("time [h]")
plt.ylabel("cloud fraction [-]")

plt.subplot(233)
plt.plot(time, trajectory.gf)
plt.xlabel("time [h]")
plt.ylabel("ground heat flux [W m-2]")

plt.subplot(236)
plt.plot(time, trajectory.le_veg)
plt.xlabel("time [h]")
plt.ylabel("latent heat flux from vegetation [W m-2]")

plt.tight_layout()
plt.show()
```

Which should give us something like the figure below.
![readme_example](figs/readme-example.png "readme example")

## Changing models, parameters and initial conditions
Now let's say you want to use a different model for the land surface.
Instead of the Jarvis Stewart model, you may choose Ags.
We also provide a configuration for that, which you can load using the `abcconfigs` module, as previously done.
You can take a look at the config [here](https://git.bgc-jena.mpg.de/abc3/abc-model/-/blob/main/src/abcconfigs/class_model/ags.py?ref_type=heads).

Now, for example, you may change from C3 to C4 with something like the following.
```python
# new parameters definition
ags_model_kwargs = cm.ags.model_kwargs
ags_model_kwargs['c3c4'] = 'c4'

# define a new land model
land_model = abcmodel.land.AgsModel(**ags_model_kwargs)
```

Then you can redefine the coupler, create a new state and integrate it to see different outcomes. You can do something similar to change initial conditions, or even recreate your own!

It is also possible to jit the model, but please remember to filter functions, which can be easily done with `equinox.filter_jit(integrate)`. Jitting a run for analysis does not bring any benefit right now, but it becomes important to accelerate eventual training loops for calibration of parameters.

## See also
The model was constructed from the [CLASS model](https://github.com/classmodel/modelpy).
For a more advanced model, see [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl).
