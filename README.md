# ABC Model
A simple model coupling biosphere and atmosphere made fully differentiable using JAX.

## Installation
These instructions work on Linux and MacOS and assume that python with pip are installed already. See below for full instructions to install on Windows.
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
surface_layer_model = abcmodel.atmos.surface_layer.ObukhovSurfaceLayerModel()
mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(**cm.bulk_mixed_layer.model_kwargs)
cloud_model = abcmodel.atmos.clouds.CumulusModel()

# setup atmos model
atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
    surface_layer=surface_layer_model,
    mixed_layer=mixed_layer_model,
    clouds=cloud_model,
)

# setup coupler
abcoupler = abcmodel.ABCoupler(
    rad=rad_model,
    land=land_model,
    atmos=atmos_model,
)

# setup initial conditions for each model
rad_state = rad_model.init_state(**cm.standard_rad.state_kwargs)
land_state = land_model.init_state(**cm.jarvis_stewart.state_kwargs)
surface_layer_state = surface_layer_model.init_state(**cm.obukhov_surface_layer.state_kwargs)
mixed_layer_state = mixed_layer_model.init_state(**cm.bulk_mixed_layer.state_kwargs)
cloud_state = cloud_model.init_state()

# setup atmos state
atmos_state = atmos_model.init_state(
    surface=surface_layer_state,
    mixed=mixed_layer_state,
    clouds=cloud_state,
)

# finally we can use the coupler to bound everything in a initial state
state = abcoupler.init_state(
    rad_state,
    land_state,
    atmos_state,
)
```


All set - let's integrate our model by defining the timestepping and the run time.
```python
# time step [s]
dt = 60.0
# total run time [s]
runtime = 12 * 3600.0

time, trajectory = abcmodel.integrate(state, abcoupler, dt, runtime, tstart)
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

## Windows installation
On Windows we need some more utilities for jax to work properly, also installing python is not as straigforward. For jax to run, you first need the Microsoft Visual C++ redistributable found [here](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022), which will require a system restart to function. Note that you might want to install uv before the restart, as the PATH update might require a restart as well (see below).

Now clone the repo and cd into it. The following section shows how to set up a python environment with [uv](https://docs.astral.sh/uv/), if you have python with pip running you can skip it.

### UV environment 
First install uv via the terminal with 
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
You can check that uv is available and running by typing `uv` in your terminal, if you receive an error, you will have to add uv to your path manually or [restart your computer](https://github.com/astral-sh/uv/issues/10014). Here or when execution uv scripts to activate environments windows [execution policiy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.4#powershell-execution-policies) might try to stop you, if that is the case you need to change or bypass it.

After uv is installed and running, create a virtual environment in the abc-model directory by running 
```
uv venv --python 3.13.0
```
this will also show you the command needed to activate the venv, which should look similar to
```
.venv\Scripts\activate
```

Lastly, while in the abc-model directory, install the abc-model with uv:
```
uv pip install -e .
```

## See also
The model was constructed from the [CLASS model](https://github.com/classmodel/modelpy).
For a more advanced model, see [ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl).
