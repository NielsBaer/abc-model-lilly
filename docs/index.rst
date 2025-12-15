.. abc-model documentation master file, created by
   sphinx-quickstart on Fri Oct 17 16:03:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

ABC Model
=========

The ABC Model is a simple model coupling the biosphere and atmos,
made fully differentiable using `JAX`_ highly inspired by the `CLASS model`_
with the goal to serve as a simple tool for experiments.

.. _JAX: https://docs.jax.dev/en/latest/index.html
.. _CLASS model: https://classmodel.github.io

This documentation provides installation instructions, a complete tutorial,
and a detailed API reference for all model components.

Installation
------------

Install directly from the repository with ``pip``:

.. code-block:: bash

   pip install git+https://git.bgc-jena.mpg.de/abc3/abc-model

Or for development, clone the repository and perform an editable install:

.. code-block:: bash

   git clone https://git.bgc-jena.mpg.de/abc3/abc-model.git
   cd abc-model
   pip install -e .

Quick Example
-------------

To set up the coupler, we will use 5 components:

1. Radiation model
2. Land surface model
3. Surface layer model
4. Mixed layer model
5. Cloud model

We provide an example configuration taken from the CLASS model, which can be
loaded through the ``abcconfigs`` module.

.. code-block:: python

   import abcconfigs.class_model as cm
   import abcmodel

   # 1. Setup models
   rad_model = abcmodel.rad.StandardRadiationModel(**cm.standard_rad.model_kwargs)
   land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)
   surface_layer_model = abcmodel.surface_layer.StandardSurfaceLayerModel()
   mixed_layer_model = abcmodel.mixed_layer.BulkMixedLayerModel(**cm.bulk_mixed_layer.model_kwargs)
   cloud_model = abcmodel.clouds.StandardCumulusModel()

   # 2. Setup the coupler with the components
   abcoupler = abcmodel.ABCoupler(
       rad=rad_model,
       land=land_model,
       surface_layer=surface_layer_model,
       mixed_layer=mixed_layer_model,
       clouds=cloud_model,
   )

   # 3. Setup initial conditions
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

   # 4. Bind everything into an initial state
   state = abcoupler.init_state(
       rad_init_conds,
       land_init_conds,
       surface_layer_init_conds,
       mixed_layer_init_conds,
       cloud_init_conds,
   )

   # 5. Integrate the model
   time, trajectory = abcmodel.integrate(
       state, abcoupler, dt=60.0, runtime=12 * 3600.0
   )

Plotting the results should give us the following figure:

.. image:: ../figs/readme-example.png
   :alt: Example output plot
   :align: center

Changing Models and Parameters
------------------------------

You can easily swap out models or change their parameters. For example, to
replace the ``JarvisStewartModel`` with ``AgsModel`` and change it from
C3 to C4 vegetation, you can do the following:

.. code-block:: python

   # New parameters definition
   ags_model_kwargs = cm.ags.model_kwargs
   ags_model_kwargs['c3c4'] = 'c4'

   # Define a new land model
   land_model = abcmodel.land.AgsModel(**ags_model_kwargs)

   # ... then redefine the coupler, create a new state, and integrate.

Performance Notes
-----------------
It is also possible to JIT the model using ``equinox.filter_jit(integrate)``.
While this does not currently benefit a single analysis run, it becomes
important for accelerating training loops during parameter calibration.

Detailed API Reference
----------------------

For a detailed description of each model and its functions, see the API section.

.. toctree::
   :hidden:

   ABC Model <source/api/abcmodel>
   ABC Configs <source/api/abcconfigs>
