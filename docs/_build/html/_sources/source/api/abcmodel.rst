
ABC Model
=========

The model's components are supposed to be documented in a structured way.
In each model page, we can see two main classes: ``InitConds`` and ``Model``.

``InitConds`` is a data class containing all the variables that make part of the ``state`` of the model
(and thereby of the ``CoupledState``) and will be updated by the model during ``run`` (diagnostics)
or ``integrate`` (prognostics, if any).

The ``Model`` class contains parameters, the ``run`` method and sometimes an ``integrate`` method.
Inside ``run``, variables ``x``, ``y``, etc, are updated using methods ``compute_x``, ``compute_y``, etc; and these
methods are documented in the order that they are called. The goal is that a reader can essentially read the
equations of each variable computation as they are done by our models and learn how things work from that.

Somemtimes, the ``state`` is updated inside a more complicated method like ``update_something``. This is sometimes
used for the modularity of our models (models inheriting other models). In that case, the models follow the same order
of updates of the parent model, but all methods of the child model overwrite the original ones.

Components
-----------

.. toctree::
   :maxdepth: 1

   abcmodel.rad
   abcmodel.land
   abcmodel.atmos

Functionalities
---------------

.. toctree::
   :maxdepth: 1

   abcmodel.abstracts
   abcmodel.coupling
   abcmodel.integration
   abcmodel.utils
