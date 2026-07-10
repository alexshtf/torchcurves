torchcurves.functional
======================

.. currentmodule:: torchcurves.functional

Normalization
-------------
Low-level helpers for mapping inputs to bounded intervals. Most users should
prefer ``input_map`` and ``torchcurves.maps`` on the module API.

.. autosummary::
   :toctree: generated
   :nosignatures:

   clamp
   rational
   arctan


Parametrized curves
-------------------
Vectorized parametric curve evaluation functions.

.. autosummary::
   :toctree: generated
   :nosignatures:

   bspline_curves
   legendre_curves


Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   uniform_augmented_knots
