torchcurves.maps
================

Input maps are callables that send raw inputs to the target interval chosen by
the curve module.

- ``LegendreCurve`` always maps to ``[-1, 1]``.
- ``BSplineBasis`` and ``BSplineCurve`` map to their effective knot interval.
- When ``BSplineBasis`` or ``BSplineCurve`` receives ``knots_config`` as an int, use
  ``parameter_range=(a, b)`` to choose that interval.

Built-in dotted presets
-----------------------

- ``real.rational``
- ``real.arctan``
- ``real.clamp``
- ``nonneg.rational``

Configured map objects
----------------------

Use the ``torchcurves.maps`` namespace when you want a configured map object:

.. code-block:: python

   import torchcurves as tc

   tc.LegendreCurve(4, 8, degree=5, input_map=tc.maps.Real.rational(scale=2.0))
   tc.BSplineCurve(4, 8, knots_config=10, input_map=tc.maps.Real.clamp(scale=0.5))
   tc.BSplineBasis(
       degree=3,
       knots_config=10,
       parameter_range=(0.0, 1.0),
       input_map=tc.maps.Nonneg.rational(),
   )

Custom input maps
-----------------

Custom callables should have signature ``f(x, out_min, out_max)``.

.. code-block:: python

   import torch

   def erf_map(scale: float = 1.0):
       def input_map(x, out_min=-1.0, out_max=1.0):
           mapped = torch.special.erf(x / scale)
           return ((mapped + 1) * (out_max - out_min)) / 2 + out_min

       return input_map
