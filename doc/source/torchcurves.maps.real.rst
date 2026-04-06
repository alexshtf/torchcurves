torchcurves.maps.real
=====================

Real-domain input maps send arbitrary real-valued inputs to the target
interval chosen by the curve module.

Available presets
-----------------

- ``real.rational``
- ``real.arctan``
- ``real.clamp``

Configured map objects
----------------------

.. code-block:: python

   import torchcurves as tc

   tc.maps.Real.rational(scale=2.0)
   tc.maps.Real.arctan(scale=2.0)
   tc.maps.Real.clamp(scale=0.5)

Typical use
-----------

.. code-block:: python

   import torchcurves as tc

   tc.LegendreCurve(4, 8, degree=5, input_map="real.rational")
   tc.BSplineCurve(4, 8, knots_config=10, input_map=tc.maps.Real.clamp(scale=0.5))
