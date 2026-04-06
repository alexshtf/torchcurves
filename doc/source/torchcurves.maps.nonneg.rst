torchcurves.maps.nonneg
=======================

Nonnegative-domain input maps are intended for features such as time, revenue,
counts, and bids.

Available presets
-----------------

- ``nonneg.rational``
- ``nonneg.arctan``

Configured map objects
----------------------

.. code-block:: python

   import torchcurves as tc

   tc.maps.Nonneg.rational(scale=2.0)
   tc.maps.Nonneg.arctan(scale=2.0)

Typical use
-----------

.. code-block:: python

   import torchcurves as tc

   tc.BSplineBasis(
       degree=3,
       knots_config=10,
       parameter_range=(0.0, 1.0),
       input_map="nonneg.rational",
   )
   tc.BSplineBasis(
       degree=3,
       knots_config=10,
       parameter_range=(0.0, 1.0),
       input_map=tc.maps.Nonneg.arctan(scale=2.0),
   )
