.. _Callbacks:

Callbacks
=========


Callback API
------------

.. autoclass:: learnrl.callbacks.Callback
   :members:


Logger
------

.. autoclass:: learnrl.callbacks.Logger
   :members:


.. _Metric_code:

Metric codes
------------

To fully represent a metric and how to aggregate it, we use metric codes as such:

    <logs_key>~<display_name>.<aggregator_function>

Where logs_key is the metric key in logs

| Display_name is the optional name that will be displayed in console.
| If not specified, the logs_key will be displayed.

Finaly aggregator_function is one of { ``avg``, ``sum``, ``last`` }:

- ``avg`` computes the average of the metric while aggregating. (default)
- ``sum`` computes the sum of the metric while aggregating.
- ``last`` only shows the last value of the metric.

Examples
~~~~~~~~

- ``reward~rwd.sum`` will aggregate the sum of rewards and display Rwd
- ``loss`` will show the average loss as Loss (no surname)
- ``dt_step~`` will show the average step_time with no name (surname is '')
- ``exploration~exp.last`` will show the last exploration value as Exp
