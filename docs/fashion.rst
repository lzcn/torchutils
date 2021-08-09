torchutils.fashion
==================

.. currentmodule:: torchutils.fashion


Module :mod:`torchutils.fashion` defines utilities for fashion outfit recommendation task.

Definitions
-----------

Usually, an outfit consisits of :math:`n` items from different :math:`|\mathcal{C}|`
categories, where :math:`\mathcal{C}` is the set of fashion categories. Morever, in
personlized outfit recommendation task, each outfit is associated with an user id.


Outfit Tuples Format
~~~~~~~~~~~~~~~~~~~~

I found that, in current fashion dataset, each item has a unique key that consisits of
digital numbers. Thus, for fast data processing, I build a :mod:`numpy.ndarray` for
each outfit as followed:

.. code-block::

    [user_id, outfit_length, [item_id], [item_category]]

- `user_id`: user id for the outfit
- `outfit_length`: number of of items in the outfit
- `[item_id]`: a list of item id, append `-1` if the lenght of outfit less than `max_length`
- `[item_category]`: the corresponding item categories. `-1` is appened.


Module Reference
----------------

.. automodule:: torchutils.fashion
    :members:
