import collections
from typing import (Any, Callable, Deque, Dict, Generic, Iterator, Mapping,
                    MutableMapping, NamedTuple, Optional, Sequence, TypeVar,
                    Union)

import jax

"""
ALL TAKEN FROM https://github.com/google-deepmind/dm-haiku
extraced to remove dependency for merge and partition
"""


def simple_dtype(dtype) -> str:
    if isinstance(dtype, type):
        dtype = dtype(0).dtype
    dtype = dtype.name
    dtype = dtype.replace("complex", "c")
    dtype = dtype.replace("double", "d")
    dtype = dtype.replace("float", "f")
    dtype = dtype.replace("uint", "u")
    dtype = dtype.replace("int", "s")
    return dtype


def format_array(x: Any) -> str:
    """Formats the given array showing dtype and shape info."""
    return simple_dtype(x.dtype) + "[" + ",".join(map(str, x.shape)) + "]"


def _copy_structure(tree):
    """Returns a copy of the given structure."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def traverse(
    structure: Any,
) -> Any:
    """Iterates over a structure yielding module names, names and values.

    NOTE: Items are iterated in key sorted order.

    Args:
      structure: The structure to traverse.

    Yields:
      Tuples of the module name, name and value from the given structure.
    """
    for module_name in sorted(structure):
        bundle = structure[module_name]
        for name in sorted(bundle):
            value = bundle[name]
            yield module_name, name, value


def _to_dict_recurse(value: Any):
    if isinstance(value, Mapping):
        return {k: _to_dict_recurse(v) for k, v in value.items()}
    else:
        return _copy_structure(value)


def to_dict(mapping: Any):
    """Returns a ``dict`` copy of the given two level structure.

    This method is guaranteed to return a copy of the input structure (e.g. even
    if the input is already a ``dict``).

    Args:
      mapping: A two level mapping as returned by ``init`` functions of Haiku
          transforms.

    Returns:
      A new two level mapping with the same contents as the input.
    """
    return _to_dict_recurse(mapping)


def to_haiku_dict(structure: Any):
    """Returns a copy of the given two level structure.

    Uses the same mapping type as Haiku will return from ``init`` or ``apply``
    functions.

    Args:
      structure: A two level mapping to copy.

    Returns:
      A new two level mapping with the same contents as the input.
    """
    return to_dict(structure)


def merge(
    *structures: Any,
    check_duplicates: bool = False,
):
    """Merges multiple input structures.

    >>> weights = {'linear': {'w': None}}
    >>> biases = {'linear': {'b': None}}
    >>> hk.data_structures.merge(weights, biases)
    {'linear': {'w': None, 'b': None}}

    When structures are not disjoint the output will contain the value from the
    last structure for each path:

    >>> weights1 = {'linear': {'w': 1}}
    >>> weights2 = {'linear': {'w': 2}}
    >>> hk.data_structures.merge(weights1, weights2)
    {'linear': {'w': 2}}

    Note: returns a new structure not a view.

    Args:
      *structures: One or more structures to merge.
      check_duplicates: If True, a ValueError will be thrown if an array is
        found in multiple structures but with a different shape and dtype.

    Returns:
      A single structure with an entry for each path in the input structures.
    """
    def array_like(o): return hasattr(o, "shape") and hasattr(o, "dtype")
    def shaped(a): return (a.shape, a.dtype) if array_like(a) else None
    def fmt(a): return format_array(a) if array_like(a) else repr(a)

    out = collections.defaultdict(dict)
    for structure in structures:
        for module_name, name, value in traverse(structure):
            if check_duplicates and (name in out[module_name]):
                previous = out[module_name][name]
                if shaped(previous) != shaped(value):
                    raise ValueError(
                        "Duplicate array found with different shape/dtype for "
                        f"{module_name}.{name}: {fmt(previous)} vs {fmt(value)}.")
            out[module_name][name] = value
    return to_haiku_dict(out)


def partition_n(
    fn: Any,
    structure: Any,
    n: int,
):
    """Partitions a structure into `n` structures.

    For a given set of parameters, you can use :func:`partition_n` to split them
    into ``n`` groups. For example, to split your parameters/gradients by module
    name:

    >>> def partition_by_module(structure):
    ...   cnt = itertools.count()
    ...   d = collections.defaultdict(lambda: next(cnt))
    ...   fn = lambda m, n, v: d[m]
    ...   return hk.data_structures.partition_n(fn, structure, len(structure))

    >>> structure = {f'layer_{i}': {'w': None, 'b': None} for i in range(3)}
    >>> for substructure in partition_by_module(structure):
    ...   print(substructure)
    {'layer_0': {'b': None, 'w': None}}
    {'layer_1': {'b': None, 'w': None}}
    {'layer_2': {'b': None, 'w': None}}

    Args:
      fn: Callable returning which bucket in ``[0, n)`` the given element should
        be output.
      structure: Haiku params or state data structure to be partitioned.
      n: The total number of buckets.

    Returns:
      A tuple of size ``n``, where each element will contain the values for which
      the function returned the current index.
    """
    out = [collections.defaultdict(dict) for _ in range(n)]
    for module_name, name, value in traverse(structure):
        i = fn(module_name, name, value)
        assert 0 <= i < n, f"{i} must be in range [0, {n})"
        out[i][module_name][name] = value
    return tuple(to_haiku_dict(o) for o in out)


def partition(
    predicate,
    structure
):
    """Partitions the input structure in two according to a given predicate.

    For a given set of parameters, you can use :func:`partition` to split them:

    >>> params = {'linear': {'w': None, 'b': None}}
    >>> predicate = lambda module_name, name, value: name == 'w'
    >>> weights, biases = hk.data_structures.partition(predicate, params)
    >>> weights
    {'linear': {'w': None}}
    >>> biases
    {'linear': {'b': None}}

    Note: returns new structures not a view.

    Args:
      predicate: criterion to be used to partition the input data.
        The ``predicate`` argument is expected to be a boolean function taking as
        inputs the name of the module, the name of a given entry in the module
        data bundle (e.g. parameter name) and the corresponding data.
      structure: Haiku params or state data structure to be partitioned.

    Returns:
      A tuple containing all the params or state as partitioned by the input
        predicate. Entries matching the predicate will be in the first structure,
        and the rest will be in the second.
    """
    def f(m, n, v): return int(not predicate(m, n, v))
    return partition_n(f, structure, 2)
