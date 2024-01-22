"""
Utility modules for simple manipulations of pytrees.
"""

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def replicate_array(array, n_devices):
    return tree_map(
        lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape),
        array,
    )


def strip_array(array):
    return tree_map(lambda x: x[0], array)
