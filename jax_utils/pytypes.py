import typing as T

import chex
import jax.numpy as jnp

ArrayTree = T.Any  # chex.ArrayTree is not supported yet
MiniBatch = ArrayTree
Variables = ArrayTree
Params = ArrayTree
ModelState = ArrayTree
PRNGDict = T.Dict[str, chex.PRNGKey]
ForwardFn = T.Callable[[Variables, MiniBatch, T.Optional[PRNGDict]],
                       T.Tuple[jnp.ndarray, T.Tuple[chex.ArrayTree, ModelState]]]
