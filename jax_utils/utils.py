import math
import typing as T
from functools import partial, wraps

import chex
import flax
import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

# From https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
INIT_GAIN: T.Dict[str, T.Callable[..., float]] = {
    'linear': lambda *_: 1.0,
    'conv': lambda *_: 1.0,
    'sigmoid': lambda *_: 1.0,
    'tanh': lambda *_: 5.0 / 3.0,
    'relu': lambda *_: math.sqrt(2.0),
    'leaky_relu': lambda ns: math.sqrt(2.0 / (1.0 + ns**2)),  # ns = negative_slope
    'selu': lambda *_: 3.0 / 4.0,
}


class Metrics(flax.struct.PyTreeNode):
    state: T.Mapping[str, T.Tuple[chex.Numeric, chex.Numeric]]

    @classmethod
    def from_names(cls, *names):
        return cls({n: (0.0, 0.0) for n in names})

    def update(self, **names_and_values):
        new_state = self.state.copy()
        new_state.update(
            # allowed to update only a subset of the metrics
            (n, (self.state[n][0] + v, self.state[n][1] + 1.0))
            for n, v in names_and_values.items())
        return self.replace(state=new_state)

    def __getitem__(self, name):
        v, c = self.state[name]
        return jnp.sum(v) / jnp.sum(c)  # sum in case array is replicated

    def __len__(self):
        return len(self.state)

    def names(self):
        return self.state.keys()

    def items(self):
        yield from zip(self.state.keys(), self.values())

    def values(self):
        return _jit_values(list(self.state.values()))

    def reset(self):
        new_inst = self.from_names(*self.state.keys())
        if self._is_replicated():
            new_inst = flax.jax_utils.replicate(new_inst)
        return new_inst

    def _is_replicated(self, item=None):
        if not item:
            item = next(iter(self.state.values()), (0.0, 0.0))
        return hasattr(item[0], 'shape') and item[0].ndim > 0 and item[0].shape[0] > 1


@jax.jit
def _jit_values(values):
    return [jnp.sum(v) / jnp.sum(c) for v, c in values]


class PRNGSeq:

    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)
        self._n = 0

    def __next__(self):
        self._n += 1
        return jax.random.fold_in(self._key, self._n)

    next = __next__

    def __repr__(self):
        return f'PRNGSequence(times_called={self._n})'


def partition_nested_dict(d, flat_left_keys):
    left, right = {}, {}
    flat_left_keys = set(flat_left_keys)
    for k, v in flatten_dict(d).items():
        if k in flat_left_keys:
            left[k] = v
        else:
            right[k] = v
    return tuple(map(unflatten_dict, (left, right)))


def merge_nested_dicts(*ds):
    merged = {}
    for d in map(flatten_dict, map(flax.core.unfreeze, ds)):
        if any(k in merged.keys() for k in d.keys()):
            raise ValueError('Key conflict!')
        merged.update(d)
    return unflatten_dict(merged)


def prep_data(ds, distributed=False):
    ldc = jax.local_device_count()

    def _prepare(x):
        x = x.numpy()
        return x.reshape((ldc, -1) + x.shape[1:]) if distributed else x

    it = map(partial(jax.tree_map, _prepare), ds)
    return flax.jax_utils.prefetch_to_device(it, 2) if distributed else it


def cos_onecycle_momentum(num_steps: int,
                          base_momentum: float = 0.85,
                          max_momentum: float = 0.95,
                          pct_start: float = 0.3) -> optax.Schedule:
    """ Based on fastai and PyTorch

    - fastai1.fast.ai/callbacks.one_cycle.html
    - pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    """
    return optax.piecewise_interpolate_schedule(
        'cosine', max_momentum, {
            int(pct_start * num_steps): base_momentum / max_momentum,
            int(num_steps): max_momentum / base_momentum
        })


def assert_dtype(f):

    @wraps(f)
    def inner(x, *args, **kwargs):
        out = f(x, *args, **kwargs)
        chex.assert_equal(x.dtype, out.dtype)
        return out

    return inner


@assert_dtype
def lerp(a, b, pct):
    chex.assert_equal_shape([a, b])  # avoid unwanted broadcasting
    chex.assert_rank(pct, {0, jnp.ndim(a)})  # avoid implicit 1-dimensions
    return a + (b - a) * pct


def truncated_normal_init(lower, upper):

    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.truncated_normal(key, lower, upper, shape, dtype)

    return init_fn
