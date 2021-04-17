import csv
import itertools
import time
import typing as tp
from collections import OrderedDict
from datetime import datetime
from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
import tabulate
from flax.traverse_util import flatten_dict, unflatten_dict
from tqdm.auto import tqdm  # notebook compatible

try:
    from flax.training import checkpoints
except Exception:
    print('Could not import flax.training.checkpoints')

MiniBatch = tp.Any
LRSchedule = tp.Callable[[int], float]
TrainStep = tp.Callable[['TrainState', MiniBatch], 'TrainState']
Variables = tp.Dict


class Metrics(flax.struct.PyTreeNode):
    state: tp.Mapping[str, tp.Tuple[float, float]]

    @classmethod
    def from_names(cls, *names):
        return cls(OrderedDict((n, (0.0, 0.0)) for n in names))

    def update(self, **names_and_values):
        new_state = self.state.copy()
        new_state.update(
            # allowed to update only a subset of the metrics
            (n, (self.state[n][0] + v, self.state[n][1] + 1.0))
            for n, v in names_and_values.items())
        return self.replace(state=new_state)

    def __getitem__(self, name):
        v, c = self.state[name]
        # sum in case array is replicated
        return jnp.sum(v) / jnp.sum(c)

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


class TrainState(flax.struct.PyTreeNode):
    trainable_params: tp.Dict
    frozen_params: tp.Dict
    model_state: tp.Dict
    opt_state: optax.OptState
    metrics: Metrics
    rngs: tp.Optional[tp.Dict[str, jnp.ndarray]]

    @property
    def params(self):
        return merge_nested_dicts(self.trainable_params, self.frozen_params)

    @property
    def variables(self):
        return {'params': self.params, **self.model_state}


def find_lr(get_train_step: tp.Callable[[optax.TransformUpdateFn], TrainStep],
            optim_factory: tp.Callable[[LRSchedule], optax.GradientTransformation],
            state: TrainState,
            train_iter: tp.Iterator[MiniBatch],
            init_value: float = 1e-8,
            final_value: float = 10.,
            beta: float = 0.98,
            num_steps: int = 100,
            distributed: bool = False) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """Learning rate finding method by Smith 2018 (arxiv.org/pdf/1803.09820.pdf)

    Modified version of Sylvain Gugger's:
    sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    mult = (final_value / init_value)**(1 / num_steps)

    def sched(step):
        return init_value * mult**step

    optimizer = optim_factory(sched)
    state = state.replace(opt_state=optimizer.init(state.trainable_params))
    train_step = jax.jit(get_train_step(optimizer.update))

    if distributed:
        train_step = jax.pmap(get_train_step(optimizer.update), axis_name='batch')
        state = flax.jax_utils.replicate(state)

    avg_loss, best_loss = 0., float('inf')
    losses: tp.List[float] = []
    lrs: tp.List[float] = []
    for batch_num, batch in enumerate(train_iter, start=1):
        state = train_step(state, batch)

        loss = state.metrics['loss']
        state = state.replace(metrics=state.metrics.reset())
        lr = sched(batch_num)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return jnp.stack(lrs), jnp.stack(losses)

        # Record the best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        lrs.append(lr)

    return jnp.stack(lrs), jnp.stack(losses)


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


def get_train_step(
    forward: tp.Callable[[Variables, MiniBatch, tp.Optional['RngDict']], 
                         tp.Tuple[jnp.ndarray, tp.Tuple['Output', 'ModelState']]],
    opt_update: optax.TransformUpdateFn,
    update_metrics: tp.Callable[..., Metrics],
    distributed: bool = False) -> TrainStep:

    def train_step(state, batch):

        def forward_fn(trainable_params):
            variables = {
                'params': merge_nested_dicts(trainable_params, state.frozen_params),
                **state.model_state
            }
            return forward(variables, batch, state.rngs)

        grad_fn = jax.value_and_grad(forward_fn, has_aux=True)
        (loss, (output, model_state)), grads = grad_fn(state.trainable_params)

        if distributed:
            grads = jax.lax.pmean(grads, axis_name='batch')
        updates, opt_state = opt_update(grads, state.opt_state, state.trainable_params)
        new_params = optax.apply_updates(state.trainable_params, updates)

        metrics = update_metrics(state=state,
                                 batch=batch,
                                 loss=loss,
                                 output=output,
                                 grads=grads,
                                 updates=updates)

        rngs = jax.tree_map(partial(jax.random.fold_in, data=0), state.rngs)

        return state.replace(trainable_params=new_params,
                             model_state=model_state,
                             opt_state=opt_state,
                             metrics=metrics,
                             rngs=rngs)

    return train_step


def prep_data(ds, distributed=False):
    ldc = jax.local_device_count()

    def _prepare(x):
        x = x.numpy()
        return x.reshape((ldc, -1) + x.shape[1:]) if distributed else x

    it = map(partial(jax.tree_map, _prepare), ds)
    return flax.jax_utils.prefetch_to_device(it, 2) if distributed else it


def cos_onecycle_momentum(num_steps,
                          base_momentum=0.85,
                          max_momentum=0.95,
                          pct_start=0.3):
    """ Based on fastai and PyTorch

    - fastai1.fast.ai/callbacks.one_cycle.html
    - pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    """
    return optax.piecewise_interpolate_schedule(
        'cosine', max_momentum, {
            int(pct_start * num_steps): base_momentum / max_momentum,
            int(num_steps): max_momentum / base_momentum
        })


class _DummyWriter:

    def flush(self):
        pass

    def write(self, _):
        pass

    def close(self):
        pass

    def scalar(self, *_):
        pass


class Reporter:

    def __init__(self,
                 train_names,
                 val_names,
                 print_names=None,
                 filefmt='%Y-%m-%d_%H-%M-%S',
                 printfmt='%Y-%m-%d %H:%M:%S',
                 tabulate_fn=partial(tabulate.tabulate,
                                     tablefmt="github",
                                     floatfmt="9.4f"),
                 write_csv=False,
                 summary_writer=_DummyWriter()):

        def get_header(x):
            header = itertools.chain(['timestamp', 'iter'], list(x),
                                     (f'val_{n}' for n in val_names))
            return list(header)

        self.train_names = train_names
        self.val_names = val_names
        self.print_names = train_names if print_names is None else print_names
        self.header_csv = get_header(self.train_names)
        self.header_print = get_header(self.print_names)
        self.tabulate_csv = partial(tabulate_fn, headers=self.header_csv)
        self.tabulate_print = partial(tabulate_fn, headers=self.header_print)
        self.log_stamp = lambda: datetime.utcnow().strftime(printfmt)
        self.open_csv = lambda: (open(
            f'{datetime.utcnow().strftime(filefmt)}_trainlog.csv', 'w', newline='')
                                 if write_csv else _DummyWriter())
        self.tb = summary_writer

    def __enter__(self):
        self.csvfile = self.open_csv()
        self.traincsv = csv.writer(self.csvfile, quoting=csv.QUOTE_MINIMAL)
        self.traincsv.writerow(self.header_csv)
        print('\n'.join(
            self.tabulate_print([[self.log_stamp(), 100_000] + [1.0] *
                                 (len(self.header_print) - 2)]).split('\n')[:2]))
        return self

    def __exit__(self, *_):  # type, value, tb):
        self.csvfile.close()

    def report(self, iteration, train_dict, val_dict={}):
        prefix = [self.log_stamp(), iteration]
        suffix = [val_dict.get(k, None) for k in self.val_names]
        csv_vals = prefix + [train_dict.get(k, None) for k in self.train_names] + suffix
        print_vals = prefix + [train_dict[k] for k in self.print_names] + suffix

        print(self.tabulate_print([print_vals]).split('\n')[2])
        self.traincsv.writerow(csv_vals)

        for k, v in train_dict.items():
            if isinstance(v, (int, float, jnp.ndarray)):
                self.tb.scalar(k, v, iteration)

        for k, v in val_dict.items():
            if isinstance(v, (int, float, jnp.ndarray)):
                self.tb.scalar(f'val_{k}', v, iteration)


def train(state: TrainState,
          train_iter: tp.Iterator[MiniBatch],
          val_iter: tp.Iterator[MiniBatch],
          train_step: TrainStep,
          val_step: tp.Callable[[MiniBatch, tp.Any, Metrics], Metrics],
          n_steps: int,
          val_freq: int,
          report_freq: int,
          reporter: Reporter,
          val_metrics: Metrics,
          distributed: bool = False,
          save_ckpts: bool = True,
          ckpt_metric: str = 'loss',
          ckpt_name: str = 'model') -> TrainState:

    assert val_freq % report_freq == 0
    assert 'time' in reporter.val_names
    assert 'time/step' in reporter.train_names

    iter_slice = itertools.islice(train_iter, 0, n_steps)
    train_iter = iter(tqdm(iter_slice, total=n_steps, desc='Training', smoothing=0))

    if distributed:
        state = flax.jax_utils.replicate(state)

    with reporter as rep:
        cur_best = -1
        start_time = time.perf_counter()
        for i, batch in enumerate(train_iter):
            state = train_step(state, batch)

            if i % report_freq == 0 or i == n_steps - 1:
                train_dict = {
                    'time/step': (time.perf_counter() - start_time) / report_freq
                }
                train_dict.update(state.metrics.items())
                state = state.replace(metrics=state.metrics.reset())

                val_dict = {}
                if i % val_freq == 0 or i == n_steps - 1:
                    start_time = time.perf_counter()
                    val_state = (state if not distributed else
                                 flax.jax_utils.unreplicate(state))
                    variables = val_state.variables
                    val_metrics = val_metrics.reset()
                    for val_batch in val_iter:
                        val_metrics = val_step(val_batch, variables, val_metrics)
                    val_dict = dict(val_metrics.items())
                    val_dict['time'] = time.perf_counter() - start_time

                    ckpt_metric_val = val_dict[ckpt_metric]
                    if save_ckpts and cur_best < ckpt_metric_val:
                        # TODO: Add comparison option (i.e. less or more is better)
                        cur_best = ckpt_metric_val
                        # OrderedDict can't be serialized
                        flat_state = jax.device_get(jax.tree_leaves(val_state))
                        checkpoints.save_checkpoint(f'ckpts_{ckpt_name}',
                                                    flat_state,
                                                    i,
                                                    keep=5)
                        if 'ckpt' in reporter.val_names:
                            val_dict['ckpt'] = 'Saved.'

                rep.report(i, train_dict, val_dict)
                start_time = time.perf_counter()

    return state if not distributed else flax.jax_utils.unreplicate(state)
