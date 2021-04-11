import csv
import itertools
import time
import traceback
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

MiniBatch = tp.Any
LRSchedule = tp.Callable[[int], float]
TrainStep = tp.Callable[['TrainState', MiniBatch], 'TrainState']
Variables = tp.Dict


class Metrics(flax.struct.PyTreeNode):
    state: tp.OrderedDict[str, tp.Tuple[float, float]]

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
        for k in self.state:
            yield k, self[k]

    def values(self):
        for _, v in self.items():
            yield v

    def reset(self):
        new_inst = self.from_names(*self.state.keys())
        if self._is_replicated():
            new_inst = flax.jax_utils.replicate(new_inst)
        return new_inst

    def _is_replicated(self, item=None):
        if not item:
            item = next(iter(self.state.values()), (0.0, 0.0))
        return hasattr(item[0], 'shape') and item[0].ndim > 0 and item[0].shape[0] > 1


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
    opt_state: tp.Dict
    metrics: Metrics

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
        train_step = jax.pmap(get_train_step(optimizer), axis_name='batch')
        state = flax.jax_utils.replicate(state)

    avg_loss, best_loss = 0., 0.
    losses: tp.List[float] = []
    lrs: tp.List[float] = []
    for batch_num, batch in enumerate(train_iter, start=1):
        state = train_step(state, batch)

        loss = state.metrics['loss']
        state = state.replace(metrics=state.metrics.reset())
        lr = sched(batch_num)
        if distributed:
            lr = flax.jax_utils.unreplicate(lr)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return jnp.stack(lrs), jnp.stack(losses)

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
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
        apply_fn: tp.Callable[[Variables, MiniBatch], tp.Tuple[tp.Any, tp.Dict]],
        loss_fn: tp.Callable[[tp.Any, MiniBatch], jnp.ndarray],
        opt_update: optax.TransformUpdateFn,
        update_metrics: tp.Callable[..., Metrics],
        distributed: bool = False) -> tp.Callable[[TrainState, MiniBatch], TrainState]:

    def train_step(state, batch):

        def forward(trainable_params):
            variables = {
                'params': merge_nested_dicts(trainable_params, state.frozen_params),
                **state.model_state
            }
            output, model_state = apply_fn(variables, batch)
            loss = loss_fn(output, batch)
            return loss, (output, model_state)

        grad_fn = jax.value_and_grad(forward, has_aux=True)
        (loss, (output, model_state)), grads = grad_fn(state.trainable_params)

        if distributed:
            grads = jax.lax.pmean(grads, axis_name='batch')
        updates, opt_state = opt_update(grads, state.opt_state, state.trainable_params)
        new_params = optax.apply_updates(state.trainable_params, updates)

        metrics = update_metrics(state=state,
                                 batch=batch,
                                 loss=loss,
                                 output=output,
                                 grads=grads)

        return state.replace(trainable_params=new_params,
                             model_state=model_state,
                             opt_state=opt_state,
                             metrics=metrics)

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


class _DummyCSVFile:

    def writerow(self, *_):
        pass

    def close(self):
        pass


# TODO: Fix formatting.
# yapf: disable
def train(cfg,
          state,
          train_iter,
          val_iter,
          train_step,
          val_step,
          filefmt='%Y-%m-%d_%H-%M-%S',
          printfmt='%Y-%m-%d %H:%M:%S',
          distributed=False,
          write_csv=False,
          save_ckpts=True,
          ckpt_metric='loss'):

    assert cfg.eval_freq % cfg.report_freq == 0

    headers = list(itertools.chain(
        ['Timestamp', 'Iter'], state.metrics.names(), ['Time/Step'],
        map('Val {}'.format, state.metrics.names()), ['Val Time', 'Ckpt?']))
    table_fn = partial(tabulate.tabulate, headers=headers,
                       tablefmt="github", floatfmt="3.4f")
    print('\n'.join(table_fn([
        [datetime.utcnow().strftime(printfmt), cfg.num_steps] +
        [100.0] * (len(headers) - 3) + [None]]).split('\n')[:2]))

    train_iter = iter(tqdm(itertools.islice(train_iter, 0, cfg.num_steps),
                           total=cfg.num_steps, desc='Training', smoothing=0))

    try:
        timestamp = datetime.utcnow().strftime(filefmt)
        csvfile = (open(f'{timestamp}_trainlog.csv', 'w', newline='')
                   if write_csv else _DummyCSVFile())
        traincsv = (csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                    if write_csv else csvfile)
        traincsv.writerow(headers)

        cur_best = -1
        start_time = time.perf_counter()
        for i, batch in enumerate(train_iter):
            state = train_step(state, batch)

            if i % cfg.report_freq == 0:
                time_per_step = (time.perf_counter() - start_time) / cfg.report_freq
                report_values = ([datetime.utcnow().strftime(printfmt), i] +
                                 list(state.metrics.values()) + [time_per_step])
                state = state.replace(metrics=state.metrics.reset())
                start_time = time.perf_counter()

                if i % cfg.eval_freq == 0:
                    start_time = time.perf_counter()
                    val_state = (state if not distributed else
                                 flax.jax_utils.unreplicate(state))
                    variables = val_state.variables
                    val_metrics = state.metrics.reset()
                    for val_batch in val_iter:
                        val_metrics = val_step(val_batch, variables, val_metrics)
                    elapsed = time.perf_counter() - start_time
                    report_values.extend(val_metrics.values() + [elapsed])

                    ckpt_metric_val = val_metrics[ckpt_metric]
                    if save_ckpts and cur_best < ckpt_metric_val:
                        report_values.append('Yes')
                        cur_best = ckpt_metric_val
                        flax.training.checkpoints.save_checkpoint(
                            f'ckpts_{cfg.name}',
                            jax.device_get(jax.tree_leaves(val_state)),
                            i,
                            keep=5)
                    else:
                        report_values.append('No')
                else:
                    report_values.extend(None for _ in range(len(val_metrics) + 2))

                print(table_fn([report_values]).split('\n')[2])
                traincsv.writerow(report_values)
                start_time = time.perf_counter()

    except Exception:
        print(traceback.format_exc())
    finally:
        csvfile.close()

    return state


# yapf: enable