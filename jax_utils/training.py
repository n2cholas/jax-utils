import csv
import itertools
import time
import typing as T
from datetime import datetime
from functools import partial

import chex
import flax
import jax
import jax.numpy as jnp
import optax
import tabulate
from tqdm.auto import tqdm  # notebook compatible

from . import pytypes as PT
from . import utils

try:
    from flax.training import checkpoints
except Exception:
    print('Could not import flax.training.checkpoints')

TrainStep = T.Callable[['TrainState', PT.MiniBatch], 'TrainState']
ValStep = T.Callable[[PT.MiniBatch, PT.Variables, utils.Metrics], utils.Metrics]


class _DummyWriter:

    def flush(self):
        pass

    def write(self, _):
        pass

    def close(self):
        pass

    def scalar(self, *_):
        pass


class TrainState(flax.struct.PyTreeNode):
    trainable_params: PT.Params
    frozen_params: PT.Params
    model_state: PT.ModelState
    opt_state: optax.OptState
    metrics: utils.Metrics
    rngs: T.Optional[PT.PRNGDict]

    @property
    def params(self):
        return utils.merge_nested_dicts(self.trainable_params, self.frozen_params)

    @property
    def variables(self):
        return {'params': self.params, **self.model_state}


def get_train_step(forward: PT.ForwardFn,
                   opt_update: optax.TransformUpdateFn,
                   update_metrics: T.Callable[..., utils.Metrics],
                   distributed: bool = False) -> TrainStep:

    def train_step(state: TrainState, batch: PT.MiniBatch):

        def forward_fn(trainable_params):
            params = utils.merge_nested_dicts(trainable_params, state.frozen_params)
            variables = {'params': params, **state.model_state}
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


class Reporter:

    def __init__(self,
                 train_names: T.Sequence[str],
                 val_names: T.Sequence[str],
                 print_names: T.Optional[T.Sequence[str]] = None,
                 filefmt: str = '%Y-%m-%d_%H-%M-%S',
                 printfmt: str = '%Y-%m-%d %H:%M:%S',
                 tabulate_fn: T.Callable = partial(tabulate.tabulate,
                                                   tablefmt="github",
                                                   floatfmt="9.4f"),
                 write_csv: bool = False,
                 summary_writer: T.Optional[T.Any] = None):

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
        self.tb = summary_writer or _DummyWriter()

    def __enter__(self):
        self.csvfile = self.open_csv()
        self.traincsv = csv.writer(self.csvfile, quoting=csv.QUOTE_MINIMAL)
        self.traincsv.writerow(self.header_csv)
        tqdm.write('\n'.join(
            self.tabulate_print([[self.log_stamp(), 100_000] + [1.0] *
                                 (len(self.header_print) - 2)]).split('\n')[:2]))
        return self

    def __exit__(self, *_):  # type, value, tb):
        self.csvfile.close()

    @T.no_type_check
    def report(self,
               iteration: int,
               train_dict: T.Mapping[str, chex.Scalar],
               val_dict: T.Mapping[str, chex.Scalar] = {}):
        prefix = [self.log_stamp(), iteration]
        suffix = [val_dict.get(k, None) for k in self.val_names]
        csv_vals = prefix + [train_dict.get(k, None) for k in self.train_names] + suffix
        print_vals = prefix + [train_dict[k] for k in self.print_names] + suffix

        tqdm.write(self.tabulate_print([print_vals]).split('\n')[2])
        self.traincsv.writerow(csv_vals)

        for k, v in train_dict.items():
            if isinstance(v, (int, float, jnp.ndarray)):
                self.tb.scalar(k, v, iteration)

        for k, v in val_dict.items():
            if isinstance(v, (int, float, jnp.ndarray)):
                self.tb.scalar(f'val_{k}', v, iteration)


def train(
    state: TrainState,
    *,
    train_iter: T.Iterator[PT.MiniBatch],
    train_step: TrainStep,
    n_steps: int,
    report_freq: int,
    reporter: Reporter,
    val_iter: T.Optional[T.Iterator[PT.MiniBatch]] = None,
    val_step: T.Optional[ValStep] = None,
    val_freq: T.Optional[int] = None,
    val_metrics: T.Optional[utils.Metrics] = None,
    distributed: bool = False,
    save_ckpts: bool = True,
    ckpt_metric: str = 'loss',
    ckpt_name: str = 'model',
    extra_report_fn: T.Optional[T.Callable[[TrainState, PT.MiniBatch, int],
                                           None]] = None,
    start_step: int = 0,
) -> TrainState:

    assert 'time/step' in reporter.train_names
    if val_step is not None:
        assert val_iter is not None
        assert val_metrics is not None
        assert val_freq is not None and val_freq % report_freq == 0
        assert 'time' in reporter.val_names

    iter_slice = itertools.islice(train_iter, 0, n_steps - start_step)
    train_iter = iter(tqdm(iter_slice, total=n_steps - start_step, desc='Training'))

    if distributed:
        state = flax.jax_utils.replicate(state)
        if hasattr(state, 'rngs'):
            pfold_in = partial(jax.pmap(jax.random.fold_in),
                               data=jnp.arange(jax.device_count()))
            state = state.replace(rngs=jax.tree_map(pfold_in, state.rngs))

    with reporter as rep:
        cur_best = -1
        start_time = time.perf_counter()
        for i, batch in enumerate(train_iter, start=start_step):
            state = train_step(state, batch)

            if i % report_freq == 0 or i == n_steps - 1:
                time_per_step = (time.perf_counter() - start_time) / report_freq
                train_dict = {'time/step': time_per_step}
                train_dict.update(state.metrics.items())
                state = state.replace(metrics=state.metrics.reset())

                val_dict = {}
                if val_freq is not None and (i % val_freq == 0 or i == n_steps - 1):
                    assert val_step is not None
                    assert val_metrics is not None
                    assert val_iter is not None

                    start_time = time.perf_counter()
                    val_state = (state if not distributed else
                                 flax.jax_utils.unreplicate(state))
                    variables = val_state.variables
                    val_metrics = T.cast(utils.Metrics, val_metrics.reset())
                    for val_batch in val_iter:
                        val_metrics = val_step(val_batch, variables, val_metrics)
                    val_dict = dict(val_metrics.items())
                    val_dict['time'] = time.perf_counter() - start_time

                    ckpt_metric_val = val_dict[ckpt_metric]
                    if save_ckpts and cur_best < ckpt_metric_val:
                        # TODO: Add comparison option (i.e. less or more is better)
                        cur_best = ckpt_metric_val
                        checkpoints.save_checkpoint(f'ckpts_{ckpt_name}',
                                                    jax.device_get(val_state),
                                                    i,
                                                    keep=5)
                        if 'ckpt' in reporter.val_names:
                            val_dict['ckpt'] = 'Saved.'

                rep.report(i, train_dict, val_dict)
                if extra_report_fn is not None:
                    if distributed:
                        rstate, rbatch = flax.jax_utils.unreplicate((state, batch))
                    else:
                        rstate, rbatch = state, batch
                    extra_report_fn(rstate, rbatch, i)
                start_time = time.perf_counter()

    return state if not distributed else flax.jax_utils.unreplicate(state)


def find_lr(get_train_step: T.Callable[[optax.TransformUpdateFn], TrainStep],
            optim_factory: T.Callable[[optax.Schedule], optax.GradientTransformation],
            state: TrainState,
            train_iter: T.Iterator[PT.MiniBatch],
            init_value: float = 1e-8,
            final_value: float = 10.,
            beta: float = 0.98,
            num_steps: int = 100,
            distributed: bool = False) -> T.Tuple[T.List[float], T.List[float]]:
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
    losses: T.List[float] = []
    lrs: T.List[float] = []
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
