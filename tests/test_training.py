from itertools import islice

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jax_utils

prng = jax.random.PRNGKey(0)
batch = {'input': jax.random.uniform(prng, shape=(16, 5), minval=0.0, maxval=10.0)}
batch['target'] = jnp.einsum('ni,i->n', batch['input'], jnp.arange(5))


def train_iter():
    while True:
        yield batch


model = nn.Dense(1)
variables = model.init(prng, batch['input'])

state = jax_utils.TrainState(trainable_params=variables['params'],
                             frozen_params={},
                             model_state={},
                             opt_state=optax.sgd(0.01).init(variables['params']),
                             metrics=jax_utils.Metrics.from_names('loss'))


def apply_fn(variables, batch):
    return model.apply(variables, batch['input']), {}


def loss_fn(output, batch):
    return optax.l2_loss(output, batch['target']).mean()


def update_metrics(state, batch, loss, output, grads):
    return state.metrics.update(loss=loss)


def eval_step(batch, variables, metrics):
    output = model.apply(variables, batch['input'])
    return metrics.update(loss=loss_fn(output, batch))


def get_find_lr_train_step(update_fn):
    return jax_utils.get_train_step(apply_fn=apply_fn,
                                    loss_fn=loss_fn,
                                    opt_update=update_fn,
                                    update_metrics=update_metrics)


def test_find_lr():
    _, losses = jax_utils.find_lr(get_train_step=get_find_lr_train_step,
                                  optim_factory=optax.sgd,
                                  state=state,
                                  train_iter=train_iter())

    assert np.argmin(losses) > 50


def test_train_step():
    train_step = jax.jit(
        jax_utils.get_train_step(apply_fn=apply_fn,
                                 loss_fn=loss_fn,
                                 opt_update=optax.sgd(0.01).update,
                                 update_metrics=update_metrics))

    state_ = train_step(state, batch)
    init_loss = state_.metrics['loss']
    for i, batch_ in enumerate(train_iter()):
        state_ = state_.replace(metrics=state_.metrics.reset())
        state_ = train_step(state_, batch)
        if i > 25:
            break

    final_loss = state_.metrics['loss']
    assert init_loss > 1000.0
    assert final_loss < 250.0


def test_train_loop():

    train_step = jax.jit(
        jax_utils.get_train_step(apply_fn=apply_fn,
                                 loss_fn=loss_fn,
                                 opt_update=optax.sgd(0.01).update,
                                 update_metrics=update_metrics))

    reporter = jax_utils.Reporter(train_names=list(state.metrics.names()) +
                                  ['time/step'],
                                  val_names=['loss', 'time'],
                                  print_names=['loss'],
                                  write_csv=False)
    _ = jax_utils.train(state=state,
                        train_iter=train_iter(),
                        val_iter=list(islice(train_iter(), 0, 10)),
                        train_step=train_step,
                        val_step=eval_step,
                        n_steps=500,
                        report_freq=20,
                        val_freq=100,
                        reporter=reporter,
                        val_metrics=state.metrics,
                        save_ckpts=False)
