"""Submission file for an NAdamW optimizer with warmup+cosine LR in Jax."""

import functools

# isort: off
# We have to turn off isort here to resolve a conflict between isort and yapf.
from typing import (Any,
                    Callable,
                    Dict,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)
# isort: on

import chex
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from algorithmic_efficiency.efficient_caspr_adaptive_full_matrix_dist_inv import efficient_caspr_adaptive_full_matrix_dist_inv, TrainingMetrics

from flax import struct

# def _default_zero_field():
#   return struct.field(
#       default_factory=functools.partial(jnp.array, 0, jnp.float32))

# @struct.dataclass
# class TrainingMetrics:
#     root_errors: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     root_errors_lambdas: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     root_failure_perc: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     root_failure_perc_lambdas: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     coeff: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
#     res: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
#     lambd: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
#     stat: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
    

_GRAD_CLIP_EPS = 1e-6


# Forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/alias.py
def nadamw(
    learning_rate: Union[float, optax.Schedule],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the official PyTorch
  implementation also follows this).
  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1).

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Whether to use bias correction.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Nadam gradient transformations are applied to all parameters.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
      scale_by_nadam(b1, b2, eps, eps_root, debias),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      scale_by_learning_rate(learning_rate))


# All functions below are forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/transform.py
def scale_by_nadam(b1: float = 0.9,
                   b2: float = 0.999,
                   eps: float = 1e-8,
                   eps_root: float = 0.0,
                   debias: bool = True,
                   power: float = 0.5) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this).

  Current code implements a simpler version with no momentum decay and slightly
  different (standard Adam) bias correction terms. The exact description can be
  found here https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Whether to use bias correction.
    power: The power to use in the preconditioner (0.5 in default adam).
  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the NAdam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  def jax_cosine_warmup(step_hint: int, hyperparameters):
    # Create learning rate schedule.
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters.learning_rate,
        transition_steps=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])
    return schedule_fn

  # Create optimizer + LR schedule.
  lr_schedule_fn = jax_cosine_warmup(workload.step_hint *0.53571619898, hyperparameters)
  opt_init_fn, opt_update_fn = efficient_caspr_adaptive_full_matrix_dist_inv(
       lr_schedule_fn,
       b1=1.0 - hyperparameters.one_minus_beta1,
       b2=hyperparameters.beta2,
       b3=hyperparameters.beta2,
       eps=1e-8,
       lamb_eps=1e-6,
       matrix_epsilon=1e-6,
       eps_root=0.0,
       block_size=384,
       preconditioning_compute_steps=20,
       start_preconditioning_step=101,
       exponent_override=0,
       nesterov=True,
       caspr_p=1,
       relative_epsilon=True,
       inverse_type="coupled newton",
       error_tolerance=1e-1,
       weight_decay=hyperparameters.weight_decay,
       global_grafting=False,
       batch_axis_name='batch',
       precond_type="left",
       log_metrics=True
     )
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip,
                       label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def agg_metrics(new_optimizer_state):

    #code for the target setting, replace Testing with the ScaleByCasprState
    # print(new_optimizer_state[1])
    metrics = new_optimizer_state[0].metrics
    # print("metrics ",metrics)
    metrics_flat,tree_def = jax.tree_util.tree_flatten(metrics,is_leaf = lambda x: isinstance(x,TrainingMetrics))
    # print("metrics_flat " ,metrics_flat)
    from functools import reduce


    
    # Define a function that will be used to add two corresponding leaves
    def add_trees(tree1, tree2):
        # This function will be applied to each leaf
        def add_leaves(leaf1, leaf2):
            if type(leaf1)==optax.MaskedNode and type(leaf2)==optax.MaskedNode:
                return jnp.array(0.0)
            if type(leaf1)==optax.MaskedNode:
                return leaf2
            if type(leaf2)==optax.MaskedNode:
                return leaf1
            return leaf1 + leaf2
        
        # Use jax.tree_map to apply add_leaves to each leaf in tree1 and tree2
        # Note that jax.tree_map can only map functions across the leaves of a single pytree,
        # so we need to use a lambda function to pass additional arguments (like leaf2 from tree2)
        return jax.tree_util.tree_map(lambda leaf1, leaf2: add_leaves(leaf1, leaf2), tree1, tree2, is_leaf=lambda x: type(x) in [optax.MaskedNode,chex.Array])

    # Assuming you have a list of 100 pytrees
    # pytrees = [pytree1, pytree2, ..., pytree100]

    # Use functools.reduce to cumulatively apply the add_trees operation across all pytrees
    aggregated_metrics = reduce(add_trees, metrics_flat)
    # Use functools.reduce to cumulatively apply the summing operation across all pytrees
    # aggregated_metrics = reduce(lambda acc, pytree: jax.tree_util.tree_multimap(sum_leaves, acc, pytree), metrics_flat[1:], metrics_flat[0])

    metrics_count = jax.tree_util.tree_map(lambda x: 1.0 if isinstance(x,chex.Array) else 0.0, metrics_flat, is_leaf=lambda x: type(x) in [optax.MaskedNode,chex.Array])

    metrics_count_agg = reduce(add_trees, metrics_count)

    # print(metrics_count)
    avg_metrics = jax.tree_util.tree_map(lambda x,y: x/y if y!=0.0 else jnp.array(0.0), aggregated_metrics,metrics_count_agg)
    # print('avg_metrics', avg_metrics)
    return avg_metrics



def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None
  
  outputs = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               optimizer_state,
                               current_param_container,
                               batch,
                               per_device_rngs,
                               grad_clip,
                               label_smoothing)
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs
  #compute optimizer metrics:
  # print(new_optimizer_state)

# @struct.dataclass
# class TrainingMetrics:
#     root_errors: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     root_errors_lambdas: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     root_failure_perc: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     root_failure_perc_lambdas: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
#     coeff: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
#     res: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
#     lambd: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
#     stat: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
  

  # print(avg_metrics)
  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    log_metrics=True
    if log_metrics:
      avg_metrics = agg_metrics(new_optimizer_state)
    if log_metrics:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss[0],
              'root_errors': avg_metrics.root_errors[0],
              'root_errors_lambdas': avg_metrics.root_errors_lambdas[0],
              'root_failure_perc': avg_metrics.root_failure_perc[0],
              'root_failure_perc_lambdas': avg_metrics.root_failure_perc_lambdas[0],
              'coeff': avg_metrics.coeff[0],
              'residual': avg_metrics.res[0],
              'lambda_trace':avg_metrics.lambd[0],
              'stat_trace': avg_metrics.stat[0],
              'grad_norm': grad_norm[0]
          }, global_step)
    else:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss[0],
              'grad_norm': grad_norm[0]
          }, global_step)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
