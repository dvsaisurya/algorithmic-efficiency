

import jax.numpy as jnp
import optax


import functools
from typing import Any, Callable, NamedTuple, Optional, Union
from jax.scipy.linalg import cho_factor, cho_solve


import chex
import jax
import math
from jax import lax
import numpy as np


# pylint:disable=no-value-for-parameter






ScalarOrSchedule = Union[float, optax.Schedule]
class TraceState(NamedTuple):
 """Holds an aggregation of past updates."""
 trace: optax.Params






MaskOrFn = Optional[Union[Any, Callable[[optax.Params], Any]]]




class ScaleByCasprState(NamedTuple):
 """State for the Adam algorithm."""
 count: chex.Array  # shape=(), dtype=jnp.int32.
 mu: optax.Updates
 nu: optax.Updates
 stats: optax.Updates
 preconds: optax.Updates
 lambdas: optax.Updates


class ShampooLRPair(NamedTuple):
 L: chex.Array
 R: chex.Array

class LambdaRLPair(NamedTuple):
 R: chex.Array
 L: chex.Array

def update_moment(updates, moments, decay, order):
 """Compute the exponential moving average of the `order`-th moment."""
 return jax.tree_util.tree_map(
     lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)




def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
 m = -1 if flip_sign else 1
 if callable(learning_rate):
   return optax.scale_by_schedule(lambda count: m * learning_rate(count))
 return optax.scale(m * learning_rate)




@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
 """Performs bias correction. It becomes a no-op as count goes to infinity."""
 # The conversion to the data type of the moment ensures that bfloat16 remains
 # bfloat16 in the optimizer state. This conversion has to be done after
 # `bias_correction_` is calculated as calculating `decay**count` in low
 # precision can result in it being rounded to 1 and subsequently a
 # "division by zero" error.
 bias_correction_ = 1 - decay**count


 # Perform division in the original precision.
 return jax.tree_util.tree_map(
     lambda t: t / bias_correction_.astype(t.dtype), moment)


def abs_sq(x: chex.Array) -> chex.Array:
 """Returns the squared norm of a (maybe complex) array.


 For real `x`, JAX generates the same HLO from this, `jnp.square(x)`, `x * x`,
 or `x**2`.


 Args:
   x: a (maybe complex) array.


 Returns:
   The squared norm of `x`.
 """
 if not isinstance(x, (np.ndarray, jnp.ndarray)):
   raise ValueError(f"`abs_sq` accepts only NDarrays, got: {x}.")
 return (x.conj() * x).real


def update_moment_per_elem_norm(updates, moments, decay, order):
 """Compute the EMA of the `order`-th moment of the element-wise norm."""


 def orderth_norm(g):
   if jnp.isrealobj(g):
     return g ** order
   else:
     half_order = order / 2
     # JAX generates different HLO for int and float `order`
     if half_order.is_integer():
       half_order = int(half_order)
     return abs_sq(g) ** half_order


 return jax.tree_util.tree_map(
     lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)




def get_merged_shape(param_shape):


   assert len(param_shape)<=4
   if len(param_shape)==4:
       new_shape = (param_shape[0]*param_shape[1]*param_shape[2], param_shape[3])
       return new_shape
   elif len(param_shape)==3:
       new_shape = (param_shape[0], param_shape[1]*param_shape[2])
       return new_shape
   else:
       return param_shape


def get_blocked_shape(merged_shape,block_size):
   assert len(merged_shape)==2
   d1, d2 = merged_shape
   return (math.ceil(d1/block_size),math.ceil(d2/block_size),block_size,block_size)


def update_stats(L,R,grad,block_size,b2):
 #TODO: update statistics once every few steps
 if len(L.shape)==1:
   return ShampooLRPair(L=L,R=R)
 #pad the gradient to be a multiple of block_size
 mgd_shape = get_merged_shape(grad.shape)
 grad = grad.reshape(mgd_shape)
 grad = jnp.pad(grad,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
 blkd_shape = get_blocked_shape(mgd_shape,block_size)
 g1,g2,_,_ = blkd_shape
 print(mgd_shape, blkd_shape)
 grad = grad.reshape(g1,block_size,g2,block_size)
 #transpose the grid dimensions to the front
 grad = grad.transpose((0,2,1,3))
 if b2==1.0:
   w1, w2 = 1.0, 1.0
 else:
   w1, w2 = b2, 1.0-b2
 L = w1*L + w2*jnp.einsum('ijkl,ijnl->ijkn',grad,grad)
 R = w1*R + w2*jnp.einsum('ijkl,ijkm->ijlm',grad,grad)
 return ShampooLRPair(L=L,R=R)



def update_lambdas(precond,lambd,prev_stat,grad,block_size,b2,count,exponent):
  if len(precond.L.shape)==1:
      return lambd
  mgd_shape = get_merged_shape(grad.shape)
  grad = grad.reshape(mgd_shape)
  grad = jnp.pad(grad,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
  blkd_shape = get_blocked_shape(mgd_shape,block_size)
  g1,g2,_,_ = blkd_shape
  print(mgd_shape, blkd_shape)
  grad = grad.reshape(g1,block_size,g2,block_size)
  #transpose the grid dimensions to the front
  grad = grad.transpose((0,2,1,3))
  print('changed code')
  #computing tr(Pr_tR_{t-1})/n
  tr = lambda x,y: jnp.sum(jnp.sum(x*y,axis=-1),axis=-1)
  Pr_t = precond.R@precond.R if exponent==2 else precond.R

  # Pr_t = precond.R
  lambdL_coeff = tr(Pr_t,prev_stat.R)[:,:,jnp.newaxis]/block_size

  #computing diag(G_tPr_tG_t^T)/n
  lambdL_res = jnp.einsum("ijkl,ijlm,ijmk->ijk",grad,Pr_t,grad.transpose(0,1,3,2))/block_size


#   lambdL = jax.lax.cond(count%4000==0,lambda : jnp.ones_like(lambd.L), lambda : jnp.minimum(0.9*lambd.L*lambdL_coeff +(1-0.9)* lambdL_res,1e30))
  # alpha = jnp.maximum((1-count/10000.0),0.0)
  alpha = 1.0
  get_b3 = lambda x:  -0.000012*x + 0.8
  b3 = get_b3(count)
  lambdL = alpha*jnp.clip(b3*lambd.L*lambdL_coeff +(1-b3)* lambdL_res,1e-24,1e24) + (1-alpha)*jnp.ones_like(lambd.L)
  # jax.debug.print('lambdL {x}', x = jnp.sum(lambdL,axis=-1)[:3,:3])
  #computing tr(PltL_{t-1})/m
  Pl_t = precond.L@precond.L if exponent==2 else precond.L

  # Pl_t = precond.L
  lambdR_coeff = tr(Pl_t, prev_stat.L)[:,:,jnp.newaxis]/block_size

  #computing diag(G_t^TPl_tG_t)/m
  lambdR_res = jnp.einsum("ijkl,ijlm,ijmk->ijk",grad.transpose(0,1,3,2),Pl_t,grad)/block_size
  lambdR =  alpha*jnp.clip(b3*lambd.R*lambdR_coeff + (1-b3)*lambdR_res,1e-24,1e24) + (1-alpha)*jnp.ones_like(lambd.R)
  # jax.debug.print('lambdR {x}', x = jnp.mean(lambdR))
#   lambdR = jax.lax.cond(count%4000==0, lambda: jnp.ones_like(lambd.R), lambda : jnp.minimum(0.9*lambd.R*lambdR_coeff + (1-0.9)*lambdR_res,1e30))
  
  return LambdaRLPair(R=lambdR, L=lambdL)


def eigh_inverse(stat,exponent=4,epsilon=1e-6,relative_epsilon=True):
 # _,max_ev = power_iteration(stat)
 # max_ev = jnp.linalg.norm(stat,ord=1)
 max_ev = jnp.trace(stat)/stat.shape[0]
 if relative_epsilon:
   epsilon = jnp.maximum(epsilon*max_ev,1e-16)
 reg_stat = stat+jnp.eye(stat.shape[0])*epsilon
 # reg_stat = stat
 eigvals,eigvecs = jnp.linalg.eigh(reg_stat)
 eigvals = jnp.maximum(eigvals,epsilon)
 inv_eigvals = jnp.power(eigvals,-1./exponent)
 mm = functools.partial(jnp.matmul,precision=lax.Precision.HIGHEST)
 error = jnp.max(jnp.abs(mm(eigvecs,(eigvals[:,jnp.newaxis]*eigvecs.T)) - stat))
 return mm(eigvecs,(inv_eigvals[:,jnp.newaxis]*eigvecs.T)),error




def cholesky_inverse(stat,exponent=4,epsilon=1e-6,relative_epsilon=True):
 #exponent is not going to be used.
 # _,max_ev = power_iteration(stat)
 # max_ev = jnp.linalg.norm(stat,ord=1)
 max_ev = jnp.trace(stat)/stat.shape[0]
 # jnp.linalg.norm(stat,ord=2)
 if relative_epsilon:
   epsilon = jnp.maximum(epsilon*max_ev,1e-16)
 reg_stat = stat+jnp.eye(stat.shape[0])*epsilon
 # reg_stat = stat
 c, lower = cho_factor(reg_stat)


 # Solve the linear system using the Cholesky factorization
 val = cho_solve((c, lower), jnp.eye(reg_stat.shape[0]))
 mm = functools.partial(jnp.matmul,precision=lax.Precision.HIGHEST)
 error = jnp.max(jnp.abs(mm(val,reg_stat) - jnp.eye(reg_stat.shape[0])))


 return val, error


def batch_stats(x, num_devices):
 """Batch `x` so that so that leading axis is num_devices."""
 n = len(x)
 b = int(n / num_devices)
 return x.reshape(num_devices,b,x.shape[-2],x.shape[-1])


def unbatch_stats(x):
 return x.reshape(-1,x.shape[-2],x.shape[-1])


def unbatch_errors(x):
 return x.reshape(-1)




def update_preconds(L,R,precond_L,precond_R,momentum,exponent,precondition,epsilon,
                   block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name=None):
 #TODO: account for padding after computing preconditioners.
 #for biases we don't do preconditioning
 if len(L.shape)==1:
   return ShampooLRPair(L=L,R=R)
 mgd_shape = get_merged_shape(momentum.shape)
 #pad the gradient to be a multiple of block size
 def precondition_false_fn(L,R,precond_L,precond_R,momentum):
   return ShampooLRPair(L=precond_L,R=precond_R)


 def precondition_true_fn(L,R,precond_L,precond_R,momentum):
   if inverse_type=='eigh':
     inverse_fn = eigh_inverse
   elif inverse_type=='cholesky':
     inverse_fn = cholesky_inverse
   elif inverse_type=='rsvd':
     raise NotImplemented
   stats = jnp.stack([L,R],axis=0)
   assert len(stats.shape)==5
   assert stats.shape[0]==2
   old_preconds = jnp.stack([precond_L,precond_R],axis=0)
   g1,g2 = stats.shape[1],stats.shape[2]
   stats_flat = stats.reshape(2*g1*g2,stats.shape[-2],stats.shape[-1])
   if batch_axis_name:
     num_devices = lax.psum(1, batch_axis_name)
   else:
     num_devices = 1
   num_statistics = 2*g1*g2


   #distribute inverse operations to multiple devices
   if batch_axis_name:
     # Pad statistics and exponents to next multiple of num_devices.
     to_pad = (-num_statistics) % num_devices
     pad_stats = jnp.zeros((to_pad,block_size,block_size))
     pad_stats = pad_stats.at[:].set(jnp.eye(block_size, dtype=stats.dtype))
     stats_flat = jnp.concatenate([stats_flat, pad_stats], axis=0)
     stats_flat_batched = batch_stats(stats_flat, num_devices)
     current_replica = lax.axis_index(batch_axis_name)
     _matrix_inverse_pth_root_vmap = jax.vmap(functools.partial(inverse_fn,exponent=exponent,epsilon=epsilon,relative_epsilon=relative_epsilon))
     preconds_flat_batched, errors_flat_batched = _matrix_inverse_pth_root_vmap(
         stats_flat_batched[current_replica]
     )
     preconds_flat_batched = jax.lax.all_gather(preconds_flat_batched, batch_axis_name)
     print('to_pad', to_pad, 'errors_flat_batched',errors_flat_batched.shape,'preconds_flat_batched',preconds_flat_batched.shape)
     errors_flat_batched = jax.lax.all_gather(errors_flat_batched, batch_axis_name)
     print('to_pad', to_pad, 'errors_flat_batched',errors_flat_batched.shape,'preconds_flat_batched',preconds_flat_batched.shape)
     if to_pad!=0:
       preconds_flat = unbatch_stats(preconds_flat_batched)[:-to_pad]
       errors_flat = unbatch_errors(errors_flat_batched)[:-to_pad]
     else:
       preconds_flat = unbatch_stats(preconds_flat_batched)
       errors_flat = unbatch_errors(errors_flat_batched)
   else:
     preconds_flat, errors_flat = jax.vmap(functools.partial(inverse_fn,exponent=exponent,epsilon=epsilon,relative_epsilon=relative_epsilon))(stats_flat)




   preconds = preconds_flat.reshape(2,g1,g2,stats.shape[-2],stats.shape[-1])
   errors = errors_flat.reshape(2,g1,g2)[:,:,:,jnp.newaxis,jnp.newaxis]
   errors = jnp.where(jnp.isnan(errors),jnp.ones_like(errors)*(error_tolerance+1.0),errors)
   # jax.debug.print('errors: {x}', x=errors)
   preconds = jnp.where(errors>error_tolerance,old_preconds,preconds)
   # print('preconds',preconds.shape)
   #account for paddings
   if mgd_shape[0]%block_size!=0:
     preconds = preconds.at[0,-1,:,mgd_shape[0]%block_size:,:].set(0)
     preconds = preconds.at[0,-1,:,:,mgd_shape[0]%block_size:].set(0)
   if mgd_shape[1]%block_size!=0:
     preconds = preconds.at[1,:,-1,mgd_shape[1]%block_size:,:].set(0)
     preconds = preconds.at[1,:,-1,:,mgd_shape[1]%block_size:].set(0)




   precond_L, precond_R= preconds[0],preconds[1]
   return ShampooLRPair(L=precond_L,R=precond_R)
 return jax.lax.cond(precondition,precondition_true_fn,precondition_false_fn,L,R,precond_L,precond_R,momentum)



def caspr_update_fn(precond,momentum,adam_update,lambd,block_size,caspr_p=2,global_grafting=False,exponent=1):
 #TODO: check whether the final momentum_reshaped retain the zeros.
 if len(adam_update.shape)==1:
   return adam_update
 orig_shape=momentum.shape
 mgd_shape = get_merged_shape(orig_shape)
 momentum_reshaped = momentum.reshape(mgd_shape)
 momentum_reshaped = jnp.pad(momentum_reshaped,
                       ((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),
                       mode='constant')
 blkd_shape = get_blocked_shape(mgd_shape,block_size)
 g1,g2,_,_ = blkd_shape
 momentum_reshaped = momentum_reshaped.reshape(g1,block_size,g2,block_size)
 #transpose the grid dimensions to the front
 momentum_reshaped = momentum_reshaped.transpose((0,2,1,3))
 if caspr_p==2:
   momentum_reshaped = jnp.einsum('ijkl,ijln->ijkn',precond.L,momentum_reshaped)+jnp.einsum('ijkl,ijnl->ijnk',precond.R,momentum_reshaped)
   momentum_reshaped = jnp.einsum('ijkl,ijln->ijkn',precond.L,momentum_reshaped)+jnp.einsum('ijkl,ijnl->ijnk',precond.R,momentum_reshaped)
 elif caspr_p==1:
   proc_lambdR = 1e-2*jnp.max(lambd.R,axis=-1)[:,:,jnp.newaxis]+lambd.R
   proc_lambdR = (proc_lambdR/(1e-30+jnp.sum(proc_lambdR,axis=-1)[:,:,jnp.newaxis]))[:,:,jnp.newaxis,:]
   m1 = jnp.einsum('ijkl,ijln->ijkn',precond.L,
    (momentum_reshaped*
     (1/(1e-30+proc_lambdR))**(1/exponent)))
  #  m1 = m1/(jnp.linalg.norm(m1.reshape(m1.shape[0],m1.shape[1],-1),axis=2)[:,:,jnp.newaxis,jnp.newaxis]+1e-30)
   proc_lambdL = 1e-2*jnp.max(lambd.L,axis=-1)[:,:,jnp.newaxis]+lambd.L
   proc_lambdL = (proc_lambdL/(1e-30+jnp.sum(proc_lambdL,axis=-1)[:,:,jnp.newaxis]))[:,:,:,jnp.newaxis]
   m2 = jnp.einsum('ijkl,ijnl->ijnk',precond.R,
    (1/(1e-30+proc_lambdL))**(1/exponent)*momentum_reshaped)
  #  if orig_shape[1]==1000 and orig_shape[0]==784:
  #   jax.debug.print("grad {x}",x = momentum_reshaped[10,10])
  #   jax.debug.print("lambdL {x}",x=lambd.L[10,10])
  #   jax.debug.print("lambdR {x}",x=lambd.R[10,10])
#    m1 = jnp.einsum('ijkl,ijln->ijkn',precond.L,momentum_reshaped)
#    m2 = jnp.einsum('ijkl,ijnl->ijnk',precond.R,momentum_reshaped)
   momentum_reshaped = m1+m2
 elif caspr_p==-1:
   momentum_reshaped = jnp.einsum('ijkl,ijln,ijnm->ijkm',precond.L,momentum_reshaped,precond.R)
 else:
   raise NotImplemented
 #unpad the momentum and reshape it back to the original shape
 momentum_reshaped = momentum_reshaped.transpose((0,2,1,3)).reshape(g1*block_size,g2*block_size)
 momentum_reshaped = momentum_reshaped[:mgd_shape[0],:mgd_shape[1]].reshape(orig_shape)
 # jax.debug.print('effect of padding: {x} ',
 #                 x=jnp.linalg.norm(momentum_reshaped[mgd_shape[0]:,:mgd_shape[1]]) if mgd_shape[0]<momentum_reshaped.shape[0] else 0.0)
 if not global_grafting:
   momentum_reshaped = momentum_reshaped/jnp.linalg.norm(momentum_reshaped.reshape(-1),ord=4) * jnp.linalg.norm(adam_update.reshape(-1),ord=4)


 return momentum_reshaped



def scale_by_caspr(
 b1: float = 0.9,
 b2: float = 0.999,
 eps: float = 1e-8,
 matrix_epsilon: float = 1e-6,
 eps_root: float = 0.0,
 block_size: int = 1024,
 preconditioning_compute_steps: int = 20,
 start_preconditioning_step: int = 101,
 exponent_override: int = 0,
 nesterov: bool = True,
 mu_dtype: Optional[chex.ArrayDType] = None,
 caspr_p: int = 2,
 relative_epsilon: bool = True,
 inverse_type: str = 'eigh',
 error_tolerance: float= 1e-2,
 verbose: bool= True,
 global_grafting: bool = False,
 batch_axis_name: Any = None
) -> optax.GradientTransformation:
 """Rescale updates according to the Adam algorithm.


 References:
   [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)


 WARNING: PyTorch and optax's adam follow Algorithm 1 of the Kingma
   and Ba's Adam paper, if reproducing old results note that TensorFlow
   used instead the formulation just before Section 2.1 of the paper.
   See https://github.com/deepmind/optax/issues/571 for more detail.


 Args:
   b1: Decay rate for the exponentially weighted average of grads.
   b2: Decay rate for the exponentially weighted average of squared grads.
   eps: Term added to the denominator to improve numerical stability.
   eps_root: Term added to the denominator inside the square-root to improve
     numerical stability when backpropagating gradients through the rescaling.
   mu_dtype: Optional `dtype` to be used for the first order accumulator; if
     `None` then the `dtype` is inferred from `params` and `updates`.


 Returns:
   A `GradientTransformation` object.
 """




 def init_fn(params):
   mu = jax.tree_util.tree_map(  # First moment
       lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
   nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment


   def stat_and_precond_init(param,state_type='stats'):
     mgd_shape = get_merged_shape(param.shape)
     if len(param.shape) > 1:
       blkd_shape = get_blocked_shape(mgd_shape,block_size)
       st = jnp.zeros(blkd_shape)
       coeff = matrix_epsilon if state_type=='stats' else 1.0
       st = st.at[:,:].set(coeff*jnp.eye(block_size))
       return ShampooLRPair(L=st,R=jnp.copy(st))
     else:
       blkd_shape = (1,)
       return ShampooLRPair(L=jnp.zeros(blkd_shape, dtype=param.dtype),R=jnp.zeros(blkd_shape, dtype=param.dtype))


   stats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='stats'), params)
   preconds = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='preconds'), params)

   def lambda_init(param):
     mgd_shape = get_merged_shape(param.shape)
     if len(param.shape) >1:
        blkd_shape = get_blocked_shape(mgd_shape,block_size)
        g1,g2,_,_ = blkd_shape
        lambdR = jnp.ones((g1,g2,block_size), dtype=param.dtype)
        lambdL = jnp.ones((g1,g2,block_size), dtype=param.dtype)
        return LambdaRLPair(R=lambdR,L=lambdL)
     else:
        blkd_shape = (1,)
        return LambdaRLPair(R=jnp.zeros(blkd_shape, dtype=param.dtype),L=jnp.zeros(blkd_shape, dtype=param.dtype))
   lambdas = jax.tree_util.tree_map(lambda_init,params)


   return ScaleByCasprState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, stats=stats, preconds=preconds, lambdas=lambdas)



 def update_fn(updates, state, params=None):
   #TODO: start preconditioning after start_preconditioning_step
   del params
   mu = update_moment(updates, state.mu, b1, 1)
   nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
   count_inc = state.count+1

   #updating statistics
   stats = jax.tree_util.tree_map(
     lambda s,u: update_stats(s.L,s.R,u,block_size,b2),
     state.stats,updates,is_leaf=lambda x: type(x).__name__=='ShampooLRPair')
   exponent = exponent_override if exponent_override !=0 else (4 if caspr_p==2 or caspr_p==-1 else 2)

   #updating preconditioners
   preconds = jax.tree_util.tree_map(lambda s,p,m: update_preconds(s.L,s.R,p.L,
                                                                 p.R,m,
                                                                 exponent,
                                                                 count_inc%preconditioning_compute_steps==0,
                                                                 matrix_epsilon,
                                                                 block_size,
                                                                 relative_epsilon,
                                                                 inverse_type,
                                                                 error_tolerance,batch_axis_name),
                                     stats,
                                     state.preconds,
                                     mu,is_leaf=lambda x: type(x).__name__=='ShampooLRPair')



   lambdas = jax.tree_util.tree_map(lambda p,l,s,u:update_lambdas(p,l,s,u,block_size,b2,count_inc,exponent),
                                     preconds,state.lambdas,state.stats,updates,
                                     is_leaf=lambda x: type(x).__name__=='LambdaRLPair' or
                                     type(x).__name__=='ShampooLRPair')
   def print_fn(m,stat_type='mu'):
     print_fn = lambda m: jax.debug.print(
       "step {st} " + stat_type + " l2 {x}, " + stat_type + " l0 1e-5 {y}, " + stat_type + " l0 1e-7 {z}, " + stat_type + " l0 1e-10 {u}",
       st=count_inc, x=jnp.linalg.norm(m.reshape(-1)), y=jnp.sum(jnp.abs(m) > 1e-5),
       z=jnp.sum(jnp.abs(m) > 1e-7), u=jnp.sum(jnp.abs(m) > 1e-10)
       )
   if verbose:
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='mu'), mu)
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='nu'), nu)
   nu_hat = bias_correction(nu, b2, count_inc)
   def nadam_fn(m,v,g):
     return  m / (jnp.sqrt(v + eps_root) + eps)
   def nesterov_mom(m,v,g):
    return (b1*m+(1-b1)*g) if nesterov else m
   mu_hat = jax.tree_util.tree_map(nesterov_mom,mu,nu_hat,updates)
   mu_hat = bias_correction(mu_hat, b1, count_inc)

   adam_updates = jax.tree_util.tree_map(
       lambda m, v, g: nadam_fn(m,v,g), mu_hat, nu_hat, updates)
   #used adam updates for rank 1 tensors, otherwise use caspr updates
   caspr_updates = jax.tree_util.tree_map(
      lambda p,m,u,l: caspr_update_fn(p,m,u,l,block_size,caspr_p,global_grafting,exponent),
      preconds,mu_hat,adam_updates,lambdas,
      is_leaf=lambda x: type(x).__name__=='ShampooLRPair' or type(x).__name__=='LambdaRLPair')
   if global_grafting:
     # Function to compute the norm of each array
     compute_norm = lambda arr: jnp.linalg.norm(arr.reshape(-1))
     # Apply the function to each leaf in the pytree
     norms = jax.tree_util.tree_map(compute_norm, adam_updates)
     # Sum the squares of the norms
     adam_global_norm = jnp.sqrt(jnp.sum(jnp.array([jnp.square(n) for n in jax.tree_util.tree_leaves(norms)])))
     norms = jax.tree_util.tree_map(compute_norm, caspr_updates)
     # Sum the squares of the norms
     caspr_global_norm = jnp.sqrt(jnp.sum(jnp.array([jnp.square(n) for n in jax.tree_util.tree_leaves(norms)])))
     ratio = adam_global_norm / (caspr_global_norm+1e-30)
     caspr_updates = jax.tree_util.tree_map(lambda x: ratio*x, caspr_updates)


   updates = jax.lax.cond(count_inc>start_preconditioning_step, lambda : caspr_updates, lambda : adam_updates)
   if verbose:
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='updates'), updates)
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='caspr_updates'), caspr_updates)
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='adam_updates'), adam_updates)
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='stats'), stats)
     jax.tree_util.tree_map(functools.partial(print_fn,stat_type='preconds'), preconds)
   return updates, ScaleByCasprState(count=count_inc, mu=mu, nu=nu, stats=stats, preconds=preconds, lambdas=lambdas)


 return optax.GradientTransformation(init_fn, update_fn)




def efficient_caspr_adaptive(
 learning_rate: ScalarOrSchedule,
 b1: float = 0.9,
 b2: float = 0.999,
 eps: float = 1e-8,
 matrix_epsilon: float = 1e-6,
 eps_root: float = 0.0,
 block_size: int = 1024,
 preconditioning_compute_steps: int = 20,
 start_preconditioning_step: int = 101,
 exponent_override: int = 0,
 nesterov: bool = True,
 mu_dtype: Optional[chex.ArrayDType] = None,
 caspr_p: int = 2,
 relative_epsilon: bool = True,
 inverse_type: str = 'eigh',
 error_tolerance: float= 1e-2,
 weight_decay: float = 1e-4,
 mask: Optional[Union[Callable[[optax.Params], Any], None]] = None,
 global_grafting: bool = False,
 batch_axis_name: Any = None
) -> optax.GradientTransformation:
 """Adam with weight decay regularization.


 AdamW uses weight decay to regularize learning towards small weights, as
 this leads to better generalization. In SGD you can also use L2 regularization
 to implement this as an additive loss term, however L2 regularization
 does not behave as intended for adaptive gradient algorithms such as Adam.


 References:
   Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101


 Args:
   learning_rate: A fixed global scaling factor.
   b1: Exponential decay rate to track the first moment of past gradients.
   b2: Exponential decay rate to track the second moment of past gradients.
   eps: A small constant applied to denominator outside of the square root
     (as in the Adam paper) to avoid dividing by zero when rescaling.
   eps_root: A small constant applied to denominator inside the square root (as
     in RMSProp), to avoid dividing by zero when rescaling. This is needed for
     instance when computing (meta-)gradients through Adam.
   mu_dtype: Optional `dtype` to be used for the first order accumulator; if
     `None` then the `dtype` is inferred from `params` and `updates`.
   weight_decay: Strength of the weight decay regularization. Note that this
     weight decay is multiplied with the learning rate. This is consistent
     with other frameworks such as PyTorch, but different from
     (Loshchilov et al, 2019) where the weight decay is only multiplied with
     the "schedule multiplier", but not the base learning rate.
   mask: A tree with same structure as (or a prefix of) the params PyTree,
     or a Callable that returns such a pytree given the params/updates.
     The leaves should be booleans, `True` for leaves/subtrees you want to
     apply the weight decay to, and `False` for those you want to skip. Note
     that the Adam gradient transformations are applied to all parameters.


 Returns:
   The corresponding `GradientTransformation`.
 """
 # Using jax.debug.print to print the parameters
 jax.debug.print("""
 learning_rate: {learning_rate},
 b1: {b1},
 b2: {b2},
 eps: {eps},
 matrix_epsilon: {matrix_epsilon},
 eps_root: {eps_root},
 block_size: {block_size},
 preconditioning_compute_steps: {preconditioning_compute_steps},
 start_preconditioning_step: {start_preconditioning_step},
 exponent_override: {exponent_override},
 nesterov: {nesterov},
 mu_dtype: {mu_dtype},
 caspr_p: {caspr_p},
 relative_epsilon: {relative_epsilon},
 error_tolerance: {error_tolerance},
 weight_decay: {weight_decay},
 mask: {mask},
 global_grafting: {global_grafting}
 """, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, matrix_epsilon=matrix_epsilon,
     eps_root=eps_root, block_size=block_size, preconditioning_compute_steps=preconditioning_compute_steps,
     start_preconditioning_step=start_preconditioning_step, exponent_override=exponent_override, nesterov=nesterov,
     mu_dtype=mu_dtype, caspr_p=caspr_p, relative_epsilon=relative_epsilon,
     error_tolerance=error_tolerance, weight_decay=weight_decay, mask=mask, global_grafting=global_grafting)
 return optax.chain(
     scale_by_caspr(
         b1, b2, eps, matrix_epsilon, eps_root, block_size,
         preconditioning_compute_steps, start_preconditioning_step,
         exponent_override, nesterov, mu_dtype,
         caspr_p, relative_epsilon, inverse_type, error_tolerance,global_grafting,batch_axis_name=batch_axis_name),
     optax.add_decayed_weights(weight_decay, mask),
     _scale_by_learning_rate(learning_rate),
 )




def efficient_cascaded_caspr(
 learning_rate: ScalarOrSchedule,
 b1: float = 0.9,
 b2: float = 0.999,
 eps: float = 1e-8,
 matrix_epsilon: float = 1e-6,
 eps_root: float = 0.0,
 block_size: int = 1024,
 preconditioning_compute_steps: int = 20,
 start_preconditioning_step: int = 101,
 exponent_override: int = 0,
 nesterov: bool = True,
 mu_dtype: Optional[chex.ArrayDType] = None,
 caspr_p: int = 2,
 relative_epsilon: bool = True,
 inverse_type: str = 'eigh',
 error_tolerance: float= 1e-2,
 weight_decay: float = 1e-4,
 mask: Optional[Union[Callable[[optax.Params], Any], None]] = None,
 global_grafting: bool = False,
 batch_axis_name: Any = None
) -> optax.GradientTransformation:
 """Adam with weight decay regularization.


 AdamW uses weight decay to regularize learning towards small weights, as
 this leads to better generalization. In SGD you can also use L2 regularization
 to implement this as an additive loss term, however L2 regularization
 does not behave as intended for adaptive gradient algorithms such as Adam.


 References:
   Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101


 Args:
   learning_rate: A fixed global scaling factor.
   b1: Exponential decay rate to track the first moment of past gradients.
   b2: Exponential decay rate to track the second moment of past gradients.
   eps: A small constant applied to denominator outside of the square root
     (as in the Adam paper) to avoid dividing by zero when rescaling.
   eps_root: A small constant applied to denominator inside the square root (as
     in RMSProp), to avoid dividing by zero when rescaling. This is needed for
     instance when computing (meta-)gradients through Adam.
   mu_dtype: Optional `dtype` to be used for the first order accumulator; if
     `None` then the `dtype` is inferred from `params` and `updates`.
   weight_decay: Strength of the weight decay regularization. Note that this
     weight decay is multiplied with the learning rate. This is consistent
     with other frameworks such as PyTorch, but different from
     (Loshchilov et al, 2019) where the weight decay is only multiplied with
     the "schedule multiplier", but not the base learning rate.
   mask: A tree with same structure as (or a prefix of) the params PyTree,
     or a Callable that returns such a pytree given the params/updates.
     The leaves should be booleans, `True` for leaves/subtrees you want to
     apply the weight decay to, and `False` for those you want to skip. Note
     that the Adam gradient transformations are applied to all parameters.


 Returns:
   The corresponding `GradientTransformation`.
 """
 # Using jax.debug.print to print the parameters
 jax.debug.print("""
 learning_rate: {learning_rate},
 b1: {b1},
 b2: {b2},
 eps: {eps},
 matrix_epsilon: {matrix_epsilon},
 eps_root: {eps_root},
 block_size: {block_size},
 preconditioning_compute_steps: {preconditioning_compute_steps},
 start_preconditioning_step: {start_preconditioning_step},
 exponent_override: {exponent_override},
 nesterov: {nesterov},
 mu_dtype: {mu_dtype},
 caspr_p: {caspr_p},
 relative_epsilon: {relative_epsilon},
 error_tolerance: {error_tolerance},
 weight_decay: {weight_decay},
 mask: {mask},
 global_grafting: {global_grafting},
 """, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, matrix_epsilon=matrix_epsilon,
     eps_root=eps_root, block_size=block_size, preconditioning_compute_steps=preconditioning_compute_steps,
     start_preconditioning_step=start_preconditioning_step, exponent_override=exponent_override, nesterov=nesterov,
     mu_dtype=mu_dtype, caspr_p=caspr_p, relative_epsilon=relative_epsilon,
     error_tolerance=error_tolerance, weight_decay=weight_decay, mask=mask, global_grafting=global_grafting)
 return optax.chain(
     scale_by_caspr(
         0.0, b2, eps, matrix_epsilon, eps_root, block_size,
         preconditioning_compute_steps, start_preconditioning_step,
         exponent_override, nesterov, mu_dtype,
         caspr_p, relative_epsilon, inverse_type, error_tolerance,global_grafting,
         batch_axis_name=batch_axis_name),
     optax.scale_by_adam(
         b1, b2, eps, eps_root),
     optax.add_decayed_weights(weight_decay, mask),
     _scale_by_learning_rate(learning_rate),
 )

