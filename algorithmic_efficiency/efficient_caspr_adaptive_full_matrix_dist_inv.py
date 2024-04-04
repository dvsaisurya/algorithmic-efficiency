

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

from reference_algorithms.paper_baselines.shampoo.jax.distributed_shampoo import matrix_inverse_pth_root, power_iteration, mat_power

from flax import struct

def _default_zero_field():
  return struct.field(
      default_factory=functools.partial(jnp.array, 0, jnp.float32))

@struct.dataclass
class TrainingMetrics:
    root_errors: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
    root_errors_lambdas: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
    root_failure_perc: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
    root_failure_perc_lambdas: Union[chex.Array, optax.MaskedNode] = _default_zero_field()
    coeff: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
    res: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
    lambd: Union[chex.Array,optax.MaskedNode] = _default_zero_field()
    stat: Union[chex.Array,optax.MaskedNode] = _default_zero_field()


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
    preconds_lambdas: optax.Updates
    oldstats: optax.Updates
    metrics: optax.Updates


# @struct.dataclass
# class TrainingMetrics:
#     """Diagnostic metrics from training."""
#     # Error for inverse-pth roots.
#     inverse_pth_root_errors: chex.Array = _default_zero_field()
#     # Iteration count for inverse-pth roots.
#     inverse_pth_root_iters: chex.Array = _default_zero_field()
#     # If final iteration error increases sufficiently, iteration terminates early.
#     # This field records the ratio of the final iteration error.
#     final_error_ratio: chex.Array = _default_zero_field()
#     # Max eigen value from either the power iteration or from LOBPCG.
#     max_eigen_value: chex.Array = _default_zero_field()
#     # Total retries of inverse pth root iterative method.
#     total_retries: chex.Array = _default_zero_field()

#     lobpcg_diagnostics: LOBPCGDiagnostics = struct.field(
#         default_factory=LOBPCGDiagnostics)
#     # Rich matrix entrywise error diagnostics, if enabled.
#     inverse_pth_root_diagnostics: InversePthRootDiagnostics = struct.field(
#         default_factory=InversePthRootDiagnostics)
#     # Diagnostics applied to the conditioned p-th root problem, after top
#     # eigenvectors are removed, if LOBPCG is being applied.
#     conditioned_inverse_pth_root_diagnostics: InversePthRootDiagnostics = (
#         struct.field(default_factory=InversePthRootDiagnostics))
#     fd: Union[FDDiagnostics,
#             optax.MaskedNode] = struct.field(default_factory=optax.MaskedNode)


class ShampooLRPair(NamedTuple):
    L: chex.Array
    R: chex.Array

class LambdaRLPair(NamedTuple):
    R: chex.Array
    L: chex.Array


class UpdateLambdas(NamedTuple):
    lambd: LambdaRLPair
    lambd_max_ev: ShampooLRPair
    coeff: ShampooLRPair
    res: ShampooLRPair

class UpdateStats(NamedTuple):
    stat: ShampooLRPair
    stat_max_ev: ShampooLRPair

def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    w1,w2 = (1-decay) if decay!=1.0 else 1.0, decay
    return jax.tree_util.tree_map(
      lambda g, t: w1 * (g ** order) + w2 * t, updates, moments)




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
    w1,w2 = (1-decay) if decay!=1.0 else 1.0, decay
    return jax.tree_util.tree_map(
        lambda g, t: w1 * orderth_norm(g) + w2 * t, updates, moments)

def get_merged_shape(param_shape):
    assert len(param_shape)<=4
    if len(param_shape)==4:
        new_shape = (param_shape[0]*param_shape[1]*param_shape[2], param_shape[3])
        return new_shape
    elif len(param_shape)==3:
        new_shape = (param_shape[0], param_shape[1]*param_shape[2]) if param_shape[1]<param_shape[0] else (param_shape[0]*param_shape[1],param_shape[2])
        return new_shape
    else:
        return param_shape



def get_blocked_shape(merged_shape,block_size):
    assert len(merged_shape)==2
    d1, d2 = merged_shape
    return (math.ceil(d1/block_size),math.ceil(d2/block_size),block_size,block_size)


def update_stats(L,R,grad,block_size,b2,precond_type="all",log_metrics=False):
    #TODO: update statistics once every few steps
    #  L, R = s.L, s.R
    #pad the gradient to be a multiple of block_size
    # if type(L).__name__!='MaskedNode':
    #     jax.debug.print("L:{L}:",L=jnp.sum(jnp.sum(L,axis=-1)!=0))
    # if type(R).__name__!='MaskedNode':
    #     jax.debug.print("R:{R}:",R=jnp.sum(jnp.sum(R,axis=-1)!=0))
    mgd_shape = get_merged_shape(grad.shape)
    if len(mgd_shape)<=1 or sum([ dim>5000 for dim in grad.shape]):
        return ShampooLRPair(L,R)
    print(mgd_shape)
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

    if precond_type=="all":
        L = w1*L + w2*jnp.einsum('ijkl,ijnl->ijkn',grad,grad)
        R = w1*R + w2*jnp.einsum('ijkl,ijkm->ijlm',grad,grad)
        return ShampooLRPair(L=L,R=R)
    elif precond_type=="left":
        L = w1*L + w2*jnp.einsum('ijkl,ijnl->ijkn',grad,grad)
        R = optax.MaskedNode()
    elif precond_type=="right":
        L = optax.MaskedNode()
        R = w1*R + w2*jnp.einsum('ijkl,ijkm->ijlm',grad,grad)
    M = np.zeros((g1,g2))
    N = np.zeros((g1,g2))
    M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
    M[:-1,:] = block_size
    N[:,-1] = mgd_shape[1]%block_size if mgd_shape[1]%block_size!=0 else block_size
    N[:,:-1] = block_size
    max_ev_R = lambda x: ((jnp.einsum("ijkk->ij",x))/N)[:,:,jnp.newaxis,jnp.newaxis]
    max_ev_L = lambda x: ((jnp.einsum("ijkk->ij",x))/M)[:,:,jnp.newaxis,jnp.newaxis]

    return UpdateStats(stat=ShampooLRPair(L,R),
                       stat_max_ev=ShampooLRPair(L=(jnp.mean(max_ev_L(L)) if log_metrics else jnp.array(1.0)) if not isinstance(L,optax.MaskedNode) else optax.MaskedNode(),
                                                 R=(jnp.mean(max_ev_R(R)) if log_metrics else jnp.array(1.0)) if not isinstance(R,optax.MaskedNode) else optax.MaskedNode()))


def update_lambdas(precond,lambd,prev_stat,grad,block_size,b3,count,exponent,precondition,matrix_epsilon,precond_type="all",log_metrics=False):
    # print('prev lambda', lambd.L.shape)
    if len(grad.shape)<=1 or sum([ dim>5000 for dim in grad.shape]):
        return UpdateLambdas(lambd=lambd,
                         lambd_max_ev=ShampooLRPair(L=optax.MaskedNode(),R=optax.MaskedNode()),
                         coeff=ShampooLRPair(L=optax.MaskedNode(),R=optax.MaskedNode()),
                         res=ShampooLRPair(L=optax.MaskedNode(),R=optax.MaskedNode()))
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


    M = np.zeros((g1,g2))
    N = np.zeros((g1,g2))
    M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
    M[:-1,:] = block_size
    N[:,-1] = mgd_shape[1]%block_size if mgd_shape[1]%block_size!=0 else block_size
    N[:,:-1] = block_size
    mm = lambda x,y: jnp.einsum('ijkl,ijlm->ijkm',x,y)

    max_ev_R = lambda x: ((jnp.einsum("ijkk->ij",x))/N)[:,:,jnp.newaxis,jnp.newaxis]

    In = jnp.zeros((g1,g2,block_size,block_size))
    In = In.at[:,:].set(jnp.eye(block_size))
    max_ev_L = lambda x: ((jnp.einsum("ijkk->ij",x))/M)[:,:,jnp.newaxis,jnp.newaxis]
    Im = jnp.zeros((g1,g2,block_size,block_size))
    Im = Im.at[:,:].set(jnp.eye(block_size))

    lambdL = optax.MaskedNode()
    lambdL_coeff_mean = optax.MaskedNode()
    lambdL_res_max_ev = optax.MaskedNode()
    lambdL_max_ev = optax.MaskedNode()
    if precond_type in ["all","right"]:
        prev_R = prev_stat.R
        Pr_t = precond.R
        if exponent==2:
            lambdL_coeff = jax.lax.cond(precondition,
                    lambda: jnp.clip(tr(Pr_t,mm(prev_R,Pr_t)),0,None)[:,:,jnp.newaxis,jnp.newaxis]/N[:,:,np.newaxis,np.newaxis],
                    lambda: jnp.ones((g1,g2,1,1)))

        else:
            lambdL_coeff = jax.lax.cond(precondition,
                    lambda: jnp.clip(tr(Pr_t,prev_R)[:,:,jnp.newaxis,jnp.newaxis],0,None)/N[:,:,np.newaxis,np.newaxis],
                    lambda: jnp.ones((g1,g2,1,1)))

        #computing diag(G_tPr_tG_t^T)/n
        if exponent!=2:
            lambdL_res = jnp.einsum("ijkl,ijlm,ijnm->ijkn",grad,Pr_t,grad)/N[:,:,np.newaxis,np.newaxis]
        else:
            right_grad = jnp.einsum("ijkl,ijlm->ijkm",grad,Pr_t)
            lambdL_res = jnp.einsum("ijkm,ijnm->ijkn",right_grad,right_grad)/N[:,:,np.newaxis,np.newaxis]


        lambdL_res_max_ev = jnp.mean(max_ev_L(lambdL_res)) if log_metrics else jnp.array(1.0)
        lambdL_coeff_mean = jnp.mean(lambdL_coeff) if log_metrics else jnp.array(1.0)

        lambdL = b3*lambd.L*lambdL_coeff +(1-b3)* lambdL_res
        lambdL_bool = (jnp.logical_or(lambdL>1e36,lambdL<-1e36)).sum(axis=-1).sum(axis=-1)
        lambdL = jnp.where(lambdL_bool[:,:,jnp.newaxis,jnp.newaxis], Im,lambdL)
        lambdL_max_ev = jnp.mean(max_ev_L(lambdL)) if log_metrics else jnp.array(1.0)
        print('after lambda L',lambdL.shape)

    lambdR = optax.MaskedNode()
    lambdR_coeff_mean = optax.MaskedNode()
    lambdR_res_max_ev = optax.MaskedNode()
    lambdR_max_ev = optax.MaskedNode()
    if precond_type in ["all","left"]:

        prev_L = prev_stat.L
        Pl_t = precond.L
        if exponent==2:
            lambdR_coeff = jax.lax.cond(precondition,
                                        lambda: jnp.clip(tr(Pl_t, prev_L@Pl_t)[:,:,jnp.newaxis,jnp.newaxis],0,None)/M[:,:,np.newaxis,np.newaxis],
                                        lambda: jnp.ones((g1,g2,1,1))    )
        else:
            lambdR_coeff = jax.lax.cond(precondition,
                                        lambda: jnp.clip(tr(Pl_t, prev_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/M[:,:,np.newaxis,np.newaxis],
                                        lambda: jnp.ones((g1,g2,1,1)))

    #   jax.debug.print('lambdR_coeff {x}',x=jnp.sum(lambdR_coeff!=1.0))
        #computing diag(G_t^TPl_tG_t)/m
        if exponent==2:
            left_grad = jnp.einsum("ijlk,ijlm->ijkm",grad,Pl_t)
            lambdR_res = jnp.einsum("ijkm,ijnm->ijkn",left_grad,left_grad)/M[:,:,np.newaxis,np.newaxis]
        else:
            lambdR_res = jnp.einsum("ijlk,ijlm,ijmn->ijkn",grad,Pl_t,grad)/M[:,:,np.newaxis,np.newaxis]
        lambdR_res_max_ev = jnp.mean(max_ev_R(lambdR_res)) if log_metrics else jnp.array(1.0)
        lambdR =  b3*lambd.R*lambdR_coeff + (1-b3)*lambdR_res
        lambdR_bool = (jnp.logical_or(lambdR>1e36,lambdR<-1e36)).sum(axis=-1).sum(axis=-1)
        lambdR = jnp.where(lambdR_bool[:,:,jnp.newaxis,jnp.newaxis],In,lambdR)
        lambdR_max_ev = jnp.mean(max_ev_R(lambdR)) if log_metrics else jnp.array(1.0)
        lambdR_coeff_mean = jnp.mean(lambdR_coeff) if log_metrics else jnp.array(1.0)

        # jax.debug.print("lambdR:{lambdR}:",lambdR=jnp.sum(jnp.sum(lambdR,axis=-1)!=0))
        # jax.debug.print("lambdR_coeff_mean {x}",x=lambdR_coeff_mean)
        print('after lambda R',lambdR.shape)
    return UpdateLambdas(lambd=LambdaRLPair(R=lambdR, L=lambdL),
                         lambd_max_ev=ShampooLRPair(L=lambdL_max_ev,R=lambdR_max_ev),
                         coeff=ShampooLRPair(L=lambdL_coeff_mean,R=lambdR_coeff_mean),
                         res=ShampooLRPair(L=lambdL_res_max_ev,R=lambdR_res_max_ev))

def eigh_inverse(stat,padding,exponent=4,epsilon=1e-6,relative_epsilon=True):
    # _,max_ev = power_iteration(stat)
    # max_ev = jnp.linalg.norm(stat,ord=1)
    # max_ev = jnp.trace(stat)/stat.shape[0]
    ix = (jnp.arange(stat.shape[0])<padding)
    stat = (stat*ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    identity = jnp.array((np.eye(stat.shape[0])*ix[:,np.newaxis])*ix[np.newaxis,:],dtype=stat.dtype)
    scale = (1e-30+jnp.trace(stat)/padding)
    stat = stat/scale

    _,max_ev = power_iteration(stat)


    if relative_epsilon:
        epsilon = jnp.maximum(epsilon*max_ev,1e-20)
    reg_stat = stat+identity*epsilon
    eigvals,eigvecs = jnp.linalg.eigh(reg_stat)

    mm = functools.partial(jnp.matmul,precision=lax.Precision.HIGHEST)
    # eigvals = jnp.einsum("ij,ij->j",mm(reg_stat,eigvecs),eigvecs,precision=lax.Precision.HIGHEST)
    # jax.debug.print("ratio positive eigvals: {x}",x=jnp.sum(eigvals>0)/eigvals.shape[0])
    # eigvals = jnp.flip(ix) * eigvals
    inv_eigvals = (jnp.maximum(eigvals, epsilon)**(-1./exponent))
    # inv_eigvals = inv_eigvals*jnp.flip(ix)
    # eigvals = jnp.maximum(eigvals,epsilon)
    #bottom most eigvals with s
    
    inv_pth_reg_stat = mm(mm(eigvecs,jnp.diag(inv_eigvals)),eigvecs.T)
    # inv_pth_reg_stat = mm(eigvecs,jnp.diag(inv_eigvals
    # inv_pth_reg_stat = mm(eigvecs,(inv_eigvals[:,jnp.newaxis]*eigvecs.T))
    inv_pth_reg_stat = (inv_pth_reg_stat*ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    error = jnp.max(jnp.abs(mat_power(inv_pth_reg_stat,p=exponent)@reg_stat - identity))
    # error = 1e-7
    return inv_pth_reg_stat*(scale**(-1/exponent)), error


def coupled_newton_inverse(stat,padding,exponent=4,epsilon=1e-6,relative_epsilon=True):
    scale = (jnp.trace(stat)/stat.shape[0]+1e-30)
    stat = stat/scale
    inv_pth_root,metrics = matrix_inverse_pth_root(stat,exponent,ridge_epsilon=epsilon,error_tolerance=1e-6,
                            relative_matrix_epsilon=relative_epsilon,padding_start=padding)
    error = metrics.inverse_pth_root_errors
    return inv_pth_root*(scale**(-1/exponent)),error

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

def batch_paddings(x, num_devices):
    """Batch `x` so that so that leading axis is num_devices."""
    n = len(x)
    b = int(n / num_devices)
    return x.reshape(num_devices,b,1)



def unbatch_stats(x):
    return x.reshape(-1,x.shape[-2],x.shape[-1])


def unbatch_errors(x):
    return x.reshape(-1)




def get_inverses(stats,preconds,paddings,errors,exponent,precondition,epsilon,
                   block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name=None):
    def precondition_false_fn(stats,preconds,paddings,errors):
        errors = jnp.zeros(preconds.shape[0])
        return preconds, errors


    def precondition_true_fn(stats,preconds,paddings,errors):
        if inverse_type=='eigh':
            inverse_fn = eigh_inverse
        elif inverse_type=='cholesky':
            inverse_fn = cholesky_inverse
        elif inverse_type=='coupled newton':
            inverse_fn = coupled_newton_inverse
        elif inverse_type=='rsvd':
            raise NotImplemented
        #  stats = jnp.stack([L,R],axis=0)
        assert len(stats.shape)==3
        assert len(preconds.shape)==3

        old_preconds = preconds
        g = stats.shape[0]
        stats_flat = stats
        paddings_flat = paddings
        if batch_axis_name:
            num_devices = lax.psum(1, batch_axis_name)
        else:
            num_devices = 1
        num_statistics = g
        #distribute inverse operations to multiple devices
        if batch_axis_name:
            # Pad statistics and exponents to next multiple of num_devices.
            to_pad = (-num_statistics) % num_devices
            pad_stats = jnp.zeros((to_pad,block_size,block_size))
            pad_paddings = jnp.ones((to_pad,1))*block_size
            pad_stats = pad_stats.at[:].set(jnp.eye(block_size, dtype=stats.dtype))

            stats_flat = jnp.concatenate([stats_flat, pad_stats], axis=0)
            paddings_flat = jnp.concatenate([paddings_flat, pad_paddings], axis=0)
            print('to_pad', to_pad, 'stats_flat',stats_flat.shape,'paddings_flat',paddings_flat.shape)
            stats_flat_batched = batch_stats(stats_flat, num_devices)
            paddings_flat_batched = batch_paddings(paddings_flat, num_devices)
            current_replica = lax.axis_index(batch_axis_name)
            _matrix_inverse_pth_root_vmap = jax.vmap(functools.partial(inverse_fn,exponent=exponent,epsilon=epsilon,relative_epsilon=relative_epsilon))
            preconds_flat_batched, errors_flat_batched = _matrix_inverse_pth_root_vmap(
                stats_flat_batched[current_replica], paddings_flat_batched[current_replica]
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
        preconds = preconds_flat.reshape(g,stats.shape[-2],stats.shape[-1])
        errors = errors_flat[:,jnp.newaxis,jnp.newaxis]
        errors = jnp.where(jnp.isnan(errors),jnp.ones_like(errors)*(error_tolerance+100.0),errors)
        # jax.debug.print('errors: {x}', x=jnp.sum(errors>error_tolerance)/errors.size)
        preconds = jnp.where(errors>error_tolerance,old_preconds,preconds)
        # print('preconds',preconds.shape)
        errors = errors[:,0,0]


        return preconds,errors
    return jax.lax.cond(precondition,precondition_true_fn,precondition_false_fn,stats,preconds,paddings,errors)



def split_array(arr, sizes):
    # Ensure the sum of sizes equals the first dimension of the array
    assert arr.shape[0] == sum(sizes), "The sum of sizes must equal the first dimension of the array"

    # Compute the cumulative sum of sizes to get split indices
    # We use [:-1] to exclude the last element, as split indices should not include the total length
    split_indices = np.cumsum(np.array(sizes))[:-1]

    # Split the array at the computed indices
    split_arrays = jnp.split(arr, split_indices)

    return split_arrays

def update_preconds_model(stats,preconds,paddings,errors,mu,
                    exponent,
                    precondition,
                    matrix_epsilon,
                    block_size,
                    relative_epsilon,
                    inverse_type,
                    error_tolerance,
                    batch_axis_name,
                    precond_type):
    stats_flat,tree_def = jax.tree_util.tree_flatten(stats)
    paddings_flat,_ = jax.tree_util.tree_flatten(paddings)
    # jax.debug.print("update_preconds_model  : {L}", L=paddings_flat)
    preconds_flat,tree_def = jax.tree_util.tree_flatten(preconds)
    errors_flat,tree_def2 = jax.tree_util.tree_flatten(errors)
    #   assert not optax.MaskedNode() in stats_flat
    orig_shapes = []
    new_preconds_flat = []
    new_stats_flat = []
    new_paddings_flat = []
    for precond_flat,stat_flat,padding_flat in zip(preconds_flat,stats_flat,paddings_flat):
        orig_shapes.append(precond_flat.shape)
        new_preconds_flat.append(precond_flat.reshape(-1,block_size,block_size))
        new_stats_flat.append(stat_flat.reshape(-1,block_size,block_size))
        new_paddings_flat.append(padding_flat.reshape(-1,1))
    preconds_flat = new_preconds_flat
    stats_flat = new_stats_flat
    paddings_flat = new_paddings_flat
    print([precond.shape for precond in preconds_flat])
    preconds_flat = jnp.concatenate(preconds_flat,axis=0)
    stats_flat = jnp.concatenate(stats_flat,axis=0)
    paddings_flat = jnp.concatenate(paddings_flat,axis=0)
    errors_flat = jnp.array(errors_flat)
    preconds_flat,errors_flat = get_inverses(stats_flat,preconds_flat,paddings_flat,errors_flat,exponent,precondition,matrix_epsilon,
                    block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name)
    #unwrapping preconds_flat
    split_sizes = ([ orig_shape[0]*orig_shape[1] for orig_shape in orig_shapes ])
    print("split_sizes",split_sizes)
    print(preconds_flat.shape,np.sum(split_sizes))
    preconds_flat = split_array(preconds_flat,split_sizes)
    errors_flat = split_array(errors_flat,split_sizes)
    # new_preconds = []
    # g = len(preconds_flat)
    # for idx in range(0,g,2):
    #   precondL,precondR = preconds_flat[idx:idx+2]
    #   orig_shapeL,orig_shapeR = orig_shapes[idx:idx+2]
    #   new_preconds.append(ShampooLRPair(L=precondL.reshape(*orig_shapeL),R=precondR.reshape(*orig_shapeR)))
    preconds_flat = [ precond.reshape(orig_shape) for precond,orig_shape in zip(preconds_flat,orig_shapes)]
    new_preconds = jax.tree_util.tree_unflatten(tree_def,preconds_flat)
    new_errors = jax.tree_util.tree_unflatten(tree_def,errors_flat)
    new_preconds = jax.lax.cond(precondition,lambda :
                jax.tree_util.tree_map(functools.partial(zeroing_preconds,precond_type=precond_type,block_size=block_size),
                                        new_preconds,mu,
                                        is_leaf=lambda x: type(x).__name__ in ['ShampooLRPair','LambdaRLPair']),
                lambda : new_preconds)
    print(new_preconds)
    return new_preconds,new_errors

def zeroing_preconds(precond,momentum,block_size=1024,precond_type="all"):

    precond_L = precond.L
    precond_R = precond.R
    mgd_shape = get_merged_shape(momentum.shape)
    if len(mgd_shape)<=1 or sum([ dim>5000 for dim in momentum.shape]):
        return precond
    if type(precond).__name__=='ShampooLRPair':
        nonzero_precond_L = (precond_type in ["all","left"])
        nonzero_precond_R = (precond_type in ["all","right"])
    else:
        nonzero_precond_L = (precond_type in ["all","right"])
        nonzero_precond_R = (precond_type in ["all","left"])


    if mgd_shape[0]%block_size!=0 and nonzero_precond_L:
        precond_L = precond_L.at[-1,:,mgd_shape[0]%block_size:,:].set(0)
        precond_L = precond_L.at[-1,:,:,mgd_shape[0]%block_size:].set(0)
    if mgd_shape[1]%block_size!=0 and nonzero_precond_R:
        precond_R = precond_R.at[:,-1,mgd_shape[1]%block_size:,:].set(0)
        precond_R = precond_R.at[:,-1,:,mgd_shape[1]%block_size:].set(0)

    return ShampooLRPair(L=precond_L,R=precond_R) if type(precond).__name__=='ShampooLRPair' else LambdaRLPair(R=precond_R,L=precond_L)


def caspr_update_fn(precond,momentum,adam_update,precond_lambd,block_size,caspr_p=2,global_grafting=False,exponent=1,precond_type="all"):
    #TODO: check whether the final momentum_reshaped retain the zeros.
    if len(adam_update.shape)==1 or sum([ dim>5000 for dim in adam_update.shape]):
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
        M = np.zeros((g1,g2))
        N = np.zeros((g1,g2))
        M[-1,:] = mgd_shape[0]%block_size if mgd_shape[0]%block_size!=0 else block_size
        M[:-1,:] = block_size
        N[:,-1] = mgd_shape[1]%block_size if mgd_shape[1]%block_size!=0 else block_size
        N[:,:-1] = block_size


        pL = precond.L
        pR = precond.R

        m1,m2 = None, None
        if precond_type in ["all","left"]:
            trfuncR = lambda x: (jnp.einsum('ijkk->ij',x)/N)[:,:,jnp.newaxis,jnp.newaxis]**(1/2)
            m1 = jnp.einsum('ijkl,ijln,ijnm->ijkm',pL,
                                            momentum_reshaped,precond_lambd.R/(1e-30+trfuncR(precond_lambd.R)))


        if precond_type in ["all","right"]:
            trfuncL = lambda x: (jnp.einsum('ijkk->ij',x)/M)[:,:,jnp.newaxis,jnp.newaxis]**(1/2)
            m2 = jnp.einsum('ijnl,ijlk,ijkm->ijnm',
                                        precond_lambd.L/(1e-30+trfuncL(precond_lambd.L)),momentum_reshaped,pR)


        # momentum_reshaped = m1+m2
        if precond_type == "left":
            momentum_reshaped = m1
        elif precond_type=="right":
            momentum_reshaped = m2
        else:
            momentum_reshaped = m1 + m2
    elif caspr_p==-1:
        momentum_reshaped = jnp.einsum('ijkl,ijln,ijnm->ijkm',precond.L,momentum_reshaped,precond.R)
    else:
        raise NotImplemented
    #unpad the momentum and reshape it back to the original shape
    momentum_reshaped = momentum_reshaped.transpose((0,2,1,3)).reshape(g1*block_size,g2*block_size)
    momentum_reshaped = momentum_reshaped[:mgd_shape[0],:mgd_shape[1]].reshape(orig_shape)
    # jax.debug.print('effect of padding: {x} ',
    #                 x=jnp.linalg.norm(momentum_reshaped[mgd_shape[0]:,:mgd_shape[1]]) if mgd_shape[0]<momentum_reshaped.shape[0] else 0.0)

    momentum_reshaped = momentum_reshaped/(1e-30+jnp.linalg.norm(momentum_reshaped.reshape(-1))) * jnp.linalg.norm(adam_update.reshape(-1))


    return momentum_reshaped



def get_paddings(s,G,block_size):
    L,R = s.L,s.R
    mgd_shape = get_merged_shape(G.shape)
    if len(mgd_shape)<=1 or sum([ dim>20000 for dim in G.shape]):
            return optax.MaskedNode()
    print(mgd_shape)
    blkd_shape = get_blocked_shape(mgd_shape,block_size)
    g1,g2,_,_ = blkd_shape
    print(mgd_shape, blkd_shape)
    padding_size_L = optax.MaskedNode()
    padding_size_R = optax.MaskedNode()
    if type(s.L).__name__!='MaskedNode':
        padding_size_L = np.ones((g1,g2),dtype=np.int32)*block_size
        if mgd_shape[0]%block_size!=0:
            padding_size_L[-1,:] = mgd_shape[0]%block_size
    if type(s.R).__name__!='MaskedNode':
        padding_size_R = np.ones((g1,g2),dtype=np.int32)*block_size
        if mgd_shape[1]%block_size!=0:
            padding_size_R[:,-1] = mgd_shape[1]%block_size

    #transpose the grid dimensions to the front
    return ShampooLRPair(L=padding_size_L,R=padding_size_R)



def scale_by_caspr(
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.8,
    eps: float = 1e-8,
    lamb_eps: float = 1e-2,
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
    batch_axis_name: Any = None,
    precond_type: str = "all",
    log_metrics: bool = False
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

    def get_padded_matrix(padding_start):
        ix = (jnp.arange(block_size)<padding_start)
        return (jnp.eye(block_size)*ix[jnp.newaxis,:])*ix[:,jnp.newaxis]

    get_padded_matrix_vmap = jax.vmap(get_padded_matrix,in_axes=0)


    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment




        def stat_and_precond_init(param,state_type='stats'):
            mgd_shape = get_merged_shape(param.shape)
            if len(param.shape) > 1 and not sum([dim>5000 for dim in param.shape]):
                blkd_shape = get_blocked_shape(mgd_shape,block_size)
                st = jnp.zeros(blkd_shape)
                paddings = get_paddings(ShampooLRPair(st,st), param, block_size)
                st_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
                st_R = get_padded_matrix_vmap(paddings.R.reshape(-1,1))
                st_L = st_L.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                st_R = st_R.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                coeff = matrix_epsilon if state_type=='stats' else 1.0
                st_L = st_L*coeff
                st_R = st_R*coeff
                # jax.debug.print("padding init fn L: {L} R: {R}", L=paddings.L,R=paddings.R)
                assert precond_type in ["left","right","all"]
                if precond_type=="left":
                    return ShampooLRPair(L=st_L,R=optax.MaskedNode())
                elif precond_type=="right":
                    return ShampooLRPair(L=optax.MaskedNode(),R=st_R)
                else:
                    return ShampooLRPair(L=st_L,R=st_R)
            else:
                return ShampooLRPair(L=optax.MaskedNode(),R=optax.MaskedNode())


        stats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='stats'), params)
        oldstats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='stats'), params)
        preconds = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='preconds'), params)




        def lambda_init(param,state_type='stats'):
            mgd_shape = get_merged_shape(param.shape)
            print('param_shape',param.shape)
            if len(param.shape) >1 and not sum([dim>5000 for dim in param.shape]):
                blkd_shape = get_blocked_shape(mgd_shape,block_size)
                st = jnp.zeros(blkd_shape)
                paddings = get_paddings(ShampooLRPair(st,st), param, block_size)
                st_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
                st_R = get_padded_matrix_vmap(paddings.R.reshape(-1,1))
                st_L = st_L.reshape(blkd_shape[0],blkd_shape[1],block_size,block_size)
                st_R = st_R.reshape(blkd_shape[0],blkd_shape[1],block_size,block_size)
                coeff = 1.0
                # coeff = matrix_epsilon if state_type=='stats' else 1.0
                st_L = st_L*coeff
                st_R = st_R*coeff
                if precond_type=="left":
                    return LambdaRLPair(R=st_R,L=optax.MaskedNode())
                elif precond_type=="right":
                    return LambdaRLPair(R=optax.MaskedNode(),L=st_L)
                else:
                    return LambdaRLPair(R=st_R,L=st_L)
            else:
                  # blkd_shape = (1,)
                return LambdaRLPair(R=optax.MaskedNode(),L=optax.MaskedNode())

        def training_metrics(param):
            mgd_shape = get_merged_shape(param.shape)
            print('param_shape',param.shape)
            if len(param.shape) >1 and not sum([dim>5000 for dim in param.shape]):
                l_metrics = TrainingMetrics()
                r_metrics = TrainingMetrics()
                if precond_type=="left":
                    r_metrics = TrainingMetrics(root_errors=optax.MaskedNode(),root_failure_perc=optax.MaskedNode())
                    l_metrics = TrainingMetrics(root_errors_lambdas=optax.MaskedNode(),root_failure_perc_lambdas=optax.MaskedNode(),coeff=optax.MaskedNode())
                elif precond_type=="right":
                    l_metrics = TrainingMetrics(root_errors=optax.MaskedNode(),root_failure_perc=optax.MaskedNode(),coeff=optax.MaskedNode())
                    r_metrics = TrainingMetrics(root_errors_lambdas=optax.MaskedNode(),root_failure_perc_lambdas=optax.MaskedNode())
                return ShampooLRPair(L =l_metrics,R=r_metrics)
            else:
                return ShampooLRPair(L=optax.MaskedNode(),R=optax.MaskedNode())
        lambdas = jax.tree_util.tree_map(functools.partial(lambda_init,state_type='stats'),params)
        preconds_lambdas = jax.tree_util.tree_map(functools.partial(lambda_init,state_type='preconds'),params)
        metrics = jax.tree_util.tree_map(training_metrics,params)

        return ScaleByCasprState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, stats=stats, preconds=preconds,
                                 lambdas=lambdas, preconds_lambdas=preconds_lambdas, oldstats=oldstats,
                                 metrics=metrics)



    def update_fn(updates, state, params=None):
        #TODO: start preconditioning after start_preconditioning_step
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = state.count+1
        print(state.stats)
        print(updates)
        #updating statistics
        stats_updates = jax.tree_util.tree_map(
          lambda s,u: update_stats(s.L,s.R,u,block_size,b2,precond_type,log_metrics=log_metrics),
          state.stats,updates,is_leaf=lambda x: type(x).__name__=='ShampooLRPair')
        stats_paddings = jax.tree_util.tree_map(lambda s,u: get_paddings(s,u,block_size), state.stats,updates,is_leaf=lambda x: type(x).__name__=='ShampooLRPair')
        lambdas_paddings = jax.tree_util.tree_map(lambda l,u: get_paddings(l,u,block_size), state.lambdas,updates,is_leaf=lambda x: type(x).__name__=='LambdaRLPair')
        stats = jax.tree_util.tree_map(lambda x: x.stat,stats_updates,is_leaf= lambda x: isinstance(x,UpdateStats))
        stats_max_ev = jax.tree_util.tree_map(lambda x: x.stat_max_ev,stats_updates,is_leaf= lambda x: isinstance(x,UpdateStats))

        exponent = exponent_override if exponent_override !=0 else (4 if caspr_p==2 or caspr_p==-1 else 2)




        nested_preconds = {"preconds":state.preconds,"preconds_lambdas":state.preconds_lambdas}
        nested_stats = {"preconds":stats,"preconds_lambdas":state.lambdas}
        nested_paddings = {"preconds":stats_paddings,"preconds_lambdas":lambdas_paddings}
        nested_mu = {"preconds":mu,"preconds_lambdas":mu}
        curr_preconds_errors = jax.tree_util.tree_map(lambda x: x.root_errors,state.metrics,is_leaf=lambda x: isinstance(x,TrainingMetrics))
        curr_preconds_lambdas_errors = jax.tree_util.tree_map(lambda x: x.root_errors_lambdas,state.metrics,is_leaf=lambda x: isinstance(x,TrainingMetrics))
        curr_preconds_failure_perc = jax.tree_util.tree_map(lambda x: x.root_failure_perc,state.metrics,is_leaf=lambda x: isinstance(x,TrainingMetrics))
        curr_coeffs = jax.tree_util.tree_map(lambda x: x.coeff,state.metrics,is_leaf=lambda x: isinstance(x,TrainingMetrics))
        curr_preconds_lambdas_failure_perc = jax.tree_util.tree_map(lambda x: x.root_failure_perc_lambdas,state.metrics,is_leaf=lambda x: isinstance(x,TrainingMetrics))
        nested_errors = {"preconds":curr_preconds_errors, "preconds_lambdas":curr_preconds_lambdas_errors}
        # print("curr_coeffs ", curr_coeffs)
        # print("coeffs ", coeffs)

        #compute paddings


        nested_preconds,nested_errors = update_preconds_model(nested_stats,nested_preconds,nested_paddings,nested_errors,nested_mu,
                                        exponent,
                                        count_inc%preconditioning_compute_steps==0,
                                        matrix_epsilon,
                                        block_size,
                                        relative_epsilon,
                                        inverse_type,
                                        error_tolerance,batch_axis_name,precond_type)

        preconds,preconds_lambdas = nested_preconds["preconds"],nested_preconds["preconds_lambdas"]
        preconds_errors,preconds_lambdas_errors = nested_errors["preconds"],nested_errors["preconds_lambdas"]

        lambda_updates = jax.tree_util.tree_map(lambda p,l,s,u:update_lambdas(p,l,s,u,block_size,b3,count_inc,exponent,
                                                                                count_inc%preconditioning_compute_steps==0,
                                                                                matrix_epsilon,precond_type,
                                                                                log_metrics),
                                          preconds,state.lambdas,state.oldstats,updates,
                                          is_leaf=lambda x: type(x).__name__=='LambdaRLPair' or
                                          type(x).__name__=='ShampooLRPair')

        lambdas = jax.tree_util.tree_map(lambda x: x.lambd,lambda_updates, is_leaf=lambda x: isinstance(x,UpdateLambdas))
        lambdas_max_ev = jax.tree_util.tree_map(lambda x: x.lambd_max_ev,lambda_updates, is_leaf=lambda x: isinstance(x,UpdateLambdas))
        coeffs = jax.tree_util.tree_map(lambda x: x.coeff,lambda_updates, is_leaf=lambda x: isinstance(x,UpdateLambdas))
        residuals = jax.tree_util.tree_map(lambda x: x.res,lambda_updates, is_leaf=lambda x: isinstance(x,UpdateLambdas))


        def regularized_stat(stat,update):
            L,R = stat.L,stat.R


            stat_L  = optax.MaskedNode()
            if type(L).__name__!='MaskedNode':
                mgd_shape = get_merged_shape(update.shape)
                blkd_shape = get_blocked_shape(mgd_shape,block_size)
                paddings = get_paddings(stat, update, block_size)
                st_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
                st_L = st_L.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                g1,g2 = L.shape[0],L.shape[1]
                trval_L = (jnp.einsum("ijkk->ij",L)/L.shape[-1]+1e-30).reshape(-1)[:,jnp.newaxis,jnp.newaxis]
                max_ev_L = (jax.vmap(power_iteration)(L.reshape(-1,block_size,block_size)/trval_L)[1]).reshape(g1,g2,1,1)
                stat_L = L+(matrix_epsilon*trval_L.reshape(g1,g2,1,1))*max_ev_L*st_L

            stat_R = optax.MaskedNode()
            if type(R).__name__!='MaskedNode':
                mgd_shape = get_merged_shape(update.shape)
                blkd_shape = get_blocked_shape(mgd_shape,block_size)
                paddings = get_paddings(stat, update, block_size)
                g1,g2 = R.shape[0],R.shape[1]
                st_R = get_padded_matrix_vmap(paddings.R.reshape(-1,1))
                st_R = st_R.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                trval_R = (jnp.einsum("ijkk->ij",R)/R.shape[-1]+1e-30).reshape(-1)[:,jnp.newaxis,jnp.newaxis]
                max_ev_R = (jax.vmap(power_iteration)(R.reshape(-1,block_size,block_size)/trval_R)[1]).reshape(g1,g2,1,1)
                stat_R = R+(matrix_epsilon*trval_R.reshape(g1,g2,1,1))*max_ev_R*st_R
            return ShampooLRPair(L=stat_L,R=stat_R)



        oldstats = jax.lax.cond(count_inc%preconditioning_compute_steps==0,
                                lambda: jax.tree_util.tree_map(regularized_stat,stats,updates,
                                is_leaf=lambda x: type(x).__name__=='ShampooLRPair'),
                                lambda: state.oldstats)

        preconds_fail_perc = jax.tree_util.tree_map(lambda x: jnp.sum(x>error_tolerance)/x.size*100,preconds_errors)
        preconds_lambdas_fail_perc = jax.tree_util.tree_map(lambda x: jnp.sum(x>error_tolerance)/x.size*100,preconds_lambdas_errors)
        preconds_lambdas_fail_perc = jax.tree_util.tree_map(lambda x: ShampooLRPair(L=x.L,R=x.R),
                                                            preconds_lambdas_fail_perc,
                                                            is_leaf=lambda x: isinstance(x,LambdaRLPair))

        preconds_errors = jax.tree_util.tree_map(lambda x: jnp.mean(x),preconds_errors)
        preconds_lambdas_errors = jax.tree_util.tree_map(lambda x: jnp.mean(x),preconds_lambdas_errors)
        print("preconds_errors",preconds_errors)
        print("curr_preconds_errors",curr_preconds_errors)

        preconds_lambdas_errors = jax.tree_util.tree_map(lambda x: ShampooLRPair(L=x.L,R=x.R),
                                                            preconds_lambdas_errors,
                                                            is_leaf=lambda x: isinstance(x,LambdaRLPair))

        preconds_errors = jax.lax.cond(count_inc%preconditioning_compute_steps==0,lambda: preconds_errors, lambda: curr_preconds_errors)
        preconds_lambdas_errors = jax.lax.cond(count_inc%preconditioning_compute_steps==0,lambda: preconds_lambdas_errors, lambda: curr_preconds_lambdas_errors)
        coeffs= jax.lax.cond(count_inc%preconditioning_compute_steps==0,lambda: coeffs, lambda: curr_coeffs)
        preconds_fail_perc = jax.lax.cond(count_inc%preconditioning_compute_steps==0,lambda: preconds_fail_perc, lambda: curr_preconds_failure_perc)
        preconds_lambdas_fail_perc = jax.lax.cond(count_inc%preconditioning_compute_steps==0,lambda: preconds_lambdas_fail_perc, lambda: curr_preconds_lambdas_failure_perc)


        print("preconds_erors", preconds_errors)
        print("preconds_lambdas_errors",preconds_lambdas_errors)
        print("preconds_fail_perc",preconds_fail_perc)
        print("preconds_lambdas_fail_perc",preconds_lambdas_fail_perc)
        print("coeffs", coeffs)
        print("residuals",residuals)
        print("lambdas_max_ev",lambdas_max_ev)
        print("stats_max_ev",stats_max_ev)

        #collect training metrics
        metrics = jax.tree_util.tree_map(lambda e,el,f,fl,c,r,l,s:TrainingMetrics(root_errors=e,
                                                                        root_errors_lambdas=el,
                                                                        root_failure_perc=f,
                                                                        root_failure_perc_lambdas=fl,
                                                                        coeff=c,
                                                                        res=r,
                                                                        lambd=l,
                                                                        stat=s),
                                    preconds_errors,
                                    preconds_lambdas_errors,
                                    preconds_fail_perc,
                                    preconds_lambdas_fail_perc,
                                    coeffs,
                                    residuals,
                                    lambdas_max_ev,
                                    stats_max_ev,
                                    is_leaf=lambda x: isinstance(x,optax.MaskedNode) or isinstance(x,chex.Array)
                                )

        def print_fn(m,stat_type='mu'):
            print_fn = lambda m: jax.debug.print(
              "step {st} " + stat_type + " l2 {x}, " + stat_type + " l0 1e-5 {y}, " + stat_type + " l0 1e-7 {z}, " + stat_type + " l0 1e-10 {u}",
              st=count_inc, x=jnp.linalg.norm(m.reshape(-1)), y=jnp.sum(jnp.abs(m) > 1e-5),
              z=jnp.sum(jnp.abs(m) > 1e-7), u=jnp.sum(jnp.abs(m) > 1e-10)
              )
            print_fn(m)
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
          lambda p,m,u,l: caspr_update_fn(p,m,u,l,block_size,caspr_p,global_grafting,exponent,precond_type),
          preconds,mu_hat,adam_updates,preconds_lambdas,
          is_leaf=lambda x: type(x).__name__=='ShampooLRPair' or type(x).__name__=='LambdaRLPair')



        updates = jax.lax.cond(count_inc>start_preconditioning_step, lambda : caspr_updates, lambda : adam_updates)
        if verbose:
            jax.tree_util.tree_map(functools.partial(print_fn,stat_type='updates'), updates)
            jax.tree_util.tree_map(functools.partial(print_fn,stat_type='caspr_updates'), caspr_updates)
            jax.tree_util.tree_map(functools.partial(print_fn,stat_type='adam_updates'), adam_updates)
            jax.tree_util.tree_map(functools.partial(print_fn,stat_type='stats'), stats)
            jax.tree_util.tree_map(functools.partial(print_fn,stat_type='preconds'), preconds)
        return updates, ScaleByCasprState(count=count_inc, mu=mu, nu=nu, stats=stats, preconds=preconds,
                                          lambdas=lambdas,preconds_lambdas=preconds_lambdas,oldstats=oldstats,
                                          metrics=metrics)

    return optax.GradientTransformation(init_fn, update_fn)




def efficient_caspr_adaptive_full_matrix_dist_inv(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.9,
    eps: float = 1e-8,
    lamb_eps: float = 1e-6,
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
    error_tolerance: float= 1e-5,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Callable[[optax.Params], Any], None]] = None,
    global_grafting: bool = False,
    batch_axis_name: Any = None,
    precond_type: str = "all",
    log_metrics: bool = False
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
    precond_type: {precond_type},
    log_metrics: {log_metrics}
    """, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, matrix_epsilon=matrix_epsilon,
        eps_root=eps_root, block_size=block_size, preconditioning_compute_steps=preconditioning_compute_steps,
        start_preconditioning_step=start_preconditioning_step, exponent_override=exponent_override, nesterov=nesterov,
        mu_dtype=mu_dtype, caspr_p=caspr_p, relative_epsilon=relative_epsilon,
        error_tolerance=error_tolerance, weight_decay=weight_decay, mask=mask, global_grafting=global_grafting,
        precond_type=precond_type,log_metrics=log_metrics)
    return optax.chain(
        scale_by_caspr(
            b1, b2, b3, eps, lamb_eps, matrix_epsilon, eps_root, block_size,
            preconditioning_compute_steps, start_preconditioning_step,
            exponent_override, nesterov, mu_dtype,
            caspr_p, relative_epsilon, inverse_type, error_tolerance,global_grafting,batch_axis_name=batch_axis_name,precond_type=precond_type,log_metrics=log_metrics),
        optax.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate),
    )
