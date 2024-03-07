import functools
from typing import Any, Callable, NamedTuple, Optional, Union
from jax.scipy.linalg import cho_factor, cho_solve


import chex
import jax
import math
from jax import lax
import numpy as np
import optax
import jax.numpy as jnp


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


class ShampooLRPair(NamedTuple):
        L: chex.Array
        R: chex.Array

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


def update_stats(L,R,grad,block_size,b2):
        #TODO: update statistics once every few steps
        #  L, R = s.L, s.R
        #pad the gradient to be a multiple of block_size
        mgd_shape = get_merged_shape(grad.shape)
        if len(mgd_shape)<=1 or sum([ dim>20000 for dim in grad.shape]):
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
        L = w1*L + w2*jnp.einsum('ijkl,ijnl->ijkn',grad,grad)
        R = w1*R + w2*jnp.einsum('ijkl,ijkm->ijlm',grad,grad)
        return ShampooLRPair(L=L,R=R)



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
    error = jnp.max(jnp.abs(mm(eigvecs,(eigvals[:,jnp.newaxis]*eigvecs.T)) - reg_stat))/(max_ev+epsilon)
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



def get_inverses(stats,preconds,exponent,precondition,epsilon,
                                     block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name=None):
        def precondition_false_fn(stats,preconds):
                return preconds


        def precondition_true_fn(stats,preconds):
                if inverse_type=='eigh':
                        inverse_fn = eigh_inverse
                elif inverse_type=='cholesky':
                        inverse_fn = cholesky_inverse
                elif inverse_type=='rsvd':
                        raise NotImplemented
                #  stats = jnp.stack([L,R],axis=0)
                assert len(stats.shape)==3
                assert len(preconds.shape)==3

                old_preconds = preconds
                g = stats.shape[0]
                stats_flat = stats
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
                preconds = preconds_flat.reshape(g,stats.shape[-2],stats.shape[-1])
                errors = errors_flat[:,jnp.newaxis,jnp.newaxis]
                errors = jnp.where(jnp.isnan(errors),jnp.ones_like(errors)*(error_tolerance+1.0),errors)
                # jax.debug.print('errors: {x}', x=errors)
                preconds = jnp.where(errors>error_tolerance,old_preconds,preconds)
                # print('preconds',preconds.shape)
                return preconds
        return jax.lax.cond(precondition,precondition_true_fn,precondition_false_fn,stats,preconds)



def split_array(arr, sizes):
        # Ensure the sum of sizes equals the first dimension of the array
        assert arr.shape[0] == sum(sizes), "The sum of sizes must equal the first dimension of the array"

        # Compute the cumulative sum of sizes to get split indices
        # We use [:-1] to exclude the last element, as split indices should not include the total length
        split_indices = np.cumsum(np.array(sizes))[:-1]

        # Split the array at the computed indices
        split_arrays = jnp.split(arr, split_indices)

        return split_arrays

def update_preconds_model(stats,preconds,mu,
                                        exponent,
                                        precondition,
                                        matrix_epsilon,
                                        block_size,
                                        relative_epsilon,
                                        inverse_type,
                                        error_tolerance,
                                        batch_axis_name):
        stats_flat,tree_def = jax.tree_util.tree_flatten(stats)
        preconds_flat,tree_def = jax.tree_util.tree_flatten(preconds)
        #   assert not optax.MaskedNode() in stats_flat
        orig_shapes = []
        new_preconds_flat = []
        new_stats_flat = []
        for precond_flat,stat_flat in zip(preconds_flat,stats_flat):
                orig_shapes.append(precond_flat.shape)
                new_preconds_flat.append(precond_flat.reshape(-1,block_size,block_size))
                new_stats_flat.append(stat_flat.reshape(-1,block_size,block_size))
        preconds_flat = new_preconds_flat
        stats_flat = new_stats_flat
        print([precond.shape for precond in preconds_flat])
        preconds_flat = jnp.concatenate(preconds_flat,axis=0)
        stats_flat = jnp.concatenate(stats_flat,axis=0)
        preconds_flat = get_inverses(stats_flat,preconds_flat,exponent,precondition,matrix_epsilon,
                                        block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name)
        #unwrapping preconds_flat
        split_sizes = ([ orig_shape[0]*orig_shape[1] for orig_shape in orig_shapes ])
        print("split_sizes",split_sizes)
        print(preconds_flat.shape,np.sum(split_sizes))
        preconds_flat = split_array(preconds_flat,split_sizes)
        # new_preconds = []
        # g = len(preconds_flat)
        # for idx in range(0,g,2):
        #   precondL,precondR = preconds_flat[idx:idx+2]
        #   orig_shapeL,orig_shapeR = orig_shapes[idx:idx+2]
        #   new_preconds.append(ShampooLRPair(L=precondL.reshape(*orig_shapeL),R=precondR.reshape(*orig_shapeR)))
        preconds_flat = [ precond.reshape(orig_shape) for precond,orig_shape in zip(preconds_flat,orig_shapes)]
        new_preconds = jax.tree_util.tree_unflatten(tree_def,preconds_flat)
        new_preconds = jax.lax.cond(precondition,lambda :
                                jax.tree_util.tree_map(zeroing_preconds,
                                                                                new_preconds,mu,
                                                                                is_leaf=lambda x: type(x).__name__=='ShampooLRPair'),
                                lambda : new_preconds)
        print(new_preconds)
        return new_preconds

def zeroing_preconds(precond,momentum):

        precond_L = precond.L
        precond_R = precond.R
        mgd_shape = get_merged_shape(momentum.shape)
        if len(mgd_shape)<=1 or sum([ dim>20000 for dim in momentum.shape]):
                return precond
        #   if precond_L is optax.MaskedNode():
        #    return ShampooLRPair(L=precond_L,R=precond_R)
        block_size = precond_L.shape[-1]
        # print('preconds',preconds.shape)
        #account for paddings
        if mgd_shape[0]%block_size!=0:
                precond_L = precond_L.at[-1,:,mgd_shape[0]%block_size:,:].set(0)
                precond_L = precond_L.at[-1,:,:,mgd_shape[0]%block_size:].set(0)
        if mgd_shape[1]%block_size!=0:
                precond_R = precond_R.at[:,-1,mgd_shape[1]%block_size:,:].set(0)
                precond_R = precond_R.at[:,-1,:,mgd_shape[1]%block_size:].set(0)

        return ShampooLRPair(L=precond_L,R=precond_R)


def caspr_update_fn(precond,momentum,adam_update,block_size,caspr_p=2,global_grafting=False):
        #TODO: check whether the final momentum_reshaped retain the zeros.
        if len(adam_update.shape)==1 or sum([ dim>20000 for dim in adam_update.shape]):
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
                m1 = jnp.einsum('ijkl,ijln->ijkn',precond.L,momentum_reshaped)
            #  m1 = m1/(jnp.linalg.norm(m1.reshape(m1.shape[0],m1.shape[1],-1),axis=2)[:,:,jnp.newaxis,jnp.newaxis]+1e-30)
                m2 = jnp.einsum('ijkl,ijnl->ijnk',precond.R,momentum_reshaped)
            #  m2 = m2/(jnp.linalg.norm(m2.reshape(m2.shape[0],m2.shape[1],-1),axis=2)[:,:,jnp.newaxis,jnp.newaxis]+1e-30)
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
        
        momentum_reshaped = momentum_reshaped/jnp.linalg.norm(momentum_reshaped.reshape(-1)) * jnp.linalg.norm(adam_update.reshape(-1))


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
                        if len(param.shape) > 1 and not sum([dim>20000 for dim in param.shape]):
                                blkd_shape = get_blocked_shape(mgd_shape,block_size)
                                st = jnp.zeros(blkd_shape)
                                coeff = matrix_epsilon if state_type=='stats' else 1.0
                                st = st.at[:,:].set(coeff*jnp.eye(block_size))
                                return ShampooLRPair(L=st,R=jnp.copy(st))
                        else:
                        #  blkd_shape = (1,)
                                return ShampooLRPair(L=optax.MaskedNode(),R=optax.MaskedNode())


                stats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='stats'), params)
                preconds = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='preconds'), params)


                return ScaleByCasprState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, stats=stats, preconds=preconds)


        def update_fn(updates, state, params=None):
                #TODO: start preconditioning after start_preconditioning_step
                del params
                mu = update_moment(updates, state.mu, b1, 1)
                nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
                count_inc = state.count+1
                print(state.stats)
                print(updates)
                stats = jax.tree_util.tree_map(
                    lambda s,u: update_stats(s.L,s.R,u,block_size,b2),
                    state.stats,updates,is_leaf=lambda x: type(x).__name__=='ShampooLRPair')
                exponent = exponent_override if exponent_override !=0 else (4 if caspr_p==2 or caspr_p==-1 else 2)
                preconds = update_preconds_model(stats,state.preconds,mu,
                                                                                exponent,
                                                                                count_inc%preconditioning_compute_steps==0,
                                                                                matrix_epsilon,
                                                                                block_size,
                                                                                relative_epsilon,
                                                                                inverse_type,
                                                                                error_tolerance,batch_axis_name)
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
                    lambda p,m,u: caspr_update_fn(p,m,u,block_size,caspr_p,global_grafting),
                    preconds,mu_hat,adam_updates,
                    is_leaf=lambda x: type(x).__name__=='ShampooLRPair')

                updates = jax.lax.cond(count_inc>start_preconditioning_step, lambda : caspr_updates, lambda : adam_updates)
            #  jax.debug.print("preconds shape {x}",x=jax.tree_util.tree_flatten(preconds)[0][2].shape)
                if verbose:
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='updates'), updates)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='caspr_updates'), caspr_updates)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='adam_updates'), adam_updates)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='stats'), stats)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='preconds'), preconds)
                return updates, ScaleByCasprState(count=count_inc, mu=mu, nu=nu, stats=stats, preconds=preconds)


        return optax.GradientTransformation(init_fn, update_fn)




def efficient_caspr_dist_inv(
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



