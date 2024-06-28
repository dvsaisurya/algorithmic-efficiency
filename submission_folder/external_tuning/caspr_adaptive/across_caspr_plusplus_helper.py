
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


from submission_folder.external_tuning.caspr_adaptive.distributed_shampoo import matrix_inverse_pth_root,mat_power,power_iteration


ScalarOrSchedule = Union[float, optax.Schedule]




MaskOrFn = Optional[Union[Any, Callable[[optax.Params], Any]]]



class ScaleByCasprState(NamedTuple):
        count: chex.Array
        mu: optax.Updates
        nu: optax.Updates
        stats: optax.Updates
        preconds: optax.Updates
        prev_stats: optax.Updates
        across_stat:optax.Updates
        across_precond:optax.Updates


class CasprLRPair(NamedTuple):
        L: chex.Array
        R: chex.Array

def update_moment(updates, moments, decay, order):
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
        bias_correction_ = 1 - decay**count
        return jax.tree_util.tree_map(
                lambda t: t / bias_correction_.astype(t.dtype), moment)


def update_moment_per_elem_norm(updates, moments, decay, order):
        w1,w2 = (1-decay) if decay!=1.0 else 1.0, decay
        return jax.tree_util.tree_map(
                lambda g, t: w1 * g**order + w2 * t, updates, moments)




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

def get_padded_matrix(padding_start,block_size=1024):
      ix = (jnp.arange(block_size)<padding_start)
      return (jnp.eye(block_size)*ix[jnp.newaxis,:])*ix[:,jnp.newaxis]




def regularized_stat(stat_L,update,block_size,matrix_epsilon):
    get_padded_matrix_vmap = jax.vmap(functools.partial(get_padded_matrix,block_size=block_size),in_axes=0)
    mgd_shape = get_merged_shape(update.shape)
    print("mgd_shape",mgd_shape)
    blkd_shape = get_blocked_shape(mgd_shape,block_size)
    g1,g2,_,_ = blkd_shape
    paddings = get_paddings(CasprLRPair(L=stat_L,R=optax.MaskedNode()), update, block_size)

    if mgd_shape[0]<=block_size:
      #this is an unpadded stat
      # print('id_L shape',id_L.shape)
      id_L = jnp.zeros((g1,g2,mgd_shape[0],mgd_shape[0]))
      id_L = id_L.at[:,:].set(jnp.eye(mgd_shape[0],mgd_shape[0]))
      print('id_L shape after',id_L.shape)
    else:
      id_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
      id_L = id_L.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
    g1,g2 = stat_L.shape[0],stat_L.shape[1]
    trval_L = (jnp.einsum("ijkk->ij",stat_L)/stat_L.shape[-1]+1e-30).reshape(-1)[:,jnp.newaxis,jnp.newaxis]
    max_ev_L = (jax.vmap(power_iteration)(stat_L.reshape(-1,stat_L.shape[-2],stat_L.shape[-1])/trval_L)[1]).reshape(g1,g2,1,1)
    print(update,stat_L,matrix_epsilon,max_ev_L,id_L)
    stat_L = stat_L+(matrix_epsilon*trval_L.reshape(g1,g2,1,1))*max_ev_L*id_L
    return stat_L



class UpdateStats(NamedTuple):
    stat: CasprLRPair
    prev_L: Any
    coeff: Any

def update_stats(L,R,prev_L,precond_L,grad,block_size,b2,fresh_preconds,matrix_epsilon,shampoo):
    print("L R precond_L grad", L, R, precond_L, grad)
    #TODO: update statistics once every few steps
    #  L, R = s.L, s.R

    mgd_shape = get_merged_shape(grad.shape)
    if len(mgd_shape)<=1 or sum([ dim>10000 for dim in mgd_shape]):
        return UpdateStats(stat=CasprLRPair(L,R),prev_L=optax.MaskedNode(),coeff=optax.MaskedNode())
    print(mgd_shape)
    grad = grad.reshape(mgd_shape)

    #let block_size be 1000
    # a) stat of type (1,1,500,500) or (1,1,1000,1000)
    # no left/right padding or reshapeing of grad needed
    # b) stat of type (2,1,1000,1000) or (2,2,1000,1000)
    # padding and reshaping of grad needed
    # c) stat of type (1,3,500,500)
    # padding of grad not needed on the left, but reshaping needed
    tr = lambda x,y: jnp.sum(jnp.sum(x*y,axis=-1),axis=-1)

    mm = lambda x,y: jnp.einsum('ijkl,ijlm->ijkm',x,y)
    if b2==1.0:
        w1, w2 = 1.0, 1.0
    else:
        w1, w2 = b2, 1.0-b2

    if mgd_shape[0]<=block_size and mgd_shape[1]<=block_size:
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1==1 and g2==1
      # jax.debug.print("tr precond_L.precondL.L {x}",x = jnp.trace(precond_L@precond_L@L)/L.shape[-1])
      if not shampoo:
        coeff = jax.lax.cond(fresh_preconds,
                lambda: jnp.clip(tr(precond_L,
                                    regularized_stat(prev_L,
                                                    grad,
                                                    block_size,
                                                    matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/mgd_shape[0],
                lambda: jnp.ones((g1,g2,1,1)))
        # jax.debug.print((precond_L))
        prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      else:
        prev_L = optax.MaskedNode()
        coeff = optax.MaskedNode()
      if not shampoo:
        L = w1*L + w2*grad@grad.T
        precond_grad = (grad.T@precond_L)
        R = w1*coeff*R +  w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/mgd_shape[0]
      else:
        L = w1*L + w2*grad@grad.T
        R = w1*R + w2*grad.T@grad

    if mgd_shape[0]>block_size and mgd_shape[1]<=block_size:
      # L will look as g1,1,block_size,block_size and R will look as g1,1,mgd_shape[1],mgd_shape[1]
      # grad is padded to the left
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2==1
      if not shampoo:
        M = np.zeros((g1,g2))
        M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
        M[:-1,:] = block_size
        coeff = jax.lax.cond(fresh_preconds,
                                            lambda: jnp.clip(tr(precond_L, regularized_stat(prev_L,
                                                    grad,
                                                    block_size,
                                                    matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/M[:,:,np.newaxis,np.newaxis],
                                            lambda: jnp.ones((g1,g2,1,1)))
        prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      else:
        prev_L = optax.MaskedNode()
        coeff = optax.MaskedNode()
      grad = jnp.pad(grad,((0,(-mgd_shape[0])%block_size),(0,0)),mode='constant')
      grad = grad.reshape(g1,block_size,g2,mgd_shape[1])
      grad = grad.transpose((0,2,1,3))
      if not shampoo:
        L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
        precond_grad = jnp.einsum("ijkl,ijln->ijnk",precond_L,grad)
        R = w1*coeff*R + w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/M[:,:,np.newaxis,np.newaxis]
      else:
        L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
        R = w1*R + w2*jnp.einsum("ijlk,ijln->ijkn",grad,grad)
    if mgd_shape[0]<=block_size and mgd_shape[1]>block_size:
      # L will look as 1,g2,mgd_shape[0],mgd_shape[0] and R will look as 1,g2,block_size,block_size
      # grad is padded to the right
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1==1 and g2!=1
      if not shampoo:
        M = np.zeros((g1,g2))
        M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
        M[:-1,:] = block_size
        coeff = jax.lax.cond(fresh_preconds,
                                            lambda: jnp.clip(tr(precond_L, regularized_stat(prev_L,
                                                    grad,
                                                    block_size,
                                                    matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/mgd_shape[0],
                                            lambda: jnp.ones((g1,g2,1,1)))
        prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      else:
        prev_L = optax.MaskedNode()
        coeff = optax.MaskedNode()
      grad = jnp.pad(grad,((0,0),(0,(-mgd_shape[1])%block_size)),mode='constant')
      grad = grad.reshape(g1,mgd_shape[0],g2,block_size)
      grad = grad.transpose((0,2,1,3))
      if not shampoo:
        L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
        precond_grad = jnp.einsum("ijkl,ijln->ijnk",precond_L,grad)
        R = w1*coeff*R + w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/mgd_shape[0]
      else:
        L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
        R = w1*R + w2*jnp.einsum("ijlk,ijln->ijkn",grad,grad)
    if mgd_shape[0]>block_size and mgd_shape[1]>block_size:
      #L and R will look like g1,g2,block_size,block_size
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2!=1
      if not shampoo:
        M = np.zeros((g1,g2))
        M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
        M[:-1,:] = block_size
        coeff = jax.lax.cond(fresh_preconds,
                                            lambda: jnp.clip(tr(precond_L, regularized_stat(prev_L,
                                                    grad,
                                                    block_size,
                                                    matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/M[:,:,np.newaxis,np.newaxis],
                                            lambda: jnp.ones((g1,g2,1,1)))
        prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      else:
        prev_L = optax.MaskedNode()
        coeff = optax.MaskedNode()
      grad = jnp.pad(grad,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
      grad = grad.reshape(g1,mgd_shape[0],g2,block_size)
      grad = grad.transpose((0,2,1,3))
      if not shampoo:
        L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
        precond_grad = jnp.einsum("ijkl,ijln->ijnk",precond_L,grad)
        R = w1*coeff*R + w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/M[:,:,np.newaxis,np.newaxis]
      else:
        L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
        R = w1*R + w2*jnp.einsum("ijlk,ijln->ijkn",grad,grad)

    return UpdateStats(stat=CasprLRPair(L,R),prev_L=prev_L,coeff=coeff)


def eigh_inverse(stat,padding,exponent=2,epsilon=1e-6,relative_epsilon=True):
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
    inv_eigvals = (jnp.maximum(eigvals, epsilon)**(-1./exponent))

    inv_pth_reg_stat = mm(mm(eigvecs,jnp.diag(inv_eigvals)),eigvecs.T)
    inv_pth_reg_stat = (inv_pth_reg_stat*ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    error = jnp.max(jnp.abs(mat_power(inv_pth_reg_stat,p=exponent)@reg_stat - identity))
    # error = 1e-7
    return inv_pth_reg_stat*(scale**(-1/exponent)), error


def coupled_newton_inverse(stat,padding,exponent=2,epsilon=1e-6,relative_epsilon=True):
    scale = (jnp.trace(stat)/stat.shape[0]+1e-30)
    stat = stat/scale
    # ix = (jnp.arange(stat.shape[0])<padding)
    # stat = stat*(ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    inv_pth_root,metrics = matrix_inverse_pth_root(stat,exponent,ridge_epsilon=epsilon,error_tolerance=1e-6,
                            relative_matrix_epsilon=relative_epsilon,padding_start=padding)
    error = metrics.inverse_pth_root_errors

    # jax.debug.print("tr value {x}", x= jnp.sum(inv_pth_root*(stat@inv_pth_root))/padding)
    # inv_pth_root = inv_pth_root *(ix[:,jnp.newaxis])*ix[jnp.newaxis,:]

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



def get_inverses(stats,paddings,exponent,epsilon,
                                     block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name=None):

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
      errors = jnp.where(jnp.isnan(errors),jnp.ones_like(errors)*(error_tolerance+1.0),errors)
      # jax.debug.print('errors: {x}', x=errors)

      # print('preconds',preconds.shape)
      return preconds,errors




def split_array(arr, sizes):
        # Ensure the sum of sizes equals the first dimension of the array
        assert arr.shape[0] == sum(sizes), "The sum of sizes must equal the first dimension of the array"

        # Compute the cumulative sum of sizes to get split indices
        # We use [:-1] to exclude the last element, as split indices should not include the total length
        split_indices = np.cumsum(np.array(sizes))[:-1]

        # Split the array at the computed indices
        split_arrays = jnp.split(arr, split_indices)

        return split_arrays

def update_preconds_model(stats,preconds,paddings,mu,
                                        exponent,
                                        matrix_epsilon,
                                        block_size,
                                        relative_epsilon,
                                        inverse_type,
                                        error_tolerance,
                                        batch_axis_name):

        stats_flat,tree_def = jax.tree_util.tree_flatten(stats)
        paddings_flat,_ = jax.tree_util.tree_flatten(paddings)

        def pad_stat(stat):
          assert len(stat.shape)==4
          return jnp.pad(stat,((0,0),(0,0),(0,block_size-stat.shape[-2]),(0,block_size-stat.shape[-1])),mode='constant')

        #   assert not optax.MaskedNode() in stats_flat
        orig_shapes = []
        new_stats_flat = []
        new_paddings_flat = []
        for stat_flat,padding_flat in zip(stats_flat,paddings_flat):
                orig_shapes.append(stat_flat.shape)
                print(stat_flat.shape)
                new_stats_flat.append(pad_stat(stat_flat).reshape(-1,block_size,block_size))
                print(padding_flat)
                new_paddings_flat.append(padding_flat.reshape(-1,1))
        stats_flat = new_stats_flat
        paddings_flat = new_paddings_flat
        stats_flat = jnp.concatenate(stats_flat,axis=0)
        paddings_flat = jnp.concatenate(paddings_flat,axis=0)
        preconds_flat,errors_flat = get_inverses(stats_flat,paddings_flat,exponent,matrix_epsilon,
                                        block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name)
        print("orig_shapes ",orig_shapes)
        #unwrapping preconds_flat
        split_sizes = ([ orig_shape[0]*orig_shape[1] for orig_shape in orig_shapes ])
        print("split_sizes",split_sizes)
        print(preconds_flat.shape,np.sum(split_sizes))
        errors_flat = split_array(errors_flat,split_sizes)
        preconds_flat = split_array(preconds_flat,split_sizes)
        errors_flat = [ error_flat.reshape(orig_shape[:2]) for error_flat,orig_shape in zip(errors_flat,orig_shapes)]
        preconds_flat = [ precond_flat[:,:orig_shape[-2],:orig_shape[-1]].reshape(orig_shape) for precond_flat,orig_shape in zip(preconds_flat,orig_shapes)]
        errors = jax.tree_util.tree_unflatten(tree_def,errors_flat)
        new_preconds = jax.tree_util.tree_unflatten(tree_def,preconds_flat)
        new_preconds = jax.tree_util.tree_map(lambda p,op,e: jnp.where(e[:,:,jnp.newaxis,jnp.newaxis]>error_tolerance,op,p),new_preconds,preconds,errors)

        return new_preconds




def caspr_update_fn(precond,momentum,adam_update,block_size,
                    ##across_code###
                    grafting=True,
                    #end across code##
                    ):
    mgd_shape = get_merged_shape(momentum.shape)
    if len(mgd_shape)<=1 or sum([ dim>10000 for dim in mgd_shape]):
        return adam_update
    
    orig_shape= momentum.shape
    momentum = momentum.reshape(mgd_shape[0],mgd_shape[1])
    tr = lambda x,y: jnp.sum(jnp.sum(x*y,axis=-1),axis=-1)
    mm = lambda x,y: jnp.einsum('ijkl,ijlm->ijkm',x,y)
    #reshaping momentum
    if mgd_shape[0]<=block_size and mgd_shape[1]<=block_size:
      momentum = momentum[jnp.newaxis,jnp.newaxis,:,:]
    if mgd_shape[0]>block_size and mgd_shape[1]<=block_size:
      # L will look as g1,1,block_size,block_size and R will look as g1,1,mgd_shape[1],mgd_shape[1]
      # grad is padded to the left
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2==1
      momentum = jnp.pad(momentum,((0,(-mgd_shape[0])%block_size),(0,0)),mode='constant')
      momentum = momentum.reshape(g1,block_size,g2,mgd_shape[1])
      momentum = momentum.transpose((0,2,1,3))
    if mgd_shape[0]<=block_size and mgd_shape[1]>block_size:
      # L will look as 1,g2,mgd_shape[0],mgd_shape[0] and R will look as 1,g2,block_size,block_size
      # grad is padded to the right
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1==1 and g2!=1
      momentum = jnp.pad(momentum,((0,0),(0,(-mgd_shape[1])%block_size)),mode='constant')
      momentum = momentum.reshape(g1,mgd_shape[0],g2,block_size)
      momentum = momentum.transpose((0,2,1,3))
    if mgd_shape[0]>block_size and mgd_shape[1]>block_size:
      #L and R will look like g1,g2,block_size,block_size
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2!=1
      momentum = jnp.pad(momentum,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
      momentum = momentum.reshape(g1,block_size,g2,block_size)
      momentum = momentum.transpose((0,2,1,3))

    #preconditioning momentum
    momentum = jnp.einsum('ijkl,ijln,ijnm->ijkm',precond.L,momentum,precond.R)
    g1,g2,m1,m2 = momentum.shape
    momentum = momentum.transpose((0,2,1,3)).reshape(g1*m1,g2*m2)
    momentum = momentum[:mgd_shape[0],:mgd_shape[1]]
    print("momentum shape",momentum.shape)
    if grafting:
        momentum = momentum/jnp.linalg.norm(momentum,ord='fro')
        momentum = momentum*jnp.linalg.norm(adam_update.reshape(-1))
    return momentum.reshape(*orig_shape)

def get_paddings(s,G,block_size):
    L,R = s.L,s.R
    mgd_shape = get_merged_shape(G.shape)
    if len(mgd_shape)<=1 or sum([ dim>10000 for dim in G.shape]):
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
    return CasprLRPair(L=padding_size_L,R=padding_size_R)

def get_paddings_init(G,block_size):

    mgd_shape = get_merged_shape(G.shape)
    if len(mgd_shape)<=1 or sum([ dim>10000 for dim in mgd_shape]):
            return optax.MaskedNode()
    print(mgd_shape)
    blkd_shape = get_blocked_shape(mgd_shape,block_size)
    g1,g2,_,_ = blkd_shape
    print(mgd_shape, blkd_shape)

    padding_size_L = np.ones((g1,g2),dtype=np.int32)*block_size
    if mgd_shape[0]%block_size!=0:
        padding_size_L[-1,:] = mgd_shape[0]%block_size

    padding_size_R = np.ones((g1,g2),dtype=np.int32)*block_size
    if mgd_shape[1]%block_size!=0:
        padding_size_R[:,-1] = mgd_shape[1]%block_size

    #transpose the grid dimensions to the front
    return CasprLRPair(L=padding_size_L,R=padding_size_R)

def scale_by_caspr(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        matrix_epsilon: float = 1e-6,
        eps_root: float = 0.0,
        block_size: int = 1000,
        preconditioning_compute_steps: int = 20,
        start_preconditioning_step: int = 101,
        exponent_override: int = 0,
        nesterov: bool = True,
        mu_dtype: Optional[chex.ArrayDType] = None,
        caspr_p: int = -1,
        relative_epsilon: bool = True,
        inverse_type: str = 'coupled newton',
        error_tolerance: float= 1e-2,
        verbose: bool= True,
        global_grafting: bool = False,
        batch_axis_name: Any = None,
        shampoo: bool = False,
        sub_block_size: int = 10
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

                def get_padded_matrix(padding_start):
                    ix = (jnp.arange(block_size)<padding_start)
                    return (jnp.eye(block_size)*ix[jnp.newaxis,:])*ix[:,jnp.newaxis]

                get_padded_matrix_vmap = jax.vmap(get_padded_matrix,in_axes=0)

                def stat_and_precond_init(param,state_type='stats'):
                  mgd_shape = get_merged_shape(param.shape)
                  if (len(param.shape) > 1 and not sum([dim>10000 for dim in param.shape])):
                      blkd_shape = get_blocked_shape(mgd_shape,block_size)
                      coeff = matrix_epsilon if state_type in ['stats','prev_stats'] else 1.0
                      jax.debug.print(state_type+' {x}',x=coeff)
                      st = jnp.zeros(blkd_shape)
                      paddings = get_paddings_init(param, block_size)

                      # jax.debug.print("padding init fn L: {L} R: {R}", L=paddings.L,R=paddings.R)
                      if mgd_shape[0]>block_size:
                        st_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
                        st_L = st_L.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                      else:
                        assert blkd_shape[0]==1
                        st_L = jnp.zeros((blkd_shape[0],blkd_shape[1],mgd_shape[0],mgd_shape[0]))
                        st_L = st_L.at[:,:].set(jnp.eye(mgd_shape[0]))
                      st_L = st_L*coeff
                      if mgd_shape[1]>block_size:
                        st_R = get_padded_matrix_vmap(paddings.R.reshape(-1,1))
                        st_R = st_R.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                      else:
                        assert blkd_shape[1]==1
                        st_R = jnp.zeros((blkd_shape[0],blkd_shape[1],mgd_shape[1],mgd_shape[1]))
                        st_R = st_R.at[:,:].set(jnp.eye(mgd_shape[1]))
                      st_R = st_R if not shampoo else st_R*coeff
                      return CasprLRPair(L=st_L,R=st_R) if state_type in ['stats','preconds'] else st_L
                  else:
                      return CasprLRPair(L=optax.MaskedNode(),R=optax.MaskedNode()) if state_type in ['stats','preconds'] else optax.MaskedNode()

                def across_layer_stats_init(params,state_type='stats'):
                    params_flat,_ = jax.tree_util.tree_flatten(params)
                    blkd_shapes = [ get_blocked_shape(get_merged_shape(param.shape), block_size) if (len(param.shape) > 1 and not sum([dim>10000 for dim in param.shape]))  else None \
                                   for param in params_flat ]
                    num_blocks = sum([ shape[0]*shape[1] if not shape is None else 0 for shape in blkd_shapes ])
                    across_block_size = num_blocks
                    print('across_block_size', across_block_size)
                    assert block_size%sub_block_size==0
                    num_across_blocks = (block_size//sub_block_size)*(block_size//sub_block_size)
                    stat = jnp.zeros((num_across_blocks, across_block_size, across_block_size))
                    stat = stat.at[:].set((matrix_epsilon if state_type=='stats' else 1.0)*jnp.eye(across_block_size))
                    return stat

                stats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='stats'), params)
                preconds = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='preconds'), params)
                prev_stats = (jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='prev_stats'), params) \
                                if not shampoo else optax.MaskedNode())
                across_stat, across_precond = across_layer_stats_init(params), across_layer_stats_init(params,state_type='preconds')
                assert len(across_stat.shape)==3 and len(across_stat.shape)==3
                return ScaleByCasprState(count=jnp.zeros([], jnp.int32), mu=mu,
                                         nu=nu, stats=stats, preconds=preconds,
                                         prev_stats=prev_stats, 
                                         across_stat=across_stat,
                                         across_precond=across_precond)

        
        ####across code#############

        def compute_all_reduce_norm(updates):
            """
            Computes the all-reduce norm of the leaves in a JAX pytree.
            Parameters:
            updates (pytree): A JAX pytree containing the updates.
            Returns:
            float: The square root of the sum of squares of the norms of the leaves.
            """
            
            # Function to compute the norm of a leaf
            def compute_norm(leaf):
                return jnp.linalg.norm(leaf.reshape(-1))

            # Compute norms for all leaves in the pytree
            leaf_norms = jax.tree_util.tree_map(compute_norm, updates)

            # Square each norm and sum these squares
            sum_of_squares = jax.tree_util.tree_reduce(
                lambda x, y: x + y, 
                jax.tree_util.tree_map(lambda x: x**2, leaf_norms), 
                initializer=0.0
            )

            # Compute the square root of the total sum of squares
            result = jnp.sqrt(sum_of_squares)

            return result


        def update_across(across_stat, across_precond, updates, adam_updates,
                          preconds, block_size, b2,precondition=False,
                          global_grafting=False):
            """
            precondition=False: Making updates to across_stats using preconditioned updates (via preconds)  
            precondition=True: preconditioning the updates via axes preconditioners- preconds and then across-precond
            """
            
            caspr_updates = jax.tree_util.tree_map(
                    lambda p,u1,u2: caspr_update_fn(p,u1,u2,block_size,grafting=False),
                    preconds, updates, 
                    adam_updates if precondition else updates,
                    is_leaf=lambda x: type(x).__name__=='CasprLRPair')
            
            #use caspr_updates to compute layer wise statistics
            caspr_updates_flat, tree_def = jax.tree_util.tree_flatten(caspr_updates)
            caspr_updates = [ upd for upd in caspr_updates_flat if len(upd.shape)>1 and not sum([dim>10000 for dim in upd.shape])]
            caspr_updates_adam = [ upd for upd in caspr_updates_flat if not (len(upd.shape)>1 and not sum([dim>10000 for dim in upd.shape]))]
            

            def merge_pad_block_subblock(upd):
                # merge, pad, block and subblock the gradient
                orig_shape = upd.shape
                mgd_shape = get_merged_shape(orig_shape)
                upd = upd.reshape(*mgd_shape)
                upd = jnp.pad(upd,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
                blkd_shape = get_blocked_shape(mgd_shape, block_size)
                g1,g2 = blkd_shape[0],blkd_shape[1]
                blocked_update = upd.reshape(g1, block_size, g2, block_size)
                blocked_update = blocked_update.transpose((0,2,1,3))
                blocked_update = blocked_update.reshape(g1*g2, block_size, block_size)
                assert block_size%sub_block_size == 0
                h = block_size//sub_block_size
                sub_blocked_update = blocked_update.reshape(g1*g2,h,sub_block_size,h,sub_block_size)
                sub_blocked_update = sub_blocked_update.transpose(1,3,0,2,4)
                sub_blocked_update = sub_blocked_update.reshape(h**2,g1*g2,sub_block_size,sub_block_size).reshape(h**2,g1*g2,sub_block_size*sub_block_size)
                return sub_blocked_update, (g1,g2,mgd_shape[0],mgd_shape[1]),orig_shape

            caspr_updates = [ merge_pad_block_subblock(upd) for upd in caspr_updates]
            caspr_updates, shape_params, orig_shapes = zip(*caspr_updates)
            caspr_update_sizes = [ upd.shape[1] for upd in caspr_updates]
            num_blocks_per_layer = [ g1*g2 for (g1,g2,_,_) in shape_params]
            across_size = sum(num_blocks_per_layer)
            assert across_size == sum(caspr_update_sizes)
            assert across_size == across_stat.shape[1], (across_size, across_stat.shape, sum(caspr_update_sizes))
            h = block_size//sub_block_size
            caspr_updates = jnp.concatenate(caspr_updates,axis=1)
            if not precondition:
                across_stat = b2*across_stat + (1-b2)*jnp.einsum("ijk,imk->ijm",caspr_updates, caspr_updates)
                return across_stat
            else:
                caspr_updates = jnp.einsum("kij,kjm->kim", across_precond, caspr_updates)
                caspr_updates = caspr_updates.reshape(h**2, across_size, sub_block_size, sub_block_size)
                caspr_updates = caspr_updates.reshape(h,h,across_size,sub_block_size,sub_block_size)
                caspr_updates = caspr_updates.transpose(2,0,3,1,4).reshape(across_size,h*sub_block_size,h*sub_block_size)
                
                indices = np.cumsum(np.array(num_blocks_per_layer))[:-1]

                # Partition the tensor caspr_updates
                caspr_updates = jnp.split(caspr_updates, indices, axis=0)
                caspr_updates = [ upd.reshape(g1,g2,block_size,block_size)\
                                 .transpose(0,2,1,3)\
                                  .reshape(g1*block_size,g2*block_size) 
                                  for upd,(g1,g2,_,_) in zip(caspr_updates,shape_params)]
                caspr_updates = [ upd[:mgd_shape0,:mgd_shape1] for upd,(_,_,mgd_shape0,mgd_shape1) in zip(caspr_updates,shape_params)]
                caspr_updates = [ upd.reshape(*orig_shape) for upd, orig_shape in zip(caspr_updates,orig_shapes) ]
                
                # combine with adam updates for the rest of the parameter shapes.
                combined_updates = []
                precondition_index = 0
                adam_index = 0
                print("len caspr_updates_flat",len(caspr_updates_flat),caspr_updates_flat,)
                print("len caspr_updates",len(caspr_updates),caspr_updates,)
                print("len caspr_updates_adam",len(caspr_updates_adam),caspr_updates_adam,)
                for upd in caspr_updates_flat:
                    if len(upd.shape)>1 and not sum([dim>10000 for dim in upd.shape]):
                        combined_updates.append(caspr_updates[precondition_index])
                        precondition_index += 1
                    else:
                        combined_updates.append(caspr_updates_adam[adam_index])
                        adam_index += 1

                caspr_updates = jax.tree_util.tree_unflatten(tree_def, combined_updates)
                
                # layer wise grafting or global grafting
                if global_grafting:
                    global_caspr_norm = compute_all_reduce_norm(caspr_updates)  
                    global_adam_norm = compute_all_reduce_norm(adam_updates)
                    caspr_updates = jax.tree_util.tree_map(
                        lambda x: global_adam_norm*x/(1e-30+global_caspr_norm))
                else:
                    caspr_updates =jax.tree_util.tree_map(
                        lambda caspr,adam: \
                        (jnp.linalg.norm(adam.reshape(-1))*
                         caspr/(1e-30+jnp.linalg.norm(caspr.reshape(-1)))) 
                        ,caspr_updates, adam_updates)
                return caspr_updates
        ####end across code##########


        def update_fn(updates, state, params=None):
                del params
                mu = update_moment(updates, state.mu, b1, 1)
                nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
                count_inc = state.count+1
                
                print(state.stats)
                print(updates)
                if not shampoo:
                  stat_updates = jax.tree_util.tree_map(
                      lambda s,u,prevl,precondl: update_stats(s.L,s.R,prevl,precondl.L,u,block_size,b2,
                      ((jnp.maximum(count_inc-1,1))%preconditioning_compute_steps==0),matrix_epsilon,shampoo),
                      state.stats,updates,state.prev_stats,state.preconds,is_leaf=lambda x: type(x).__name__=='CasprLRPair')
                else:
                  stat_updates = jax.tree_util.tree_map(
                      lambda s,u,precondl: update_stats(s.L,s.R,optax.MaskedNode(),precondl.L,u,block_size,b2,
                      ((jnp.maximum(count_inc-1,1))%preconditioning_compute_steps==0),matrix_epsilon,shampoo),
                      state.stats,updates,state.preconds,is_leaf=lambda x: type(x).__name__=='CasprLRPair')
                

                ######across stats code##############
                across_stat = update_across(state.across_stat,
                                            state.across_precond, updates,
                                            updates, state.preconds,
                                            block_size, b2, precondition=False)
                ######end across stats code##########


                stats = jax.tree_util.tree_map(lambda x: x.stat, stat_updates,is_leaf=lambda x: type(x).__name__=='UpdateStats')
                if not shampoo:
                    prev_stats = jax.tree_util.tree_map(lambda x: x.prev_L, stat_updates,is_leaf=lambda x: type(x).__name__=='UpdateStats')
                else:
                    prev_stats = optax.MaskedNode()

                exponent = exponent_override if exponent_override !=0 else (2 if not shampoo else 4)
                stats_paddings = jax.tree_util.tree_map(lambda s,u: get_paddings(s,u,block_size), state.stats,updates,is_leaf=lambda x: type(x).__name__=='CasprLRPair')


                preconds = jax.lax.cond((count_inc)%preconditioning_compute_steps==0, lambda: update_preconds_model(stats,state.preconds,stats_paddings,mu,
                                                                                exponent,
                                                                                matrix_epsilon,
                                                                                block_size,
                                                                                relative_epsilon,
                                                                                inverse_type,
                                                                                error_tolerance,batch_axis_name),lambda: state.preconds)

                #######across stats code##############
                across_block_size = across_stat.shape[-1]
                across_precond,across_error = get_inverses(across_stat,jnp.zeros((across_stat.shape[0],1)),
                                              exponent,matrix_epsilon,across_block_size,relative_epsilon,'coupled newton',
                                              error_tolerance,batch_axis_name)  
                
                across_precond = jnp.where(across_error>=error_tolerance,state.across_precond,across_precond)
                #######end across code################

                nu_hat = bias_correction(nu, b2, count_inc)
                def nadam_fn(m,v,g):
                        return  m / (jnp.sqrt(v + eps_root) + eps)
                def nesterov_mom(m,v,g):
                        return (b1*m+(1-b1)*g) if nesterov else m
                mu_hat = jax.tree_util.tree_map(nesterov_mom,mu,nu_hat,updates)
                mu_hat = bias_correction(mu_hat, b1, count_inc)
                adam_updates = jax.tree_util.tree_map(
                        lambda m, v, g: nadam_fn(m,v,g), mu_hat, nu_hat, updates)
                
                #used adam updates for rank 1 tensors and large parameters,
                #otherwise use caspr updates
                caspr_updates = jax.tree_util.tree_map(
                    lambda p,m,u: caspr_update_fn(p,m,u,block_size),
                    preconds,mu_hat,adam_updates,
                    is_leaf=lambda x: type(x).__name__=='CasprLRPair')
                
                #########across code#####################
                caspr_updates = update_across(
                                            ####args not used###
                                            across_stat,
                                            ####################
                                            across_precond,
                                            mu, adam_updates, preconds,
                                            block_size, b2, precondition=True,
                                            global_grafting=global_grafting)
                #########################################

                updates = jax.lax.cond(count_inc>start_preconditioning_step, lambda : caspr_updates, lambda : adam_updates)
                return updates, ScaleByCasprState(count=count_inc, mu=mu, nu=nu,
                                                stats=stats, preconds=preconds,
                                                  prev_stats=prev_stats,
                                                  across_stat=across_stat,
                                                  across_precond=across_precond)


        return optax.GradientTransformation(init_fn, update_fn)




def efficient_caspr_adaptive_full_matrix_dist_inv_optimized(
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
        caspr_p: int = -1,
        relative_epsilon: bool = True,
        inverse_type: str = 'coupled newton',
        error_tolerance: float= 1e-2,
        weight_decay: float = 1e-4,
        mask: Optional[Union[Callable[[optax.Params], Any], None]] = None,
        global_grafting: bool = False,
        batch_axis_name: Any = None,
        shampoo: bool = False,
        #######across code#######
        sub_block_size = 1024,
        #######end across code###
        ) -> optax.GradientTransformation:
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
                        caspr_p, relative_epsilon, inverse_type, error_tolerance,global_grafting,batch_axis_name=batch_axis_name,shampoo=shampoo,
                        sub_block_size=sub_block_size,),
                optax.add_decayed_weights(weight_decay, mask),
                _scale_by_learning_rate(learning_rate),
        )