"""Optional replay-grounded prior prediction loss; upstream files stay unchanged.

Experimental preset: one central10-transition prior rollout per replay sequence,
using recorded actions. Existing heads predict recorded targets with weight0.1.
The start posterior is detached; this auxiliary loss cannot move the encoder to
make its own prediction targets easier. No new network/optimizer/state variables.
"""
from functools import partial
import jax
import jax.numpy as jnp
from embodied.agents.dreamerv3 import agent as upstream
from embodied.agents.dreamerv3 import jaxagent

HORIZON=10
WEIGHT=0.1


def valid_steps(is_first,is_terminal,start,horizon):
    # A terminal TARGET is learnable. A terminal SOURCE or reset target ends
    # the valid chain. Never predict through an episode/reset boundary.
    valid=(~is_terminal[:,start:start+horizon]) & (~is_first[:,start+1:start+horizon+1])
    return jnp.cumprod(valid.astype(jnp.float32),axis=1)


def prior_loss(wm,data,features,ignore_inputs=()):
    length=data['is_first'].shape[1]
    if length<=HORIZON:
        raise ValueError('Prior-prediction preset requires sequence length>10')
    start=(length-HORIZON-1)//2
    state={k:jax.lax.stop_gradient(features[k][:,start]) for k in wm.rssm.initial(1)}
    actions={k:data[k][:,start:start+HORIZON] for k in wm.act_space}
    imagined=wm.rssm.imagine(state,actions)
    valid=valid_steps(data['is_first'],data['is_terminal'],start,HORIZON)
    denom=jnp.maximum(valid.sum(),1.)
    loss=jnp.array(0.,jnp.float32);metrics={}
    for name,head in wm.heads.items():
        inputs=imagined if name in wm.config.grad_heads else jax.tree_util.tree_map(jax.lax.stop_gradient,imagined)
        out=head(inputs)
        distributions=out if isinstance(out,dict) else {name:out}
        for key,dist in distributions.items():
            if key in ignore_inputs:continue
            target=jax.lax.stop_gradient(data[key][:,start+1:start+HORIZON+1].astype(jnp.float32))
            nll=-dist.log_prob(target)
            term=jnp.sum(jnp.where(valid>0,nll,0.))/denom
            loss=loss+wm.scales[key]*term
            metrics['prior_prediction_'+key+'_nll']=term
    metrics.update(prior_prediction_unweighted_loss=loss,prior_prediction_valid_fraction=valid.mean())
    return WEIGHT*loss,metrics


def augmented_loss(wm,original,data,carry,ignore_inputs=()):
    base,(features,newcarry,metrics)=original(data,carry,ignore_inputs)
    extra,diagnostics=prior_loss(wm,data,features,ignore_inputs)
    total=base+extra
    metrics={**metrics,**diagnostics,'model_base_loss':base,
             'prior_prediction_loss':extra,'model_loss':total}
    return total,(features,newcarry,metrics)


class PriorPredictionInner(upstream.Agent.inner):
    def __init__(self,obs_space,act_space,config):
        super().__init__(obs_space,act_space,config)
        # Instance-local replacement: a control agent keeps its original method,
        # even when both agent types live in the same interpreter.
        original=self.wm.loss
        self.wm.loss=partial(augmented_loss,self.wm,original)


Agent=jaxagent.Wrapper(PriorPredictionInner)
