"""Small synthetic checks; no agent, optimizer, environment or training loop."""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/humanoid-bench/dreamerv3'))
import jax
import jax.numpy as jnp
import numpy as np
from prior_prediction import HORIZON, WEIGHT, prior_loss, valid_steps


class PriorPredictionTests(unittest.TestCase):
    def test_boundaries(self):
        first=np.zeros((3,12),bool);terminal=first.copy()
        terminal[0,3]=True  # Terminal TARGET at3 allowed, source3 ends chain.
        first[1,3]=True     # Reset TARGET at3 never predicted.
        terminal[2,0]=True  # Entire rollout invalid.
        result=valid_steps(jnp.array(first),jnp.array(terminal),0,5)
        np.testing.assert_array_equal(result,[[1,1,1,0,0],[1,1,0,0,0],[0,0,0,0,0]])

    def fake_model(self):
        class RSSM:
            def initial(self,b):return {'deter':jnp.zeros((b,1))}
            def imagine(self,z,a):
                return {'deter':z['deter'][:,None]+jnp.cumsum(a['action'],axis=1)}
        class Distribution:
            def __init__(self,p):self.p=p
            def log_prob(self,target):return -jnp.square(self.p-target).sum(-1)
        return SimpleNamespace(rssm=RSSM(),act_space={'action':None},
            config=SimpleNamespace(grad_heads=['decoder']),scales={'vector':1.},
            heads={'decoder':lambda z:{'vector':Distribution(z['deter'])}})

    def data(self):
        length=16
        return dict(is_first=jnp.zeros((1,length),bool),is_terminal=jnp.zeros((1,length),bool),
            action=jnp.ones((1,length,1)),vector=jnp.arange(length,dtype=jnp.float32)[None,:,None])

    def test_recorded_action_target_alignment(self):
        data=self.data();features={'deter':data['vector']}
        loss,metrics=prior_loss(self.fake_model(),data,features)
        self.assertEqual(float(loss),0.)
        self.assertEqual(float(metrics['prior_prediction_valid_fraction']),1.)

    def test_weight_and_detached_start(self):
        data=self.data();data['vector']=jnp.zeros_like(data['vector'])
        features={'deter':jnp.zeros_like(data['vector'])}
        wm=self.fake_model()
        loss,_=prior_loss(wm,data,features)
        self.assertAlmostEqual(float(loss),WEIGHT*np.square(np.arange(1,HORIZON+1)).mean(),places=5)
        grad=jax.grad(lambda z:prior_loss(wm,data,{'deter':z})[0])(features['deter'])
        np.testing.assert_array_equal(grad,np.zeros_like(grad))
        action_grad=jax.grad(lambda a:prior_loss(wm,{**data,'action':a},features)[0])(data['action'])
        self.assertGreater(float(jnp.linalg.norm(action_grad)),0.)

    def test_no_valid_targets_and_ignored_head(self):
        data=self.data();data['is_first']=jnp.ones_like(data['is_first'])
        features={'deter':data['vector']}
        self.assertEqual(float(prior_loss(self.fake_model(),data,features)[0]),0.)
        self.assertEqual(float(prior_loss(self.fake_model(),self.data(),features,('vector',))[0]),0.)


if __name__=='__main__':unittest.main()
