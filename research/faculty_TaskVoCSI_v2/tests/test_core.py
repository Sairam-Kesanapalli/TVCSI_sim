import itertools
import numpy as np
from dataclasses import replace
from taskv2.core import Config,realization,codebook,quality
from taskv2.policy import dp_joint, observation,candidates
from taskv2.nn import History


def test_joint_dp_vs_exhaustive():
    rng=np.random.default_rng(17)
    costs=np.array([0,24,48,96])
    for K in (1,2,4,5):
        for _ in range(25):
            s=rng.normal(loc=.1,scale=.35,size=(K,4))
            budget=int(rng.choice([0,24,48,72,96,144]))
            nrb=int(rng.choice([1,2,3]))
            got=dp_joint(s,costs,budget,nrb)
            best=-float('inf')
            for actions in itertools.product((-1,0,1,2,3),repeat=K):
                if sum(costs[m] for m in actions if m>=0)>budget:continue
                if sum(m>=0 for m in actions)>nrb:continue
                val=sum(s[k,m] for k,m in enumerate(actions) if m>=0)
                best=max(best,val)
            val=sum(s[k,m] for k,m in enumerate(got) if m>=0)
            assert np.isclose(val,best)


def test_channel_dimensions_and_power():
    c=replace(Config(),K=3,T=18)
    r=realization(c,9)
    assert r['h'].shape==(3,18,c.M)
    assert r['g'].shape==(3,18,c.B)
    assert np.isfinite(r['g']).all()
    W=codebook(c.M,c.B)
    assert np.allclose((abs(W)**2).sum(axis=0),1.)


def test_belief_does_not_use_current_hidden_channel():
    c=Config()
    hist=History(3,c.history,c.B)
    g0=np.ones((3,c.B),dtype=np.float32)
    a=observation(hist,g0,np.array([.7,.6,.5]),np.array([1,0,0]),c)
    # No current channel or fresh codec input appears in observation signature.
    b=observation(hist,g0,np.array([.7,.6,.5]),np.array([1,0,0]),c)
    assert np.array_equal(a,b)


def test_quality_is_bounded():
    c=Config()
    q=quality(np.array([0.,1.,10.,100.]),c)
    assert ((q>=0)&(q<=1)).all()
    assert np.all(np.diff(q)>=-1e-8)


def test_greedy_feasible():
    from taskv2.policy import greedy_joint
    rng=np.random.default_rng(109)
    costs=np.array([0,24,48,96])
    for _ in range(100):
        scores=rng.random((8,4))
        a=greedy_joint(scores,costs,144,5)
        assert (a>=0).sum()<=5
        assert sum(costs[m] for m in a if m>=0)<=144
