"""Held-out learned-CSI model diagnostics and online inference/optimization runtimes."""
from __future__ import annotations
import time
import numpy as np
from .core import realization
from .nn import History,codec_decode,predict_gain
from .policy import dp_joint,greedy_joint,observation


def benchmark_models(cfg,codec,predictor,cal,seed=350):
    assert seed not in cfg.train_seeds and seed not in cfg.val_seeds
    rng=np.random.default_rng(seed)
    r=realization(cfg,seed)
    fb=codec_decode(codec,r['hhat'])
    K,T=cfg.K,cfg.T
    H=History(K,cfg.history,cfg.B)
    recorder={name:{'accuracy':[],'regret':[]} for name in ['GRU','persistence']+
              ['codec_'+str(m) for m in range(1,len(cfg.feedback_bits))]}
    for t in range(T):
        g=r['g'][:,t]
        best=g.argmax(axis=-1)
        maximum=g.max(axis=-1)
        belief=predict_gain(predictor,H.hist)
        for name,estimate in [('GRU',belief),('persistence',H.last)]+[(f'codec_{m}',fb[:,t,m-1]) for m in range(1,len(cfg.feedback_bits))]:
            b=estimate.argmax(axis=-1)
            recorder[name]['accuracy'].append(float(np.mean(b==best)))
            recorder[name]['regret'].append(float(np.mean(1-g[np.arange(K),b]/(maximum+1e-9))))
        m=np.where(rng.random(K)<.4,rng.integers(1,len(cfg.feedback_bits),K),0)
        observed=np.zeros((K,cfg.B),np.float32)
        k=np.flatnonzero(m>0)
        if len(k):observed[k]=fb[k,t,m[k]-1]
        H.update(observed,m)
    results={name:{a:float(np.mean(q[a][cfg.warmup:])) for a in ('accuracy','regret')}
             for name,q in recorder.items()}
    values=rng.uniform(0,1,(K,len(cfg.feedback_bits)))
    costs=np.array(cfg.feedback_bits)
    hbatch=H.hist.copy()
    obs=observation(H,np.ones((K,cfg.B),np.float32),r['v'][:,-1],r['critical'],cfg)
    timings={}
    for name,fn in [
        ('gru',lambda:predict_gain(predictor,hbatch)),
        ('calibrator',lambda:cal.predict(obs)),
        ('dp',lambda:dp_joint(values,costs,cfg.budget_bits,cfg.n_rb)),
        ('greedy',lambda:greedy_joint(values,costs,cfg.budget_bits,cfg.n_rb))]:
        fn()  # warm up
        t0=time.perf_counter()
        for _ in range(100):fn()
        timings[name+'_ms_per_call']=float(1000*(time.perf_counter()-t0)/100)
    results['timings']=timings
    return results
