"""Calibrated conditional task-quality estimation and exact restricted MCKP + RB solver."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .core import delivered_q,choose_representation,sinr_from_gain
from .nn import History,codec_decode,predict_gain
from .core import realization


def observation(hist, g0, v, critical, cfg):
    """Only information already held by the gNB at the START of slot t.
    No h(t), pilot hhat(t), decoded *current* feedback, or true current outcomes.
    """
    idx=np.argsort(g0,axis=1)
    best=np.take_along_axis(g0,idx[:,-1:],axis=1)[:,0]
    second=np.take_along_axis(g0,idx[:,-2:-1],axis=1)[:,0]
    features=np.column_stack((np.log1p(g0),np.log1p(hist.last),
                              np.minimum(hist.age,100)/25.,
                              best/(best+second+1e-6),
                              np.log1p(best),
                              hist.last_mode/3.,
                              critical.astype(float),
                              np.full(len(g0),cfg.snr_db/30.),
                              np.full(len(g0),cfg.speed_kmh/60.),
                              np.log1p(sinr_from_gain(best,cfg))))
    return features.astype(np.float32)


def candidates(g0,fb,real_g,cfg):
    """Counterfactual per-mode task quality, computed ONLY for offline labels/oracle/evaluation.
    Semantic representation selected using the corresponding *estimated* beam gain.
    Physical success evaluated using hidden true channel gains.
    """
    ghat=np.concatenate((g0[:,None,:],fb),axis=1)
    b=ghat.argmax(axis=-1)   # [K,modes]
    estimated=np.take_along_axis(ghat,b[...,None],axis=-1)[...,0]
    levels=choose_representation(estimated,cfg)
    actual=np.take_along_axis(np.broadcast_to(real_g[:,None,:],ghat.shape),b[...,None],axis=-1)[...,0]
    Q=delivered_q(actual,levels,cfg)
    # Regret 0..1 and ideal quality with exact best beam and oracle source-level decision.
    regret=1-actual/(real_g.max(axis=-1)[:,None]+1e-9)
    return Q,b,levels,regret


def exploratory_samples(cfg,codec,predictor,seeds,probabilities=(.15,.40,.75)):
    X,Y,Vi=[],[],[]
    for seed in seeds:
        r=realization(cfg,seed)
        fb_all=codec_decode(codec,r['hhat'])
        for j,p in enumerate(probabilities):
            hist=History(cfg.K,cfg.history,cfg.B)
            rng=np.random.default_rng(int(seed)*101+j)
            for t in range(cfg.T):
                g0=predict_gain(predictor,hist.hist)
                v=r['v'][:,t]
                features=observation(hist,g0,v,r['critical'],cfg)
                Q,_,_,_=candidates(g0,fb_all[:,t],r['g'][:,t],cfg)
                X.append(features)
                Y.append(Q.astype(np.float32))
                Vi.append((Q<cfg.critical_qmin).astype(np.float32))
                m=np.where(rng.random(cfg.K)<p,
                           rng.integers(1,len(cfg.feedback_bits),size=cfg.K),0)
                observed=np.zeros((cfg.K,cfg.B),np.float32)
                k=np.where(m>0)[0]
                if len(k):observed[k]=fb_all[k,t,m[k]-1]
                hist.update(observed,m)
    return np.concatenate(X),np.concatenate(Y),np.concatenate(Vi)

@dataclass
class Calibrator:
    quality: object
    violation: object

    def predict(self,features):
        q=np.clip(self.quality.predict(features),0,1)
        v=np.clip(self.violation.predict(features),0,1)
        return q,v


def fit_calibrator(cfg,codec,predictor):
    X,Y,V=exploratory_samples(cfg,codec,predictor,cfg.train_seeds)
    # Explicit sample-level supervision across all modes, not E[regret] substituted into Q.
    qm=ExtraTreesRegressor(n_estimators=48,min_samples_leaf=8,n_jobs=1,random_state=17,max_features=.8)
    vm=ExtraTreesRegressor(n_estimators=32,min_samples_leaf=10,n_jobs=1,random_state=18,max_features=.8)
    qm.fit(X,Y);vm.fit(X,V)
    Xv,Yv,Vv=exploratory_samples(cfg,codec,predictor,cfg.val_seeds)
    pred,prob=np.clip(qm.predict(Xv),0,1),np.clip(vm.predict(Xv),0,1)
    metrics={
        'val_q_mae':float(np.mean(np.abs(pred-Yv))),
        'val_q_rmse':float(np.sqrt(np.mean((pred-Yv)**2))),
        'val_outage_brier':float(np.mean((prob-Vv)**2)),
        'val_n':int(len(Xv)),
        'per_mode_mae':[float(np.mean(np.abs(pred[:,j]-Yv[:,j]))) for j in range(Y.shape[1])],
        'per_mode_gain_corr':[]
    }
    for j in range(1,Y.shape[1]):
        a,b=pred[:,j]-pred[:,0],Yv[:,j]-Yv[:,0]
        c=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-10 and b.std()>1e-10 else 0.
        metrics['per_mode_gain_corr'].append(c)
    return Calibrator(qm,vm),metrics


def dp_joint(scores,costs,budget,n_rb):
    """Exact per-slot resource allocation for restricted model:
    each user is either unserved (-1), or served using exactly 1 RB and feedback mode 0..F.
    Feedback costs integer, capacity <= budget; complexity O(K * budget_units * RBs * F).
    For the committed fixed-power OFDMA model only; NOT an exact solver for original P1.
    """
    scores=np.asarray(scores,dtype=float)
    costs=np.asarray(costs,dtype=np.int64)
    assert costs[0]==0 and np.all(costs[1:]>=0)
    K,M=scores.shape
    positive=costs[costs>0]
    step=int(np.gcd.reduce(positive)) if len(positive) else 1
    assert np.all(costs%step==0)
    w=costs//step
    cap=int(np.floor(budget/step+1e-10))
    R=min(n_rb,K)
    dp=np.full((cap+1,R+1),-np.inf);dp[0,0]=0.
    back=np.full((K,cap+1,R+1),-2,dtype=np.int16)
    for k in range(K):
        nxt=dp.copy()
        back[k][np.isfinite(dp)]=-1  # skip user
        for m in range(M):
            c=int(w[m])
            if c>cap:continue
            cand=dp[:cap-c+1,:R]+scores[k,m]
            target=nxt[c:,1:]
            better=cand>target+1e-12
            target[better]=cand[better]
            flags=back[k,c:,1:]
            flags[better]=m
        dp=nxt
    end=np.unravel_index(int(np.argmax(dp)),dp.shape)
    used,rbs=end
    actions=np.full(K,-1,dtype=np.int16)
    for k in range(K-1,-1,-1):
        a=int(back[k,used,rbs]);assert a>=-1
        actions[k]=a
        if a>=0:used-=int(w[a]);rbs-=1
    assert sum(costs[a] for a in actions if a>=0)<=budget
    assert (actions>=0).sum()<=n_rb
    return actions


def greedy_joint(scores,costs,budget,n_rb):
    """Construct feasible schedule at mode zero, then repeatedly buy best marginal upgrades.
    Includes profitable swaps: replaces a currently served user by a more valuable one
    when the replacement can use available feedback capacity.
    No optimality guarantee, but is not artificially biased toward zero-cost actions.
    """
    K,M=scores.shape
    actions=np.full(K,-1,dtype=int)
    top=np.argsort(-scores[:,0])[:n_rb]
    actions[top]=np.where(scores[top,0]>0,0,-1)
    left=float(budget)
    while True:
        best=(1e-12,None)
        for k in range(K):
            old=int(actions[k]);old_val=scores[k,old] if old>=0 else 0.
            old_cost=costs[old] if old>=0 else 0
            for m in range(M):
                if m==old:continue
                delta=scores[k,m]-old_val
                new_cost=costs[m]-old_cost
                if delta<=0 or new_cost>left:continue
                if old<0 and np.sum(actions>=0)>=n_rb:
                    for j in np.flatnonzero(actions>=0):
                        freed=float(costs[actions[j]])
                        gain=delta-scores[j,actions[j]]
                        if gain>0 and new_cost-freed<=left:
                            ratio=gain/(1+max(new_cost-freed,0))
                            if ratio>best[0]: best=(ratio,(k,m,j))
                else:
                    ratio=delta/(1+max(new_cost,0))
                    if ratio>best[0]:best=(ratio,(k,m,None))
        if best[1] is None:break
        k,m,j=best[1]
        if j is not None:
            left+=costs[actions[j]]
            actions[j]=-1
        if actions[k]>=0:left+=costs[actions[k]]
        actions[k]=m;left-=costs[m]
    assert np.sum(actions>=0)<=n_rb and left>=-1e-8
    return actions


def pre_scores(q,violation,v,critical,cfg,include_importance=True,include_risk=True):
    w=v if include_importance else np.ones_like(v)
    values=w[:,None]*q-np.asarray(cfg.feedback_bits)[None,:]*cfg.bit_penalty
    if include_risk:
        values=values-cfg.risk_penalty*critical[:,None]*violation
    return values


def baseline_modes(name, hist, g0, v, t,cfg):
    """Select fresh-feedback users without current CSI, obeying the *same* bit budget."""
    K=len(v);costs=np.asarray(cfg.feedback_bits)
    modes=np.zeros(K,dtype=int)
    high=len(costs)-1
    if name=='prediction':return modes
    if name=='periodic':order=np.roll(np.arange(K),-(t%K))
    elif name=='age':order=np.argsort(-hist.age,kind='stable')
    elif name=='importance':order=np.argsort(-v,kind='stable')
    elif name=='channel':order=np.argsort(g0.max(axis=1),kind='stable')
    elif name=='uncertainty':
        sorted_g=np.sort(g0,axis=1)
        margin=(sorted_g[:,-1]-sorted_g[:,-2])/(sorted_g[:,-1]+1e-6)
        order=np.argsort(margin,kind='stable')
    else:raise ValueError(name)
    # Fill the shared budget using multiple available feedback resolutions;
    # avoids leaving e.g. 48 bits unused after one 96-bit request.
    left=int(cfg.budget_bits)
    for k in order:
        feasible=np.where(costs<=left)[0]
        mode=int(feasible[-1])
        if mode==0:break
        modes[k]=mode
        left-=int(costs[mode])
    return modes
