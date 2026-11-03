"""Strict causal multi-policy simulation, paired seeds, oracle-feasibility and diagnostics."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .core import realization,choose_representation,delivered_q,quality,sinr_from_gain
from .nn import History,codec_decode,predict_gain
from .policy import observation,candidates,dp_joint,greedy_joint,baseline_modes,pre_scores

SCHEMES=('CTVC-DP','CTVC-greedy','no-importance','no-risk','periodic','age','uncertainty','importance','channel','prediction','genie')


def run_one(cfg,codec,predictor,cal,seed,schemes=SCHEMES,trace=False):
    r=realization(cfg,seed)
    K,T,B=cfg.K,cfg.T,cfg.B
    Wbits=np.array(cfg.feedback_bits)
    fb_all=codec_decode(codec,r['hhat'])
    states={name:History(K,cfg.history,B) for name in schemes}
    logs={name:[] for name in schemes}
    estimator_logs=[]
    # All schemes share exactly the same user channels and task-importance sequence.
    # Their feedback history is DIFFERENT because each scheme owns a causal state.
    for t in range(T):
        true_g=r['g'][:,t];v=r['v'][:,t]
        # Batch neural and forest inference across policies (important for tractability).
        names=list(schemes)
        h_batch=np.concatenate([states[n].hist for n in names],axis=0)
        gb=predict_gain(predictor,h_batch).reshape(len(names),K,B)
        feat=np.concatenate([observation(states[n],gb[i],v,r['critical'],cfg)
                             for i,n in enumerate(names)],axis=0)
        qp,vp=cal.predict(feat)
        qp=qp.reshape(len(names),K,-1);vp=vp.reshape(len(names),K,-1)
        for i,name in enumerate(names):
            state=states[name]
            g0=gb[i]
            q_pred,p_bad=qp[i],vp[i]
            # Offline counterfactual outcomes for metric evaluation only, never passed to online policy.
            Q_all,b_all,l_all,d_all=candidates(g0,fb_all[:,t],true_g,cfg)
            if name in ('CTVC-DP','CTVC-greedy','no-importance','no-risk','genie'):
                if name=='genie':
                    scores=pre_scores(Q_all,(Q_all<cfg.critical_qmin).astype(float),v,r['critical'],cfg)
                else:
                    scores=pre_scores(q_pred,p_bad,v,r['critical'],cfg,
                                      include_importance=name!='no-importance',
                                      include_risk=name!='no-risk')
                choose=greedy_joint(scores,Wbits,cfg.budget_bits,cfg.n_rb) if name=='CTVC-greedy' else dp_joint(scores,Wbits,cfg.budget_bits,cfg.n_rb)
                modes=np.where(choose>=0,choose,0)
                served=choose>=0
            else:
                modes=baseline_modes(name,state,g0,v,t,cfg)
                # Shared post-feedback scheduler among non-joint baselines:
                # sees the mode-specific decoded CSI only after feedback is selected.
                b=b_all[np.arange(K),modes]
                estimated=np.concatenate((g0[:,None,:],fb_all[:,t]),axis=1)
                gain_hat=estimated[np.arange(K),modes,b]
                score=v*quality(gain_hat,cfg)
                served=np.zeros(K,dtype=bool)
                served[np.argsort(-score)[:cfg.n_rb]]=True
                # Baseline may request feedback from unscheduled users, a legitimate overhead.
            assert Wbits[modes].sum()<=cfg.budget_bits
            assert served.sum()<=cfg.n_rb
            idx=np.arange(K)
            b=b_all[idx,modes]
            lev=l_all[idx,modes]
            Q=Q_all[idx,modes]*served
            # a critical user not served suffers outage; report feasible-reference outages separately.
            crit=r['critical'];viol=float((Q[crit]<cfg.critical_qmin).mean()) if crit.sum() else np.nan
            ideal_gain=true_g.max(axis=1)
            ideal_Q=quality(ideal_gain,cfg)
            infeas=float((ideal_Q[crit]<cfg.critical_qmin).mean()) if crit.sum() else np.nan
            prev_beam=state.last_beam.copy()
            switches=float(np.mean((b!=prev_beam)&served)) if t>0 else 0.
            state.last_beam=np.where(served,b,prev_beam)
            observed=np.zeros((K,B),np.float32)
            pos=np.where(modes>0)[0]
            if len(pos):observed[pos]=fb_all[pos,t,modes[pos]-1]
            state.update(observed,modes)
            taskloss=float(np.sum(v*(1-Q)))
            logs[name].append(dict(seed=seed,slot=t,scheme=name,
                utility=float((v*Q).sum()),loss=taskloss,
                regret=float(d_all[idx,modes][served].mean()) if served.any() else np.nan,
                bits=float(Wbits[modes].sum()),served=int(served.sum()),
                outage=viol,ideal_outage=infeas,switch=switches))
            if name=='CTVC-DP' and t>=cfg.warmup:
                # Evaluation-only estimator evidence, NEVER available to policy.
                estimator_logs.append((q_pred.copy(),Q_all.copy(),p_bad.copy(),(Q_all<cfg.critical_qmin).copy()))
    out=[];losses={}
    for name,ls in logs.items():
        frame=pd.DataFrame(ls)
        frame=frame[frame.slot>=cfg.warmup]
        losses[name]=frame.loss.to_numpy()
        values={c:float(frame[c].mean()) for c in ('utility','loss','regret','bits','outage','ideal_outage','switch','served')}
        values['cvar95']=float(np.mean(np.sort(frame.loss.to_numpy())[-max(1,int(np.ceil(.05*len(frame)))):]))
        out.append(dict(seed=seed,scheme=name,**values))
    est={}
    if estimator_logs:
        a=np.concatenate([x[0] for x in estimator_logs])
        b=np.concatenate([x[1] for x in estimator_logs])
        risk=np.concatenate([x[2] for x in estimator_logs]);y=np.concatenate([x[3] for x in estimator_logs])
        est={'mae':float(np.abs(a-b).mean()),'brier':float(((risk-y)**2).mean()),'gain_corr':[]}
        for m in range(1,a.shape[1]):
            ap=a[:,m]-a[:,0];bp=b[:,m]-b[:,0]
            est['gain_corr'].append(float(np.corrcoef(ap,bp)[0,1]) if ap.std()>1e-8 and bp.std()>1e-8 else 0.)
    return out,est,(logs if trace else None)


def evaluate(cfg,codec,predictor,cal,seeds=None,schemes=SCHEMES,save=None):
    seeds=list(cfg.test_seeds if seeds is None else seeds)
    assert not(set(seeds)&set(cfg.train_seeds)) and not(set(seeds)&set(cfg.val_seeds)), 'hold-out leakage'
    records=[];estimates=[]
    for seed in seeds:
        rs,e,_=run_one(cfg,codec,predictor,cal,seed,schemes)
        records.extend(rs);estimates.append(e)
    df=pd.DataFrame(records)
    summary=[]
    for name,g in df.groupby('scheme',sort=False):
        d={'scheme':name}
        for metric in ['utility','bits','outage','ideal_outage','regret','cvar95','switch','served']:
            z=g[metric].dropna().to_numpy()
            d[metric]=float(z.mean()) if len(z) else np.nan
            d[metric+'_ci95']=float(stats.t.ppf(.975,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z))) if len(z)>1 else np.nan
        summary.append(d)
    summary=pd.DataFrame(summary)
    if save:
        save=Path(save);save.mkdir(parents=True,exist_ok=True)
        (save/'test_manifest.json').write_text(json.dumps({'test_seeds':seeds,'schemes':list(schemes),
            'independent_replicates':len(seeds),'confidence_interval_unit':'seed'},indent=2))
        df.to_csv(save/'per_seed.csv',index=False)
        summary.to_csv(save/'summary.csv',index=False)
        paired_comparisons(df).to_csv(save/'paired_comparisons.csv',index=False)
        with open(save/'test_calibration.json','w') as f:json.dump({'per_seed':estimates},f,indent=2)
    return df,summary


def sweep(cfg,codec,predictor,cal,out):
    # Fixed budget capacity versus user count is handled by changes to cfg.K keeping budget fixed.
    from dataclasses import replace
    rows=[]
    for axis,values in [('budget_bits',[48,96,144,192]),('speed_kmh',[5,25,50]),('K',[4,8,12]),('snr_db',[12.,20.,28.])]:
        for val in values:
            c=replace(cfg,**{axis:val})
            # Explicit distribution shift experiments reuse frozen learned models/calibrator.
            _,summary=evaluate(c,codec,predictor,cal,seeds=cfg.test_seeds[:min(5,len(cfg.test_seeds))])
            for d in summary.to_dict('records'):rows.append(dict(axis=axis,x=val,**d))
    df=pd.DataFrame(rows)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'sweeps.csv',index=False)
    return df


def plot_results(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path)
    summary=pd.read_csv(path/'summary.csv')
    sw=pd.read_csv(path/'sweeps.csv')
    for axis in sw.axis.unique():
        fig,ax=plt.subplots(figsize=(7,4.5))
        data=sw[sw.axis==axis]
        for scheme in ['CTVC-DP','periodic','age','uncertainty','importance','genie']:
            g=data[data.scheme==scheme].sort_values('x')
            if not len(g):continue
            ax.errorbar(g.x,g.utility,yerr=g.utility_ci95,label=scheme,marker='o',capsize=2)
        ax.set_xlabel(axis);ax.set_ylabel('Task utility / slot (synthetic surrogate)')
        ax.set_title(f'TaskVoCSI-v2: utility vs {axis}');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(path/f'utility_vs_{axis}.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    g=summary.set_index('scheme')
    ax.bar(np.arange(len(g)),g.utility.values)
    ax.set_xticks(np.arange(len(g)),g.index,rotation=50,ha='right',fontsize=8)
    ax.set_ylabel('Mean synthetic task utility / slot')
    fig.tight_layout();fig.savefig(path/'default_utility.pdf');plt.close(fig)


def paired_comparisons(df,proposed='CTVC-DP',metric='utility'):
    """Paired seed-level mean differences, 95% CI, two-sided t test, Holm correction.
    Do NOT treat correlated slots as independent observations.
    """
    pivot=df.pivot(index='seed',columns='scheme',values=metric)
    out=[]
    if proposed not in pivot: return pd.DataFrame()
    for scheme in pivot:
        if scheme in (proposed,'genie'):continue
        paired=(pivot[proposed]-pivot[scheme]).dropna().to_numpy()
        if len(paired)<2:continue
        sem=paired.std(ddof=1)/np.sqrt(len(paired))
        p=stats.ttest_1samp(paired,0).pvalue
        out.append(dict(proposed=proposed,baseline=scheme,n=len(paired),
                        mean_difference=float(paired.mean()),
                        ci95_halfwidth=float(stats.t.ppf(.975,len(paired)-1)*sem),
                        p_uncorrected=float(p)))
    out.sort(key=lambda x:x['p_uncorrected'])
    max_so_far=0
    for i,d in enumerate(out):
        max_so_far=max(max_so_far,(len(out)-i)*d['p_uncorrected'])
        d['p_holm']=float(min(1.,max_so_far))
    return pd.DataFrame(out)


def plot_default(out):
    """Default experiment figures independent of expensive sweeps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    summ=pd.read_csv(out/'summary.csv')
    f,a=plt.subplots(figsize=(9,4.5))
    pos=np.arange(len(summ))
    a.bar(pos,summ.utility,yerr=summ.utility_ci95,capsize=3)
    a.set_xticks(pos,summ.scheme,rotation=45,ha='right',fontsize=8)
    a.set_ylabel('Mean expected task utility / slot')
    a.set_title('Held-out evaluation (each confidence interval uses independent seeds)')
    f.tight_layout();f.savefig(out/'fig_default_utility.pdf');plt.close(f)
    if (out/'digits_task_quality.npz').exists():
        with np.load(out/'digits_task_quality.npz') as z:
            f,a=plt.subplots(figsize=(6.5,4.3))
            for d,acc in zip(z['dimensions'],z['accuracy']):
                a.plot(z['snr_db'],acc,label=f'{int(d)} PCA features ({int(d)*5} bits)',marker='.')
            a.set(xlabel='Uncoded BPSK Eb/N0 [dB]',ylabel='Measured digit-classification accuracy',ylim=(0,1))
            a.legend();f.tight_layout();f.savefig(out/'fig_measured_semantic_task.pdf');plt.close(f)
