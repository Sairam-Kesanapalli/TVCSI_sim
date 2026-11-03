#!/usr/bin/env python3
"""Re-evaluate a TRAINED model on additional never-before-used channel seeds and sweep shifts."""
import argparse,json
from pathlib import Path
import joblib,torch
from taskv2.core import Config
from taskv2.nn import ProgressiveCodec,BeamGRU
from taskv2.eval import evaluate,sweep,plot_results,plot_default


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/digits_full')
    p.add_argument('--test-seeds',type=int,default=None)
    p.add_argument('--sweeps',action='store_true')
    a=p.parse_args()
    out=Path(a.out)
    config=json.loads((out/'config.json').read_text())
    # Support extraction to a different computer/project folder.
    if config.get('quality_source')=='digits' and not Path(config.get('quality_table','')).exists():
        config['quality_table']=str(out/'digits_task_quality.npz')
    cfg=Config(**config).validate()
    if a.test_seeds: cfg.test_seeds=tuple(range(301,301+a.test_seeds));cfg.validate()
    dims=[int(n)//6 for n in cfg.feedback_bits[1:]]
    codec=ProgressiveCodec(cfg.M,cfg.B,dims)
    pred=BeamGRU(cfg.B)
    weights=torch.load(out/'networks.pt',map_location='cpu',weights_only=True)
    codec.load_state_dict(weights['codec']);pred.load_state_dict(weights['predictor'])
    codec.eval();pred.eval()
    # Treat joblib files as trusted local project artifacts; joblib is unsafe for untrusted files.
    cal=joblib.load(out/'calibrator.joblib')
    _,results=evaluate(cfg,codec,pred,cal,save=out)
    plot_default(out)
    from taskv2.diagnostics import benchmark_models
    (out/'model_diagnostics.json').write_text(json.dumps(benchmark_models(cfg,codec,pred,cal,seed=350),indent=2))
    print(results[['scheme','utility','utility_ci95','bits','outage']].round(4).to_string(index=False),flush=True)
    if a.sweeps:
        sweep(cfg,codec,pred,cal,out)
        plot_results(out)
    print('Results:',out)

if __name__=='__main__':main()
