#!/usr/bin/env python3
"""Reproducible TaskVoCSI v2 pipeline. Use --quick for a smoke run; full is a research starting point."""
import argparse,json,random,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from taskv2.core import Config
from taskv2.nn import train_models
from taskv2.policy import fit_calibrator
from taskv2.eval import evaluate,sweep,plot_results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measured-task',action='store_true',help='use quantized-PCA digit classification measured task-quality table')
    p.add_argument('--quick',action='store_true',help='small smoke run; NOT publication results')
    p.add_argument('--out',default='results/v2')
    p.add_argument('--skip-sweep',action='store_true')
    p.add_argument('--test-seeds',type=int,default=None)
    a=p.parse_args()
    np.random.seed(7);random.seed(7);torch.manual_seed(7)
    cfg=Config()
    if a.quick:
        cfg=replace(cfg,T=48,train_seeds=(101,102),val_seeds=(201,),test_seeds=(301,302),
                    codec_epochs=3,predictor_epochs=3,warmup=5,quick=True)
    if a.test_seeds is not None:
        cfg=replace(cfg,test_seeds=tuple(301+i for i in range(a.test_seeds)))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    task_log={}
    if a.measured_task:
        from taskv2.digits_task import make_task_table
        path=out/'digits_task_quality.npz'
        # Each source level transmits exactly d*5-bit quantized PCA coefficients.
        cfg=replace(cfg,quality_source='digits',quality_table=str(path),rb_hz=80000.,
                    representations=((40,.65,-5.),(80,.85,-1.),(160,1.,3.)),
                    critical_qmin=.75)
        task_log=make_task_table(path,repeat=2 if a.quick else 5)
        print('Measured task curves:',task_log,flush=True)
    cfg.validate()
    (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
    start=time.perf_counter()
    print('Step 1: train real quantized progressive CSI codec and causal beam GRU',flush=True)
    codec,pred,nn_log=train_models(cfg)
    torch.save({'codec':codec.state_dict(),'predictor':pred.state_dict()},out/'networks.pt')
    print('Step 2: fit offline task-quality and outage predictors without test access',flush=True)
    cal,fit_log=fit_calibrator(cfg,codec,pred)
    import joblib
    joblib.dump(cal,out/'calibrator.joblib')
    print('Validation:',fit_log,flush=True)
    print('Step 3: run paired held-out closed-loop tests',flush=True)
    per_seed,summary=evaluate(cfg,codec,pred,cal,save=out)
    from taskv2.eval import plot_default
    plot_default(out)
    from taskv2.diagnostics import benchmark_models
    diagnostics=benchmark_models(cfg,codec,pred,cal,seed=350)
    (out/'model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(summary[['scheme','utility','utility_ci95','bits','outage','ideal_outage']].round(4).to_string(index=False),flush=True)
    if not a.skip_sweep:
        print('Step 4: run sensitivity/distribution-shift sweeps (models frozen)',flush=True)
        sweep(cfg,codec,pred,cal,out)
        plot_results(out)
    metadata={'quick':a.quick,'seconds':time.perf_counter()-start,'nn_training':nn_log,'measured_task':task_log,
              'calibration_validation':fit_log,'warning':
              'Synthetic channels; quality is a measured digit classification lookup only with --measured-task, otherwise a sigmoid surrogate. ' 
              'Fixed-power orthogonal-RB restricted system; full original P1 not implemented.'}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Saved:',out,'Elapsed seconds:',round(metadata['seconds'],1),flush=True)

if __name__=='__main__':main()
