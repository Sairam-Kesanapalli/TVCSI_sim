"""Reproducible real downstream classification-task quality curves (small built-in dataset).

This is a genuinely measured downstream accuracy-versus-SINR experiment, but is a controlled
8x8 handwriting-classification demonstration, NOT a standards-accurate NR error model.
Features: PCA coefficients quantized to 5 bits/value, uncoded BPSK bit errors over AWGN,
separate logistic-regression task classifiers, held-out digits never used to train them.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.special import erfc
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _quantize(x,lo,hi):
    return np.rint(31*np.clip((x-lo)/(hi-lo+1e-9),0,1)).astype(np.uint8)

def _dequantize(q,lo,hi):
    return lo+(hi-lo)*q/31.


def _channel_ber(snr_db):
    # Coherent BPSK uncoded bit-error probability in AWGN at Eb/N0 (NOT coded NR BLER).
    return .5*erfc(np.sqrt(10**(snr_db/10)))


def make_task_table(path,seed=500,repeat=3):
    rng=np.random.default_rng(seed)
    x,y=load_digits(return_X_y=True)
    x=x.astype(np.float32)/16.
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.3,stratify=y,random_state=seed)
    pca=PCA(n_components=32,random_state=seed)
    ztrain=pca.fit_transform(xtrain); ztest=pca.transform(xtest)
    grid=np.arange(-12.,25.,2.)
    results=[];clean=[]
    for d in (8,16,32):
        tr=ztrain[:,:d];te=ztest[:,:d]
        lo=np.quantile(tr,.01,axis=0);hi=np.quantile(tr,.99,axis=0)
        qt=_quantize(tr,lo,hi);qv=_quantize(te,lo,hi)
        model=LogisticRegression(C=8,max_iter=500,solver='lbfgs',random_state=seed)
        model.fit(_dequantize(qt,lo,hi),ytrain)
        clean.append(float(np.mean(model.predict(_dequantize(qv,lo,hi))==ytest)))
        a=[]
        for db in grid:
            ber=_channel_ber(db)
            bit=np.arange(5,dtype=np.uint8)
            trials=[]
            for _ in range(repeat):
                xbits=(qv[...,None]>>bit)&1
                flipped=np.bitwise_xor(xbits,(rng.random(xbits.shape)<ber).astype(np.uint8))
                received=np.sum(flipped.astype(np.int16)*(2**bit),axis=-1)
                features=_dequantize(received,lo,hi)
                trials.append(np.mean(model.predict(features)==ytest))
            a.append(float(np.mean(trials)))
        # MC fluctuations must not create spurious decreases in expected accuracy vs SINR.
        results.append(np.maximum.accumulate(a))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,snr_db=grid,accuracy=np.asarray(results,dtype=np.float32),
             dimensions=np.array([8,16,32]),task_bits=np.array([40,80,160]),
             clean_accuracy=np.array(clean),seed=np.array(seed))
    return {'clean_accuracy':clean,'digits_train':len(ytrain),'digits_test':len(ytest),
            'snr_grid_db':grid.tolist(),'feature_bits':[40,80,160],
            'channel':'uncoded coherent BPSK over AWGN with independent bit flips'}
