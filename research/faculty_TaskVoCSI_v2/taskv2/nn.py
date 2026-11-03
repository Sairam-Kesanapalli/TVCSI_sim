"""Trainable, *actual quantized* feedback codec and causal irregular-feedback GRU predictor."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .core import realization

# Fixed reproducible CPU execution, no GPU requirement.
torch.set_num_threads(2)

def h_features(h):
    return np.concatenate((h.real,h.imag),axis=-1).astype(np.float32)

class ProgressiveCodec(nn.Module):
    def __init__(self,M,B,dims):
        super().__init__()
        self.dims=dims
        self.enc=nn.Sequential(nn.Linear(2*M,96),nn.ReLU(),nn.Linear(96,dims[-1]),nn.Tanh())
        self.dec=nn.Sequential(nn.Linear(dims[-1],96),nn.ReLU(),nn.Linear(96,B),nn.Softplus())
    def forward(self,x,mode):
        z=self.enc(x)
        # Six bits per latent scalar, with straight-through gradient during training.
        zq=torch.round((z+1)*31.5)/31.5-1
        z=z+(zq-z).detach()
        if mode<len(self.dims):
            mask=torch.zeros_like(z);mask[:,:self.dims[mode]]=1
            z=z*mask
        return self.dec(z)  # predicts log(1+gain)

class BeamGRU(nn.Module):
    def __init__(self,B):
        super().__init__()
        self.gru=nn.GRU(B+2,48,batch_first=True)
        self.head=nn.Sequential(nn.Linear(48,64),nn.ReLU(),nn.Linear(64,B),nn.Softplus())
    def forward(self,x):
        h,_=self.gru(x)
        return self.head(h[:,-1])


def fit_nn(model,X,Y,epochs,batch,seed,lr=0.002,mode=None):
    torch.manual_seed(seed)
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(Y))
    dl=DataLoader(ds,batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(epochs):
        model.train()
        losses=[]
        for x,y in dl:
            opt.zero_grad()
            if mode=='codec':
                pred=torch.stack([model(x,m) for m in range(len(model.dims))],dim=0)
                loss=((pred-y[None])**2).mean()
                # Ranking matters in beam selection: penalize mistakes between predicted beam gains.
                target=torch.argmax(y,dim=-1)
                ce=sum(nn.functional.cross_entropy(pred[m],target) for m in range(len(model.dims)))/len(model.dims)
                loss=loss+.08*ce
            else:
                loss=nn.functional.mse_loss(model(x),y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step();losses.append(loss.item())
    model.eval()
    return float(np.mean(losses))

@torch.no_grad()
def codec_decode(model,hhat):
    x=torch.from_numpy(h_features(hhat).reshape(-1,2*model.enc[0].in_features//2))
    out=[]
    for mode in range(len(model.dims)):
        lg=model(x,mode).numpy().reshape(*hhat.shape[:-1],-1)
        out.append(np.expm1(lg).clip(0).astype(np.float32))
    return np.stack(out,axis=-2)  # [..., modes-positive, B]

@torch.no_grad()
def predict_gain(model,hist):
    # hist [K,H,B+2] available before slot t
    loggain=model(torch.from_numpy(hist.astype(np.float32))).numpy()
    return np.expm1(loggain).clip(0).astype(np.float32)

class History:
    def __init__(self,K,H,B):
        self.K,self.H,self.B=K,H,B
        self.hist=np.zeros((K,H,B+2),np.float32)
        self.last=np.zeros((K,B),np.float32)
        self.age=np.full(K,25,dtype=np.int32)
        self.last_mode=np.zeros(K,np.int32)
        self.last_beam=np.zeros(K,np.int32)
    def update(self,gains,modes):
        # gains are decoded gains ONLY for users who fed back this slot.
        updated=modes>0
        self.last[updated]=gains[updated]
        self.age[updated]=0
        self.age[~updated]+=1
        self.hist[:,:-1]=self.hist[:,1:].copy()
        self.hist[:,-1,:self.B]=np.log1p(self.last)
        self.hist[:,-1,self.B]=np.minimum(self.age,100)/25.
        self.hist[:,-1,self.B+1]=updated.astype(np.float32)
        self.last_mode=modes.copy()


def predictor_training_data(cfg,codec,seed_list,policies=(.15,.40,.70)):
    X,Y=[],[]
    for seed in seed_list:
        r=realization(cfg,seed)
        allg=codec_decode(codec,r['hhat']) # [K,T,modes+,B]
        for j,p in enumerate(policies):
            rng=np.random.default_rng(seed*100+j)
            hist=History(cfg.K,cfg.history,cfg.B)
            for t in range(cfg.T):
                if t>=cfg.history:
                    X.append(hist.hist.copy())
                    Y.append(np.log1p(r['g'][:,t]))
                m=np.where(rng.random(cfg.K)<p,rng.integers(1,len(cfg.feedback_bits),cfg.K),0)
                received=np.zeros((cfg.K,cfg.B),np.float32)
                idx=np.flatnonzero(m>0)
                if len(idx): received[idx]=allg[idx,t,m[idx]-1]
                hist.update(received,m)
    return np.concatenate(X).astype(np.float32),np.concatenate(Y).astype(np.float32)


def train_models(cfg):
    train=[realization(cfg,s) for s in cfg.train_seeds]
    X=np.concatenate([h_features(r['hhat']).reshape(-1,2*cfg.M) for r in train]).astype(np.float32)
    Y=np.concatenate([np.log1p(r['g']).reshape(-1,cfg.B) for r in train]).astype(np.float32)
    torch.manual_seed(333)
    codec=ProgressiveCodec(cfg.M,cfg.B,[x//6 for x in cfg.feedback_bits[1:]])
    e1=fit_nn(codec,X,Y,cfg.codec_epochs,cfg.batch_size,333,mode='codec')
    Xp,Yp=predictor_training_data(cfg,codec,cfg.train_seeds)
    torch.manual_seed(444)
    predictor=BeamGRU(cfg.B)
    e2=fit_nn(predictor,Xp,Yp,cfg.predictor_epochs,cfg.batch_size,444)
    return codec.eval(),predictor.eval(),{'codec_loss':e1,'predictor_loss':e2,'n_codec':len(X),'n_predictor':len(Xp)}
