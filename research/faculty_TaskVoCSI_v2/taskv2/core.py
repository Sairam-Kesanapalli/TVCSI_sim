"""Controlled synthetic wireless benchmark. No trained networks or datasets are implied."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from scipy.special import j0

@dataclass
class Config:
    M: int = 16
    B: int = 16
    K: int = 8
    T: int = 140
    fc_hz: float = 3.5e9
    slot_s: float = 0.001
    speed_kmh: float = 25.
    paths: int = 4
    angle_walk: float = 0.00012  # radians/(km/h)/slot
    snr_db: float = 20.
    pilot_snr_db: float = 20.
    n_rb: int = 5
    rb_hz: float = 1e6
    feedback_bits: tuple = (0, 24, 48, 96) # four modes including predict only
    budget_bits: int = 144
    crit_share: float = 0.25
    critical_qmin: float = 0.55
    risk_penalty: float = 0.12
    bit_penalty: float = 0.00005
    switch_penalty: float = 0.002
    q_slope_db: float = 3.5
    quality_source: str = "surrogate"
    quality_table: str = ""
    # (payload bits, max semantic quality, sigmoid threshold dB)
    representations: tuple = ((350, 0.58, -5.), (1100, 0.83, -1.), (2800, 1.0, 3.))
    history: int = 5
    train_seeds: tuple = (101,102,103,104,105,106)
    val_seeds: tuple = (201,202)
    test_seeds: tuple = tuple(range(301,321))
    warmup: int = 10
    codec_epochs: int = 12
    predictor_epochs: int = 10
    batch_size: int = 256
    val_samples: int = 6000
    quick: bool = False

    def validate(self):
        assert self.M>0 and self.B>1 and self.K>0 and self.T>self.warmup
        assert self.n_rb>0 and self.history>0
        assert len(self.feedback_bits)>=2 and self.feedback_bits[0]==0
        assert all(x<y for x,y in zip(self.feedback_bits,self.feedback_bits[1:]))
        assert self.critical_qmin>=0 and self.critical_qmin<=1
        assert not (set(self.train_seeds)&set(self.val_seeds) or set(self.train_seeds)&set(self.test_seeds) or set(self.val_seeds)&set(self.test_seeds))
        # Current codec uses 6 bits per scalar: 4,8,16 latent dimensions.
        assert all(c%6==0 for c in self.feedback_bits[1:])
        return self

    def to_dict(self): return asdict(self)


def cn(rng,shape):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)


def codebook(M,B):
    n=np.arange(M)[:,None]
    u=2*np.arange(B)[None,:]/B-1
    return (np.exp(1j*np.pi*n*u)/np.sqrt(M)).astype(np.complex64)


def channel(cfg,seed):
    """AR(1) path gains with J0 Doppler lag correlation; spatial ULA multipath. Synthetic only."""
    rng=np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(3)[0])
    K,T,M,P=cfg.K,cfg.T,cfg.M,cfg.paths
    fD=(cfg.speed_kmh/3.6)*cfg.fc_hz/299792458.
    rho=float(j0(2*np.pi*fD*cfg.slot_s))
    alpha=np.zeros((K,T,P),complex)
    alpha[:,0]=cn(rng,(K,P))
    noise=cn(rng,(K,T-1,P))
    for t in range(1,T): alpha[:,t]=rho*alpha[:,t-1]+np.sqrt(max(0,1-rho*rho))*noise[:,t-1]
    angles=np.empty((K,T,P))
    angles[:,0]=rng.uniform(-np.pi/2,np.pi/2,(K,P))
    angles[:,1:]=angles[:,:1]+np.cumsum(rng.standard_normal((K,T-1,P))*cfg.angle_walk*cfg.speed_kmh,axis=1)
    n=np.arange(M)
    steering=np.exp(1j*np.pi*np.sin(angles)[...,None]*n)/np.sqrt(M)
    h=np.sqrt(M/P)*np.einsum('ktp,ktpm->ktm',alpha,steering)
    return h.astype(np.complex64)


def realization(cfg, seed):
    h=channel(cfg,seed)
    rng1,rng2=np.random.SeedSequence(int(seed)+600000).spawn(2)
    p=np.random.default_rng(rng1)
    imp=np.random.default_rng(rng2)
    # Assumes orthogonal pilots and normalized E|h_m|^2≈1; not a learned channel estimator.
    hhat=h+cn(p,h.shape).astype(np.complex64)*np.sqrt(10**(-cfg.pilot_snr_db/10))
    g=np.abs(h.conj()@codebook(cfg.M,cfg.B))**2
    critical=np.arange(cfg.K)<round(cfg.K*cfg.crit_share)
    # State-dependent importance with temporal correlations; user mix fixed across trajectory.
    high=imp.random(cfg.K)<np.where(critical,0.75,0.35)
    v=np.empty((cfg.K,cfg.T),np.float32)
    for t in range(cfg.T):
        high=np.where(high,imp.random(cfg.K)>0.08,imp.random(cfg.K)<0.15)
        v[:,t]=np.where(high,imp.uniform(.7,1.,cfg.K),imp.uniform(.1,.35,cfg.K))
    return {'h':h,'hhat':hhat,'g':g.astype(np.float32),'v':v,'critical':critical}


def sinr_from_gain(g, cfg):
    # Fixed per-RB power and independent orthogonal downlink RBs (no MU-MIMO interference).
    return np.maximum(g,0)*10**(cfg.snr_db/10)/cfg.M


@lru_cache(maxsize=8)
def _read_quality_table(path):
    with np.load(path) as z:
        return z['snr_db'].copy(),z['accuracy'].copy(),z['task_bits'].copy()


def quality(gain, cfg, critical=None, return_levels=False):
    """Task outcome expectation, conditional on physical SINR and selected representation.
    Supports synthetic smooth quality curves or a task-measured BPSK/digits accuracy LUT.
    Also enforces the finite per-user 1-RB payload capacity.
    """
    gamma=sinr_from_gain(np.asarray(gain),cfg)
    gamma_db=10*np.log10(np.maximum(gamma,1e-10))
    cap=cfg.rb_hz*cfg.slot_s*np.log2(1+gamma)
    levels=[]
    if cfg.quality_source=='digits':
        grid,accuracies,bits_vec=_read_quality_table(cfg.quality_table)
        assert tuple(int(x) for x in bits_vec)==tuple(int(x[0]) for x in cfg.representations)
    for i,(bits,qmax,th) in enumerate(cfg.representations):
        if cfg.quality_source=='digits':
            q=np.interp(gamma_db,grid,accuracies[i])
        elif cfg.quality_source=='surrogate':
            q=qmax/(1+np.exp(np.clip(-(gamma_db-th)/cfg.q_slope_db,-50,50)))
        else: raise ValueError(f'unknown quality source: {cfg.quality_source}')
        levels.append(np.where(cap>=bits,q,0.))
    out=np.stack(levels,axis=-1)
    if return_levels:return out
    return out.max(axis=-1)


def choose_representation(estimated_gain,cfg):
    return quality(estimated_gain,cfg,return_levels=True).argmax(axis=-1)


def delivered_q(true_gain,choice,cfg):
    levels=quality(true_gain,cfg,return_levels=True)
    return np.take_along_axis(levels,np.asarray(choice)[...,None],axis=-1)[...,0]
