# TaskVoCSI v2 — Auditable research implementation

This is a **new, standalone replacement experiment** inspired by the Group 5 source upload. The older sources are preserved under `legacy_sources/` for comparison; that directory is *not executable as provided*, because the original repository's `models.py`, YAML configuration and tests were not supplied.

## Contents

- `taskv2/core.py`: normalized synthetic multipath channel, perfect-realization labels, noisy pilot estimate, semantic-importance process and task-quality **surrogate**.
- `taskv2/nn.py`: PyTorch **trained progressive CSI feedback encoder/decoder with STE quantization** (6 bits/latent value) and a **trained causal GRU** using the gNB's irregular feedback history.
- `taskv2/policy.py`: offline exploration on training seeds; a trained **conditional expected task-quality and outage estimator**; exact 2-resource dynamic programming for feedback bits and scheduled users; greedy option and baselines.
- `taskv2/eval.py`: independent per-policy prediction states, paired held-out trajectories, a clearly labeled per-slot genie, source-level selection, task delivery accounting, a best-codebook-beam feasibility floor, confidence intervals and plots.
- `run.py`: train -> validate -> test -> sweep -> figures, saving provenance and model weights.
- `tests/`: DP/exhaustive tests and physical/numerical invariants.

## Run

Python >=3.10; CPU works (no GPU required).

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run.py --quick --out results/smoke
python run.py --out results/full
```

The `--quick` setting deliberately uses tiny training data and only 3 training epochs; it verifies the pipeline but **must not be reported as publication evidence**. The full run is a more meaningful initial experiment, not guaranteed to beat baselines. Freeze method hyperparameters on validation seeds, report the *test-seed* results once, and do not select models based on test performance.

Outputs: `summary.csv`, `per_seed.csv`, `test_calibration.json`, `run_metadata.json`, `networks.pt`, `calibrator.joblib`, `sweeps.csv`, and PDF plots. Each held-out test seed supplies one independent observation for confidence intervals; samples from consecutive slots are **not** treated as independent replicates.

## Research scope / limitations: read before writing the paper

1. The currently **implemented** physical system is **FDD-style downlink codebook beam selection on a ULA with orthogonal RBs and fixed per-RB transmission power**, *not* multiuser-interference-aware MU-MIMO precoding. A user either receives one RB or is unscheduled. No transmit-power optimization or true RAN deployment is claimed.
2. All semantic task quality is a **parameterized sigmoid surrogate** with selectable payload-size/quality levels. It is not derived from real images or text tasks. Replace `quality()` and `choose_representation()` with a measured per-source-level task quality vs packet success/SINR table from an actual downstream model before making semantic communication claims.
3. The channel is a synthetic, time-correlated multipath model. The uploaded `DeepMIMO` loader was a placeholder, and this version does **not** claim to validate DeepMIMO. Hold-out scenarios and independent physical-channel datasets should be added before journal or strong conference claims.
4. The CSI codec uses scalar uniform quantization and real transmitted latent dimensions; bitrate counts **6 bits per transmitted value** only. Feedback headers, CQI/PMI signaling, error protection, UL feedback errors and delay are not modeled.
5. The beam GRU is trained on simulated *irregular feedback histories*. At inference, mode 0 has access only to past information; positive-mode CSI is decoded after the feedback decision. The offline counterfactual labels use true channels, but never enter online non-genie decisions.
6. **Reliability is currently a soft penalty plus reported outage**, not a hard chance constraint. A best-beam feasibility bound is reported because even an ideal beam may fail; feasibility-aware admission/risk guarantees are future work.
7. The constrained DP is exact **only for the restricted per-slot additive estimated-value model**, with one-RB/user and a shared feedback budget; it is *not* an exact oracle for the original full P1, nor a global closed-loop policy oracle.
8. The per-slot `genie` reads true channel counterfactual outcomes and is intended as an **information benchmark**, not a guaranteed upper bound across policies with different histories.
9. The `sweep()` reuses trained models under different speed/SNR/population and therefore measures an *unadapted distribution shift*. It is not a comparison of separately retrained models.
10. A learned predictor may underperform persistence or periodic feedback. Do not delete inconvenient negative results; diagnose predictor/calibrator shift, confidence, and rate assumptions.

## Most important next research improvements

- Measure **real task accuracy** for a light image-classifier/feature-transfer task, at several representation sizes and physical link quality values; then replace the sigmoid with the measured data.
- Test at least one independent, physically generated channel dataset/trajectory set and compare LS/LMMSE or a genuinely learned CSI estimator under explicit pilot overhead assumptions.
- Refit feedback-value models with uncertainty-calibrated prediction intervals and evaluate tail-risk constraints, including impossible-outage regions.
- Introduce multiuser beam precoding/interference and genuinely joint source/RB/power optimization only if those claims will appear in the paper; otherwise use a title accurately scoped to CSI feedback and beam scheduling.
- Report actual gain-prediction correlation, balanced task classes, exact bit accounting, utility/feedback tradeoffs, and confidence intervals over *independent seeds*.

## Suggested honest conference title for THIS code version

**Calibrated Task-Value-Aware CSI Feedback and Beam Scheduling Under Limited Control Resources**

Do not represent this baseline project as a finished Tier-1 conference result without additional task data, physical validation, stronger baselines and consistently positive held-out results.

## Optional genuine (small) downstream semantic task

```bash
python run.py --quick --measured-task --out results/digits_smoke
python run.py --measured-task --out results/digits_full
```

The script trains an 8x8 handwritten-digit task model from `sklearn.datasets.load_digits`, quantizes 8/16/32 PCA features at 5 bits per real component (40/80/160 bits), and injects uncoded BPSK AWGN bit errors at a grid of signal-to-noise ratios. It measures held-out classification accuracy for every semantic source level, saves the resulting quality curves in `digits_task_quality.npz`, and uses those measured curves in scheduling. All source bits are accounted for in the restricted 1-RB transmission.

This is **real measured task accuracy**, not a simulated sigmoid; however, it is a toy application and uses *uncoded ideal BPSK*, not 5G NR LDPC/Polar/HARQ or real over-the-air measurements. It does not by itself prove semantic generalization across realistic application datasets. Generated importance weights remain synthetic and independent of digit labels.

## Model checks and paired inference

`model_diagnostics.json` records held-out top-1 beam accuracy and normalized beam regret for every quantized feedback resolution, GRU prediction and a persistence baseline, plus timing benchmarks for learned inference and feedback allocation. `paired_comparisons.csv` reports seed-paired utility differences, 95% t confidence intervals and Holm-adjusted p-values over the available baselines. These checks must be included in any research write-up. 20 independent channel test seeds are configured by default for a full experiment; the full sensitivity sweep intentionally uses only the first five seeds as an *exploratory* screening run, not conclusive statistical evidence. Increase this count when preparing final sensitivity tables.

To reuse saved trained models without retraining:

```bash
python resume.py --out results/digits_full --test-seeds 20
python resume.py --out results/digits_full --test-seeds 20 --sweeps
```

**Do not unpickle `calibrator.joblib` from untrusted sources**; this loader is intended only for locally generated project artifacts.
