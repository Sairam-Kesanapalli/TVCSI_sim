# What is verified, what is not (TaskVoCSI v2)

## Code correctness verified in the execution environment

- `python -m pytest -q`: all five tests pass, including 100+ random knapsack/RB cases checked against exhaustive enumeration.
- `python run.py --quick --measured-task ...` executes the complete pipeline, produces measured digit task accuracy, trained quantized CSI codec, trained GRU, conditional quality/outage regressor, comparative simulations, 13 sweep conditions, and PDF figures. This run has tiny training data and is NOT publication evidence.
- `python run.py --measured-task --skip-sweep --out results/digits_full` executed and saved trained model artifacts plus calibration/test metrics; `resume.py --out results/digits_full --test-seeds 10` confirmed 10 held-out seeds.

## Preliminary 10-seed measured-task test

The exported `results/digits_full/summary.csv` is the source of truth. Values in this run included:

| scheme | Mean task utility | Mean CSI bits/slot |
|---|---:|---:|
| CTVC-DP | 3.6018 | 136.69 |
| periodic | 3.4647 | 144.00 |
| age | 3.4787 | 144.00 |
| uncertainty | 3.5009 | 144.00 |
| importance | 3.5321 | 144.00 |
| channel-quality | 3.5277 | 144.00 |
| genie (privileged information) | 3.8497 | 15.30 |

Paired comparisons use the exact seed-paired test, not separate mean confidence interval overlap. Holm-adjusted p-values are in `paired_comparisons.csv`; the differences versus channel-quality-based and uncertainty baselines do NOT both survive this correction at a 0.05 threshold. The no-risk ablation is comparable to the proposed risk-aware policy; no reliability guarantee has been demonstrated. These results must not be cherry-picked as universal superiority.

## Work remaining before an ambitious conference submission

1. Run at least 20–30 independent seeds for the final paper and expand independent scenarios. Tune only on train/validation seeds, never on held-out tests.
2. Replace the synthetic channel with physically realistic, independently split mobility channel traces and retain the synthetic model as a controlled ablation.
3. Replace the small handwriting task with a credible application (perception or edge inference) and use actual task errors under a coded NR-like PHY, including practical CSI/feedback overhead.
4. Incorporate a reliable critical-message guarantee/admission mechanism. A soft risk penalty is not a hard chance constraint. Quantify infeasibility and service denial separately.
5. If claiming the original full P1, implement real MU-MIMO interference, RB/power optimization and matching solution baselines. Otherwise publish the narrower fixed-power OFDMA formulation as implemented.
6. Measure end-to-end model inference latency in the target deployment environment. Current timing numbers are CPU development-machine benchmarks.

## Reproducibility caveats

- The generated example artifacts and saved weights are convenience snapshots. The pre-trained `.joblib` should only be loaded from a trusted archive.
- Running `run.py` retrains and overwrites previous results in `--out`. Use a new folder for each experimental setting.
- Trained test predictions and simulator metrics are driven by the neural model and calibrator; numerical performance is not guaranteed to replicate bit-for-bit across PyTorch/NumPy/BLAS versions or hardware. Code seeds are fixed for repeatability, not as a substitute for multi-seed experiments.
