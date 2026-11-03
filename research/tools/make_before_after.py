"""Side-by-side BEFORE / AFTER images of the main figures (side track, our TVCSI_sim).

BEFORE = Phase 2 run (old mean-regret plug-in estimator, placeholder models), kept in
         deprecated/phase2_results/figures/.
AFTER  = S7 final run (learned estimator + trained codec), figures/final_20261103/.
Both use the same 10 test seeds and the same figure code. Note: the y-axis ranges differ between
the two panels (each figure scales to its own data).

Run from the repository root:  python research/tools/make_before_after.py
Writes figures/before_after/<name>_before_after.png (PNG only, never overwrites the inputs).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "deprecated" / "phase2_results" / "figures"
AFTER = ROOT / "figures" / "final_20261103"
OUT = ROOT / "figures" / "before_after"
NAMES = ["fig01_utility_vs_budget", "fig02_utility_vs_users", "fig03_utility_vs_speed",
         "fig04_utility_vs_critical_share", "fig05_utility_vs_snr", "fig10_bits_to_target_vs_speed",
         "fig12_violation_vs_budget", "fig15_objective_vs_budget"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in NAMES:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6.2), dpi=110)
        for axis, p, title in ((ax[0], BEFORE / f"{n}.png", "BEFORE: Phase 2 (old plug-in estimator, dummy models)"),
                               (ax[1], AFTER / f"{n}.png", "AFTER: S7 final (learned estimator + trained codec)")):
            axis.imshow(mpimg.imread(p))
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.axis("off")
        fig.suptitle(f"{n}  |  side track (our TVCSI_sim), 10 test seeds  |  note: y-axis ranges differ", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"{n}_before_after.png")
        plt.close(fig)
        print("wrote", OUT / f"{n}_before_after.png")


if __name__ == "__main__":
    main()
