"""SYNTHETIC fixture - NOT real BRFSS data. Seed-generated to reproduce the DIRECTION and
APPROXIMATE MAGNITUDE of the published NHANES->BRFSS fairness finding (elderly [age>=60]
AUC < young AUC, gap ~= 0.135; Diabetes T2D paper, IEEE CIPHER 2026). It does NOT reproduce
the full published run: the published p<0.001 was computed on n=1,285,783; this fixture is
~1,200 rows, so the regression test asserts a weaker significance threshold (p<0.01) and the
gap within a tolerance band. Purpose: regression-test the audit pipeline only.

Run: python tests/fixtures/_generate_healthcare_fixture.py
"""

import os

import numpy as np
import pandas as pd

SEED = 20260626
N_PER_GROUP = 600  # half positive, half negative within each age group


def _block(rng, sep, label):
    half = N_PER_GROUP // 2
    y = np.r_[np.ones(half, int), np.zeros(half, int)]
    raw = np.r_[rng.normal(sep, 1.0, half), rng.normal(0.0, 1.0, half)]
    s = 1.0 / (1.0 + np.exp(-raw))
    return pd.DataFrame({"y_true": y, "y_score": s, "age_group": label})


def build():
    rng = np.random.default_rng(SEED)
    # sep tuned so realized AUCs land near young~0.742, elderly~0.607 (gap ~0.135)
    young = _block(rng, sep=0.92, label="young")
    elderly = _block(rng, sep=0.39, label="elderly")
    return pd.concat([young, elderly], ignore_index=True)


HEADER = (
    "# SYNTHETIC fixture - NOT real BRFSS data. Seed-generated to reproduce the direction\n"
    "# and approximate magnitude of the published elderly<young AUC gap (~0.135). Not a\n"
    "# reproduction of the full published run (published p<0.001 was on n=1,285,783).\n"
)


if __name__ == "__main__":
    df = build()
    path = os.path.join(os.path.dirname(__file__), "healthcare_subsample.csv")
    with open(path, "w", newline="") as f:
        f.write(HEADER)
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)")
