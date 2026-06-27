"""SYNTHETIC fixture - NOT real HMDA data. Seed-generated to reproduce the DIRECTION and
APPROXIMATE MAGNITUDE of the mortgage-fairness findings in the lending papers (minority
approval rate < reference, gap WIDENING over years; plus a heterogeneous racial effect on
approval concentrated where x0 > 0). It does NOT reproduce the published HMDA runs (which
used millions of records); this fixture is ~1,500 synthetic rows. Purpose: regression-test
the lending audit pipeline only - the descriptive annual disparate-impact gap and (when
econml is installed) the sign of the subgroup CATE.

Run: python tests/fixtures/_generate_lending_fixture.py
"""

import os

import numpy as np
import pandas as pd

SEED = 20260627
YEARS = [2019, 2020, 2021]
N_PER_GROUP = 250  # per group, per year
YEAR_PENALTY = {2019: 0.10, 2020: 0.25, 2021: 0.40}  # widening minority penalty
HET = 0.5  # extra minority penalty where x0 > 0 (planted heterogeneous effect)


def build():
    rng = np.random.default_rng(SEED)
    frames = []
    for yr in YEARS:
        for grp in ("reference", "minority"):
            n = N_PER_GROUP
            x0 = rng.normal(size=n)
            x1 = rng.normal(size=n)
            x2 = rng.normal(size=n)
            t = 1.0 if grp == "minority" else 0.0
            latent = (
                0.8
                + 0.5 * x0
                + 0.3 * x1
                - (YEAR_PENALTY[yr] + HET * (x0 > 0)) * t
                + rng.normal(scale=0.3, size=n)
            )
            approved = (latent > 0.5).astype(int)
            frames.append(
                pd.DataFrame(
                    {
                        "year": yr,
                        "group": grp,
                        "approved": approved,
                        "x0": x0,
                        "x1": x1,
                        "x2": x2,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


HEADER = (
    "# SYNTHETIC fixture - NOT real HMDA data. Seed-generated to reproduce the direction\n"
    "# and approximate magnitude of the mortgage-fairness findings (minority approval rate\n"
    "# < reference, gap widening over years; heterogeneous effect where x0>0). Not a\n"
    "# reproduction of the published HMDA runs (millions of records).\n"
)


if __name__ == "__main__":
    df = build()
    path = os.path.join(os.path.dirname(__file__), "lending_subsample.csv")
    with open(path, "w", newline="") as f:
        f.write(HEADER)
        df.to_csv(f, index=False)
    print(f"wrote {path} ({len(df)} rows)")
