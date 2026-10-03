"""Reproduce the sample size calculations in the synopsis (Table 3).

Run:  python src/sample_size.py
Design parameters (cluster size, class share, test share, event rate) are taken from the
panel built by 02_build_panel.py; effect sizes, alpha and power are set by the researcher.
"""
import math
import numpy as np
from scipy.stats import norm, ncf, f as fdist
from statsmodels.stats.power import TTestPower

ALPHA, POWER = 0.05, 0.80
Z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
RHO = 0.05            # conservative intra-class correlation (observed ICC of d_der is about 0)
M_FULL = 14.07        # mean quarterly observations per cell, full panel
M_TEST = 3.67         # mean observations per cell, out-of-time test period
SHARE_RQ2 = 0.5171    # share of Retail secured by real estate + Other retail
TEST_SHARE = 0.2200   # test observations / all observations (RQ3, RQ4)
PREV = 0.2096         # share of next-quarter rises above 0.10 pp (RQ4)


def de(m, rho=RHO):
    return 1 + (m - 1) * rho


def f2_n(f2, u, k):
    """Smallest n for the F-test of u coefficients in a k-predictor regression (Cohen, 1988)."""
    for n in range(k + 2, 100000):
        v = n - k - 1
        if 1 - ncf.cdf(fdist.ppf(1 - ALPHA, u, v), u, v, f2 * n) >= POWER:
            return n


def hm_var(a, n1, n0):
    """Hanley & McNeil (1982) variance of an AUC."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)


def auc_n(a1=0.70, a2=0.74, r=0.60, p=PREV):
    for n in range(100, 100000, 10):
        e = p * n
        v1, v2 = hm_var(a1, e, n - e), hm_var(a2, e, n - e)
        se = math.sqrt(v1 + v2 - 2 * r * math.sqrt(v1 * v2))
        if (a2 - a1) / se >= Z:
            return n, e, v1, v2, se


if __name__ == "__main__":
    DEF, DET = de(M_FULL), de(M_TEST)
    print(f"Design effects: full = {DEF:.4f}, test = {DET:.4f}")
    n1 = f2_n(0.02, 1, 14); N1 = math.ceil(n1 * DEF)
    print(f"RQ1: n = {n1}, x DE = {n1*DEF:.1f} -> N = {N1}")
    n2 = f2_n(0.01, 1, 17); a2 = math.ceil(n2 * DEF); N2 = math.ceil(a2 / SHARE_RQ2)
    print(f"RQ2: n = {n2}, x DE = {n2*DEF:.1f} -> {a2}; / share {SHARE_RQ2} = {a2/SHARE_RQ2:.1f} -> N = {N2}")
    n3 = math.ceil(TTestPower().solve_power(effect_size=0.10, alpha=ALPHA, power=POWER))
    t3 = math.ceil(n3 * DET); N3 = math.ceil(t3 / TEST_SHARE)
    print(f"RQ3: n = {n3}, x DE = {n3*DET:.1f} -> test {t3}; / {TEST_SHARE} = {t3/TEST_SHARE:.1f} -> N = {N3}")
    n4, e4, v1, v2, se = auc_n(); t4 = math.ceil(n4 * DET); N4 = math.ceil(t4 / TEST_SHARE)
    print(f"RQ4: n = {n4} (events {e4:.0f}), V1 = {v1:.6f}, V2 = {v2:.6f}, SE = {se:.5f}, z = {0.04/se:.3f}; "
          f"x DE = {n4*DET:.1f} -> test {t4}; / {TEST_SHARE} = {t4/TEST_SHARE:.1f} -> N = {N4}")
    print(f"Final N = max = {max(N1, N2, N3, N4)}")
