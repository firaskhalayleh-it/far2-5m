"""Pre-registered statistical machinery (v0.3). Every gate is an inequality
over defined estimators — margin AND significance — never wording. The formal
decision rules live in docs/CRITERIA.md; this module is their implementation
(pure math, no scipy dependency; exact tests where n permits).

Conventions (fixed across the project):
  alpha            0.05, one-sided, per pre-registered comparison
  proportion CI    Wilson score interval at z = 1.96
  superiority      p1 > p2 + m: two-proportion z on (p1 - p2), pooled SE,
                   gate = (phat1 - phat2 >= m) AND (p_value < alpha)
  paired probes    exact McNemar (binomial b vs b+c at 1/2)
  small n          any test with fewer than N_MIN=20 eligible episodes is
                   reported as NOT-TESTABLE, never as a pass
"""

import math

ALPHA = 0.05
Z95 = 1.959963984540054
N_MIN = 20


def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p)."""
    if n < 0 or not 0.0 <= p <= 1.0:
        return float("nan")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    lp, lq = math.log(p), math.log(1.0 - p)
    total = 0.0
    for i in range(k, n + 1):
        total += math.exp(_log_comb(n, i) + i * lp + (n - i) * lq)
    return min(1.0, total)


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = ph + z2 / (2 * n)
    spread = z * math.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n))
    return ((centre - spread) / den, (centre + spread) / den)


def _phi_sf(z: float) -> float:
    """P(Z >= z) for standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_prop_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """One-sided two-proportion z-test, H1: p1 > p2. Returns the difference,
    its Wald 95% CI, z, and the one-sided p-value (pooled SE). NaNs when a
    group is empty."""
    if n1 <= 0 or n2 <= 0:
        nan = float("nan")
        return {"d": nan, "lo": nan, "hi": nan, "z": nan, "p": nan,
                "n1": n1, "n2": n2}
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    pp = (k1 + k2) / (n1 + n2)
    se_pooled = math.sqrt(max(pp * (1 - pp) * (1 / n1 + 1 / n2), 1e-12))
    se_wald = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = d / se_pooled
    return {"d": d, "lo": d - Z95 * se_wald, "hi": d + Z95 * se_wald,
            "z": z, "p": _phi_sf(z), "n1": n1, "n2": n2}


def ols2(y: list[float], x1: list[float], x2: list[float]) -> dict:
    """OLS of y on TWO standardized regressors — the Mathematical
    Generalization Surface test: e.g. iterations on (D, L). Returns per-
    regressor beta (standardized), z and one-sided p for beta > 0, plus n.
    The claim pattern: beta_D > 0 significant while |beta_L| stays small."""
    n = len(y)
    if n < N_MIN or len(set(x1)) < 2 or len(set(x2)) < 2:
        return {"n": n, "testable": False}

    def std(v):
        mu = sum(v) / n
        sd = (sum((a - mu) ** 2 for a in v) / max(n - 1, 1)) ** 0.5
        return [(a - mu) / sd for a in v] if sd > 0 else [0.0] * n

    ys, a, b = std(y), std(x1), std(x2)
    # normal equations for [b1, b2] on standardized data (intercept = 0)
    saa = sum(u * u for u in a)
    sbb = sum(u * u for u in b)
    sab = sum(u * v for u, v in zip(a, b))
    say = sum(u * v for u, v in zip(a, ys))
    sby = sum(u * v for u, v in zip(b, ys))
    det = saa * sbb - sab * sab
    if abs(det) < 1e-9:
        return {"n": n, "testable": False}
    b1 = (sbb * say - sab * sby) / det
    b2 = (saa * sby - sab * say) / det
    resid = [v - b1 * u - b2 * w for v, u, w in zip(ys, a, b)]
    s2 = sum(r * r for r in resid) / max(n - 3, 1)
    se1 = (s2 * sbb / det) ** 0.5
    se2 = (s2 * saa / det) ** 0.5
    z1, z2 = b1 / max(se1, 1e-9), b2 / max(se2, 1e-9)
    return {"n": n, "testable": True,
            "beta1": b1, "se1": se1, "z1": z1, "p1_pos": _phi_sf(z1),
            "beta2": b2, "se2": se2, "z2": z2, "p2_pos": _phi_sf(z2)}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact one-sided McNemar p-value for paired binary outcomes:
    b = pairs where A succeeded and B failed, c = the reverse.
    H1: A better than B — p = P(X >= b), X ~ Binomial(b + c, 1/2)."""
    if b + c == 0:
        return float("nan")
    return binom_sf(b, b + c, 0.5)


# ---- gate-ready decision rules ----------------------------------------
def superiority(k1: int, n1: int, k2: int, n2: int, margin: float,
                alpha: float = ALPHA) -> tuple[bool, str]:
    """Gate rule: phat1 - phat2 >= margin AND one-sided p < alpha.
    Returns (passed, detail-string with the full math)."""
    t = two_prop_test(k1, n1, k2, n2)
    if n1 < N_MIN or n2 < N_MIN:
        return (False, f"NOT-TESTABLE: n1={n1}, n2={n2} < {N_MIN}")
    ok = (t["d"] >= margin) and (t["p"] < alpha)
    return (ok, f"Δ={t['d']:+.1%} [{t['lo']:+.1%},{t['hi']:+.1%}] "
                f"vs margin {margin:+.1%}, z={t['z']:.2f}, p={t['p']:.4f} "
                f"(n={n1}/{n2})")


def non_inferiority(k1: int, n1: int, k2: int, n2: int, margin: float,
                    alpha: float = ALPHA) -> tuple[bool, str]:
    """Gate rule: p1 is not worse than p2 by more than `margin` —
    H0: p1 <= p2 - margin, rejected when the one-sided p of
    (p1 - (p2 - margin)) > 0 is < alpha. Standard shifted z-test."""
    if n1 < N_MIN or n2 < N_MIN:
        return (False, f"NOT-TESTABLE: n1={n1}, n2={n2} < {N_MIN}")
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2 + margin
    se = math.sqrt(max(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2, 1e-12))
    z = d / se
    p = _phi_sf(z)
    ok = p < alpha
    return (ok, f"Δ={p1 - p2:+.1%}, NI margin {margin:.1%}, z={z:.2f}, "
                f"p={p:.4f} (n={n1}/{n2})")


def rate_at_least(k: int, n: int, point_min: float, ci_floor: float,
                  ) -> tuple[bool, str]:
    """Gate rule for a single proportion: point estimate >= point_min AND
    Wilson lower bound >= ci_floor."""
    if n < N_MIN:
        return (False, f"NOT-TESTABLE: n={n} < {N_MIN}")
    lo, hi = wilson_interval(k, n)
    ph = k / n
    ok = ph >= point_min and lo >= ci_floor
    return (ok, f"{ph:.1%} [{lo:.1%},{hi:.1%}] vs point>={point_min:.0%}, "
                f"CI-floor>={ci_floor:.0%} (n={n})")


def rate_above_baseline(k: int, n: int, p0: float, margin: float,
                        alpha: float = ALPHA) -> tuple[bool, str]:
    """Gate rule vs a KNOWN baseline probability (e.g. probe chance):
    phat >= p0 + margin AND exact one-sided binomial p (X >= k | p0) < alpha."""
    if n < N_MIN:
        return (False, f"NOT-TESTABLE: n={n} < {N_MIN}")
    ph = k / n
    p = binom_sf(k, n, p0)
    ok = ph >= p0 + margin and p < alpha
    return (ok, f"{ph:.1%} vs baseline {p0:.1%}+{margin:.0%}, "
                f"exact p={p:.2e} (n={n})")


def rate_at_most(k: int, n: int, point_max: float, ci_ceiling: float,
                 ) -> tuple[bool, str]:
    """Gate rule for a bounded-above proportion: point <= point_max AND
    Wilson upper bound <= ci_ceiling."""
    if n < N_MIN:
        return (False, f"NOT-TESTABLE: n={n} < {N_MIN}")
    lo, hi = wilson_interval(k, n)
    ph = k / n
    ok = ph <= point_max and hi <= ci_ceiling
    return (ok, f"{ph:.1%} [{lo:.1%},{hi:.1%}] vs point<={point_max:.0%}, "
                f"CI-ceiling<={ci_ceiling:.0%} (n={n})")
