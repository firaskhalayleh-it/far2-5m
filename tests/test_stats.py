"""The statistical gate machinery against hand-computed values — the gates
are only as honest as these functions."""

import math

from far2.eval.stats import (binom_sf, mcnemar_exact, non_inferiority,
                             rate_at_least, rate_at_most, superiority,
                             two_prop_test, wilson_interval)


def test_binom_sf_hand_values():
    assert abs(binom_sf(10, 10, 0.5) - 0.5 ** 10) < 1e-12
    assert binom_sf(0, 10, 0.5) == 1.0
    assert binom_sf(11, 10, 0.5) == 0.0
    # P(X >= 8 | n=10, p=0.5) = (45 + 10 + 1) / 1024
    assert abs(binom_sf(8, 10, 0.5) - 56 / 1024) < 1e-12
    assert binom_sf(3, 10, 0.0) == 0.0
    assert binom_sf(3, 10, 1.0) == 1.0


def test_wilson_interval_hand_values():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0
    assert abs(lo - 0.4901) < 0.01 and abs(hi - 0.9433) < 0.01
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.35
    assert math.isnan(wilson_interval(0, 0)[0])


def test_two_prop_test():
    t = two_prop_test(80, 100, 60, 100)
    assert abs(t["d"] - 0.20) < 1e-12
    assert 3.0 < t["z"] < 3.2
    assert t["p"] < 0.01
    assert t["lo"] < 0.20 < t["hi"]
    tie = two_prop_test(50, 100, 50, 100)
    assert abs(tie["p"] - 0.5) < 1e-9
    assert math.isnan(two_prop_test(1, 0, 1, 10)["p"])


def test_mcnemar_exact():
    # b=9, c=1: P(X >= 9 | n=10, p=.5) = 11/1024
    assert abs(mcnemar_exact(9, 1) - 11 / 1024) < 1e-12
    assert math.isnan(mcnemar_exact(0, 0))
    assert abs(mcnemar_exact(1, 1) - 0.75) < 1e-12


def test_gate_rules():
    ok, detail = superiority(80, 100, 60, 100, margin=0.05)
    assert ok and "p=" in detail
    # same margin but tiny n -> NOT-TESTABLE, never a pass
    ok_small, detail_small = superiority(8, 10, 2, 10, margin=0.05)
    assert not ok_small and "NOT-TESTABLE" in detail_small
    # margin met but insignificant -> fail
    ok_insig, _ = superiority(11, 20, 10, 20, margin=0.03)
    assert not ok_insig
    # z = (0.70 - 0.72 + 0.10) / sqrt(.7*.3/1000 + .72*.28/1000) ~ 3.94
    ok_ni, _ = non_inferiority(700, 1000, 720, 1000, margin=0.10)
    assert ok_ni
    # at n=100 the same margin is NOT significant (SE ~ 6.4%) -> honest fail
    ok_ni_small, _ = non_inferiority(70, 100, 72, 100, margin=0.10)
    assert not ok_ni_small
    ok_ni_bad, _ = non_inferiority(400, 1000, 720, 1000, margin=0.10)
    assert not ok_ni_bad
    ok_rl, _ = rate_at_least(30, 100, point_min=0.10, ci_floor=0.05)
    assert ok_rl
    ok_rl_bad, _ = rate_at_least(6, 100, point_min=0.10, ci_floor=0.05)
    assert not ok_rl_bad
    ok_rm, _ = rate_at_most(5, 100, point_max=0.15, ci_ceiling=0.25)
    assert ok_rm
    ok_rm_bad, _ = rate_at_most(30, 100, point_max=0.15, ci_ceiling=0.25)
    assert not ok_rm_bad
