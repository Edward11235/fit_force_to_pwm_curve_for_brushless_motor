"""
Utility to invert the fitted PWM->Force quadratic models into Force->PWM.

We model two segments (1100-1500) and (1500-1900) in vertex form with vertex at 1500:
  Segment 1 (low):    F = a1*(PWM-1500)^2 + d1
  Segment 2 (high):   F = a2*(PWM-1500)^2 + d2

Given a desired force F, we solve the quadratic for PWM in the appropriate
segment using the quadratic formula and pick the root that falls within the
segment bounds.

The four global parameters below should be filled with the values printed by
`fit_from_txt_piecewise.py` (in Newtons). This module accepts forces in kgf
and converts to Newtons internally before inversion.
"""

from typing import Tuple
import math


# Global coefficients (example values output by fit_from_txt_piecewise.py)
# Units: Newtons
# Segment 1 (1100-1500): F = a1*p^2 + b1*p + c1
A1 = -1.371921163e-04
B1 = 4.652741433e-01
C1 = -3.87238288e+02

# Segment 2 (1500-1900): F = a2*p^2 + b2*p + c2
A2 = 1.84721848e-04
B2 = -4.889302056e-01
C2 = 3.153301465e+02


SEGMENT1 = (1100.0, 1500.0)
SEGMENT2 = (1500.0, 1900.0)
VERTEX = 1500.0
KGF_TO_NEWTON = 9.80665
NEAR_ZERO_FORCE_N = 0.01
TXT_FILE = "16_volt_motor_parameters.txt"


def invert_vertex_quadratic(a: float, d: float, force_n: float, pwm_range: tuple) -> float:
    """Deprecated in general quadratic mode; kept for compatibility (unused)."""
    raise NotImplementedError


def invert_general_quadratic(a: float, b: float, c: float, force_n: float, pwm_range: tuple) -> float:
    # Solve a*p^2 + b*p + (c - F) = 0
    A = a
    B = b
    C = c - force_n
    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            raise ValueError("Degenerate equation")
        p = -C / B
        lo, hi = pwm_range
        if lo - 1e-6 <= p <= hi + 1e-6:
            return float(p)
        raise ValueError("Linear solution outside range")
    disc = B * B - 4.0 * A * C
    if disc < 0:
        raise ValueError("No real roots")
    sqrt_disc = math.sqrt(disc)
    p1 = (-B + sqrt_disc) / (2.0 * A)
    p2 = (-B - sqrt_disc) / (2.0 * A)
    lo, hi = pwm_range
    candidates = [p for p in (p1, p2) if lo - 1e-6 <= p <= hi + 1e-6]
    if candidates:
        # choose closest to range midpoint
        mid = 0.5 * (lo + hi)
        return float(min(candidates, key=lambda p: abs(p - mid)))
    # fallback to closest
    candidate = min((p1, p2), key=lambda p: min(abs(p - lo), abs(p - hi)))
    if lo - 5.0 <= candidate <= hi + 5.0:
        return float(candidate)
    raise ValueError("No valid root in or near range")


def force_to_pwm(force_kgf: float,
                 a1: float = A1, b1: float = B1, c1: float = C1,
                 a2: float = A2, b2: float = B2, c2: float = C2) -> float:
    """Convert desired force in kgf to PWM using piecewise general quadratic inverse."""
    force_n = float(force_kgf) * KGF_TO_NEWTON
    if -NEAR_ZERO_FORCE_N <= force_n <= NEAR_ZERO_FORCE_N:
        return VERTEX
    try:
        pwm = invert_general_quadratic(a1, b1, c1, force_n, SEGMENT1)
    except Exception:
        pwm = invert_general_quadratic(a2, b2, c2, force_n, SEGMENT2)
    # Clamp to valid range
    if pwm < SEGMENT1[0]:
        return SEGMENT1[0]
    if pwm > SEGMENT2[1]:
        return SEGMENT2[1]
    return pwm


# Standalone copy-pasteable version (no external module constants required)
# Inputs: force in kgf; returns PWM clamped to [1100,1900]
# Coefficients are embedded from current fit.
def force_to_pwm_independent(force_kgf: float) -> float:
    A1L = -1.371921163e-04
    B1L = 4.652741433e-01
    C1L = -3.87238288e+02
    A2H = 1.84721848e-04
    B2H = -4.889302056e-01
    C2H = 3.153301465e+02
    KGF_TO_N = 9.80665
    NEAR_ZERO_N = 0.01
    LO, HI = 1100.0, 1900.0
    MID = 1500.0

    def inv_quad(a: float, b: float, c: float, F: float, lo: float, hi: float) -> float:
        A = a
        B = b
        C = c - F
        if abs(A) < 1e-12:
            if abs(B) < 1e-12:
                raise ValueError
            p = -C / B
            if lo - 1e-6 <= p <= hi + 1e-6:
                return float(p)
            raise ValueError
        disc = B * B - 4.0 * A * C
        if disc < 0:
            raise ValueError
        sd = math.sqrt(disc)
        p1 = (-B + sd) / (2.0 * A)
        p2 = (-B - sd) / (2.0 * A)
        cands = [p for p in (p1, p2) if lo - 1e-6 <= p <= hi + 1e-6]
        if cands:
            mid = 0.5 * (lo + hi)
            return float(min(cands, key=lambda p: abs(p - mid)))
        cand = min((p1, p2), key=lambda p: min(abs(p - lo), abs(p - hi)))
        if lo - 5.0 <= cand <= hi + 5.0:
            return float(cand)
        raise ValueError

    F_n = float(force_kgf) * KGF_TO_N
    if -NEAR_ZERO_N <= F_n <= NEAR_ZERO_N:
        return MID
    if F_n < 0:
        pwm = inv_quad(A1L, B1L, C1L, F_n, 1100.0, 1500.0)
    elif F_n > 0:
        pwm = inv_quad(A2H, B2H, C2H, F_n, 1500.0, 1900.0)
    if pwm < LO:
        return LO
    if pwm > HI:
        return HI
    return pwm


if __name__ == "__main__":
    # Quick sanity checks and MAE over the dataset
    print("Standalone checks:")
    for f in [-2.0, -1.0, -0.05, 0.0, 0.05, 0.5, 1.0, 2.0]:
        try:
            p = force_to_pwm(f)
            print(f"force {f:6.2f} kgf -> pwm {p:8.2f}")
        except Exception as err:
            print(f"force {f:6.2f} kgf -> error: {err}")

    # Evaluate inverse MAE on the TXT file
    import os
    import numpy as np

    txt_path = os.path.join(os.path.dirname(__file__), TXT_FILE)
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = [p for p in line.replace("\t", " ").split(" ") if p]
            if len(parts) < 6:
                continue
            try:
                pwm = float(parts[0])
                force_kgf = float(parts[5])
            except ValueError:
                continue
            # Skip dead-zone rows
            if 1470.0 <= pwm <= 1530.0:
                continue
            rows.append((pwm, force_kgf))
    if not rows:
        print("No rows parsed for MAE evaluation.")
    else:
        arr = np.array(rows, dtype=float)
        pwm_true = arr[:, 0]
        force_kgf = arr[:, 1]
        pwm_pred = np.array([force_to_pwm_independent(f) for f in force_kgf], dtype=float)
        mae_pwm = float(np.mean(np.abs(pwm_true - pwm_pred)))
        print(f"Force->PWM MAE over dataset (excluding dead-zone): {mae_pwm:.6g} µs")


    force = 0.6
    pwm = force_to_pwm_independent(force)
    print(pwm)