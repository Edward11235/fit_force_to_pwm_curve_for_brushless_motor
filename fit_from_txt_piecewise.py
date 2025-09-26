import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


TXT_FILE = "16_volt_motor_parameters.txt"
KGF_TO_NEWTON = 9.80665
VERTEX = 1500.0
SEGMENT1 = (1100.0, 1500.0)
SEGMENT2 = (1500.0, 1900.0)


@dataclass
class Quadratic:
    # Model: F = a*p^2 + b*p + c
    a: float
    b: float
    c: float

    def predict(self, pwm: np.ndarray) -> np.ndarray:
        return self.a * pwm ** 2 + self.b * pwm + self.c


def load_pwm_force_from_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    # First line is header; columns are tab or spaces separated
    data = []
    for line in lines[1:]:
        parts = [p for p in line.replace("\t", " ").split(" ") if p != ""]
        if len(parts) < 6:
            continue
        try:
            pwm = float(parts[0])
            force_kgf = float(parts[5])
            data.append((pwm, force_kgf))
        except ValueError:
            continue
    arr = np.array(data, dtype=float)
    pwm = arr[:, 0]
    force_newton = arr[:, 1] * KGF_TO_NEWTON
    return pwm, force_newton


def fit_quadratic(pwm: np.ndarray, force_n: np.ndarray) -> Quadratic:
    # Fit F = a*p^2 + b*p + c via linear least squares on features [p^2, p, 1]
    X = np.vstack([pwm ** 2, pwm, np.ones_like(pwm)]).T
    params, *_ = np.linalg.lstsq(X, force_n, rcond=None)
    a, b, c = params.tolist()
    return Quadratic(a=a, b=b, c=c)


def main():
    txt_path = os.path.join(os.path.dirname(__file__), TXT_FILE)
    pwm, force_n = load_pwm_force_from_txt(txt_path)

    # Exclude the dead-zone region where force is exactly zero around 1500 µs
    # per requirement: PWM in [1470, 1530] should be removed from fitting
    exclude_deadzone = (pwm >= 1470.0) & (pwm <= 1530.0)
    pwm_fit = pwm[~exclude_deadzone]
    force_fit = force_n[~exclude_deadzone]

    mask_low = (pwm_fit >= SEGMENT1[0]) & (pwm_fit <= SEGMENT1[1])
    mask_high = (pwm_fit >= SEGMENT2[0]) & (pwm_fit <= SEGMENT2[1])

    # Create 80/20 splits per segment for evaluation
    rng = default_rng(123)
    def split(x, y):
        n = x.size
        idx = np.arange(n)
        rng.shuffle(idx)
        test_n = max(1, int(0.2 * n))
        test_idx = idx[:test_n]
        train_idx = idx[test_n:]
        return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])

    (x1_tr, y1_tr), (x1_te, y1_te) = split(pwm_fit[mask_low], force_fit[mask_low])
    (x2_tr, y2_tr), (x2_te, y2_te) = split(pwm_fit[mask_high], force_fit[mask_high])

    model_low = fit_quadratic(x1_tr, y1_tr)
    model_high = fit_quadratic(x2_tr, y2_tr)

    print("Segment 1 (1100-1500) Newton units: F = a1*p^2 + b1*p + c1")
    print(f"a1 = {model_low.a:.10g}")
    print(f"b1 = {model_low.b:.10g}")
    print(f"c1 = {model_low.c:.10g}")
    print("Segment 2 (1500-1900) Newton units: F = a2*p^2 + b2*p + c2")
    print(f"a2 = {model_high.a:.10g}")
    print(f"b2 = {model_high.b:.10g}")
    print(f"c2 = {model_high.c:.10g}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pwm_fit[mask_low], force_fit[mask_low], s=12, c="#1f77b4", label="Data 1100-1500")
    ax.scatter(pwm_fit[mask_high], force_fit[mask_high], s=12, c="#ff7f0e", label="Data 1500-1900")
    # Optionally plot excluded dead-zone points in grey for reference
    ax.scatter(pwm[exclude_deadzone], force_n[exclude_deadzone], s=10, c="#999999", label="Excluded dead-zone")

    xs1 = np.linspace(SEGMENT1[0], SEGMENT1[1], 300)
    xs2 = np.linspace(SEGMENT2[0], SEGMENT2[1], 300)
    ax.plot(xs1, model_low.predict(xs1), c="#1f77b4", lw=2, label="Fit 1100-1500")
    ax.plot(xs2, model_high.predict(xs2), c="#ff7f0e", lw=2, label="Fit 1500-1900")

    ax.axvline(VERTEX, color="#444", ls="--", lw=1, label="Vertex 1500")
    ax.set_xlabel("PWM (µs)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Piecewise Vertex-Quadratic Fit (vertex at 1500)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    out_path = os.path.join(os.path.dirname(__file__), "piecewise_fit.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved plot to {out_path}")

    # Evaluate MAE on held-out test sets
    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))
    mae1 = mae(y1_te, model_low.predict(x1_te)) if x1_te.size else float("nan")
    mae2 = mae(y2_te, model_high.predict(x2_te)) if x2_te.size else float("nan")
    total = x1_te.size + x2_te.size
    weighted = (mae1 * x1_te.size + mae2 * x2_te.size) / total if total else float("nan")
    print(f"Test MAE segment 1100-1500: {mae1:.6g} N (n={x1_te.size})")
    print(f"Test MAE segment 1500-1900: {mae2:.6g} N (n={x2_te.size})")
    print(f"Weighted average test MAE: {weighted:.6g} N")


if __name__ == "__main__":
    main()


