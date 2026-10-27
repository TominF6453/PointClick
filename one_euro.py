"""One Euro filter: an adaptive low-pass filter for noisy interactive input.

Casiez, Roussel and Vogel, "1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
Interactive Systems", CHI 2012. Smooths heavily when the input moves slowly (removing jitter) and
lightly when it moves fast (reducing lag).
"""

import math


def smoothing_factor(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters one scalar value. The parameters can be changed between calls for live tuning."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev = 0.0

    def __call__(self, x: float, t: float) -> float:
        """Filter value x sampled at time t (seconds)."""
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x

        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev

        # Smoothed speed sets the cutoff: faster movement means less smoothing.
        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + smoothing_factor(self.d_cutoff, dt) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self.x_prev + smoothing_factor(cutoff, dt) * (x - self.x_prev)

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat
