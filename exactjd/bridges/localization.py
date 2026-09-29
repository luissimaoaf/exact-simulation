"""Paper 1 §9: open-domain localization (Algorithm 2) with one retained rough driver.

Effective open-domain data: an exhaustion  K_n ⊂ U_n ⋐ int K_{n+1} ⊂ D  of the open domain D by boxes (here rectangles),
with coefficient bounds supplied per shell by the model (`coeff_bounds` on the shell box).  A shell is *admissible for a
candidate* only if (i) the whole candidate enclosure (centre path ± radius) lies strictly inside U_n and (ii) all physical
and auxiliary state boxes used by the verifier lie in U_n.  Failing shells are skipped; **the driver is never resampled**
and no containment event is conditioned on (retrospective, not a stopping time of the diffusion).

`shell_search` dovetails (shell n, driver level r, cutoff q) over a finite budget and returns the first admissible triple
together with the shell index; the caller then runs `RoughGuidedBridge.tube/tail` with the shell's bounds.  The verifier
itself is the same as in the global case (it only uses bounds on its specified boxes), hence the certified pipeline
inherits the same feasibility limits (see docs/CHECKLIST.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..rng import Budget, BudgetExhausted
from .cpverify import CertificateFailure
from .rough_guided import RoughGuidedBridge, RoughEnclosure


@dataclass
class BoxExhaustion:
    """K_n = [lo_n, hi_n] (componentwise), strictly nested with margins; U_n = K_n shrunk by `margin` fraction of the gap."""

    los: list[np.ndarray]
    his: list[np.ndarray]

    @classmethod
    def heston(cls, p0: float, z0: float, n: int = 12):
        """D = R x (0,inf):  K_j = [p0-2^j, p0+2^j] x [z0 2^{-j-1}... , z0 + 2^{j}] (dist to z=0 and |p| grow with j)."""
        los = [np.array([p0 - 2.0**j, z0 * 2.0 ** (-j) * 0.5]) for j in range(n)]
        his = [np.array([p0 + 2.0**j, z0 + 2.0**j]) for j in range(n)]
        return cls(los, his)

    def inside(self, X: np.ndarray, radius: float, j: int) -> bool:
        lo, hi = self.los[j], self.his[j]
        return bool(np.all(X.min(axis=0) - radius > lo) and np.all(X.max(axis=0) + radius < hi))

    def box(self, j: int):
        lo, hi = self.los[j], self.his[j]
        return 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo))


class LocalizedGuidedBridge(RoughGuidedBridge):
    """Guided bridge in an open domain D with retrospective shell selection (one retained driver)."""

    def __init__(self, model, exhaustion: BoxExhaustion, *a, **kw):
        super().__init__(model, *a, **kw)
        self.exh = exhaustion

    def center_checked(self, n: int, q: int, sub: int = 2):
        """Centre path; raises CertificateFailure if it leaves D (the candidate is then rejected, not repaired)."""
        try:
            with np.errstate(invalid="raise", divide="raise"):
                return self.center(n, q, sub)
        except (FloatingPointError, ValueError) as e:
            raise CertificateFailure(f"centre path left the domain (numerical failure: {e})")

    def shell_search(self, levels=range(5, 11), max_shell: int | None = None, diagnostic_radius: float | None = None,
                     budget: Budget | None = None) -> dict:
        """Dovetail over (driver level r, cutoff q=r-2, shell j): first candidate whose centre ± radius lies in U_j.

        radius = certified E_pre-style radius if available else `diagnostic_radius` (UNCERTIFIED, flagged)."""
        max_shell = len(self.exh.los) if max_shell is None else max_shell
        tried = []
        for r in levels:
            q = max(2, r - 2)
            try:
                ts, xs = self.center_checked(r, q)
            except CertificateFailure as e:
                tried.append((r, None, str(e)))
                continue
            rad = diagnostic_radius if diagnostic_radius is not None else self.driver(r)["delta"]
            for j in range(max_shell):
                if self.exh.inside(np.vstack([xs, self.v[None, :]]), rad, j):
                    return dict(level=r, cutoff=q, shell=j, radius_used=rad,
                                status="diagnostic-UNCERTIFIED" if diagnostic_radius is not None else "certified-radius",
                                center_range=(xs.min(0).tolist(), xs.max(0).tolist()), tried=tried)
            tried.append((r, None, "no shell contains centre ± radius"))
        raise BudgetExhausted(f"shell search exhausted levels {list(levels)}; attempts: {tried[-3:]}")
