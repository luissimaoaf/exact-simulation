"""Finite controlled-Picard block certificate (Paper 1 Appendix C.1-C.2, Theorem 6.2 'CPVerify').

`verify_block` tests, on one block of length ell and for driver amplitudes q1=[U]_alpha ell^alpha,
q2=[U]_{2alpha} ell^{2alpha}, that the controlled-Picard map

    P_{z,U}(Y)_t = z + int f(s,Y_s) ds + int V(s,Y_s) dU_s ,   P(Y)' = V(Y)

leaves the component ball (C.1) invariant (strict inequalities m_i > Phi_i(m)) and is a strict contraction in the
weighted max norm of the 4x4 nonnegative difference matrix L (weights (1,4(1+K1),4(1+K1),1)), then propagates the
two-driver difference ("driver ball" forcing (C.12)-(C.13), incoming radius r_in) to the outgoing radius

        d0 <= ||f_forcing||_w / (1 - lambda_w).                                   (31)

Everything is evaluated in double precision from *supplied* sup-norm coefficient bounds (K_r for D^rV, r<=3,
theta_r time-Lipschitz moduli, F0,F1 for the drift f).  Floating point replaces outward rounding: documented weakening.
The sewing constant is S_alpha = 1/(1-2^{1-3 alpha}) (eq. (29)); it requires alpha>1/3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


class CertificateFailure(RuntimeError):
    """A finite certificate could not be produced with the given search budget (never a wrong answer)."""


def sewing_constant(alpha: float) -> float:
    assert alpha > 1 / 3
    return 1.0 / (1.0 - 2.0 ** (1 - 3 * alpha))


@dataclass
class Bounds:
    """Sup-norm bounds valid on the state box of a block."""
    K: tuple[float, float, float, float]          # K0..K3 for D^r V
    th: tuple[float, float, float] = (0.0, 0.0, 0.0)   # theta_0..2 (time-Lipschitz moduli of D^rV); 0 if autonomous
    F0: float = 0.0                                # sup |f|
    F1: float = 0.0                                # sup ||Df||


@dataclass
class BlockResult:
    ok: bool
    m: tuple = ()
    lam: float = math.inf
    radius: float = math.inf                       # outgoing/within-block difference radius d0
    lam_matrix: np.ndarray | None = None
    reason: str = ""


def _phi(m, K, th, F0, ell, q1, q2, Sa, rin):
    K0, K1, K2, K3 = K
    m0, m1, m2, m3 = m
    aY = m1 * q1 + m3
    h2 = K2 * aY * m1 + K1 * m2 + th[1] * ell * m1
    rH = K1 * m3 + 0.5 * K2 * aY**2 + th[0] * ell
    D = rH * q1 + h2 * q2
    Phi0 = rin + F0 * ell + K0 * q1 + K1 * m1 * q2 + Sa * D
    Phi1 = K0
    Phi2 = K1 * aY + th[0] * ell
    Phi3 = F0 * ell + K1 * m1 * q2 + Sa * D
    return np.array([Phi0, Phi1, Phi2, Phi3]), aY, h2, rH


def difference_matrix(m, K, th, F1, ell, q1, q2, Sa):
    """4x4 nonnegative matrix L of (C.9)-(C.11) (columns = d0,d1,d2,d3)."""
    K0, K1, K2, K3 = K
    m0, m1, m2, m3 = m
    aY = m1 * q1 + m3
    L = np.zeros((4, 4))
    for j in range(4):
        d = np.zeros(4)
        d[j] = 1.0
        d0, d1, d2, d3 = d
        aD = q1 * d1 + d3
        u0 = K1 * d0
        u2 = (2 * K2 * aY + th[1] * ell) * d0 + K1 * aD
        v0 = K2 * m1 * d0 + K1 * d1
        v2 = ((2 * K3 * aY * m1 + K2 * m2 + th[2] * ell * m1) * d0 + 2 * K2 * m1 * aD
              + (K2 * aY + th[1] * ell) * d1 + K1 * d2)
        rD = (K2 * m3 + K3 * aY**2 + th[1] * ell) * d0 + 2 * K2 * aY * aD + K1 * d3
        DD = rD * q1 + v2 * q2
        L[0, j] = F1 * ell * d0 + u0 * q1 + v0 * q2 + Sa * DD
        L[1, j] = u0
        L[2, j] = u2
        L[3, j] = F1 * ell * d0 + v0 * q2 + Sa * DD
    return L


def verify_block(bounds_fn, ell: float, alpha: float, q1: float, q2: float, r_in: float, delta: float,
                 Sa: float | None = None, max_tries: int = 40) -> BlockResult:
    """Search rational-like candidates (C.15) until invariance + contraction hold; return the difference radius.

    bounds_fn(state_radius) -> Bounds valid on the box of that radius around the block's centre state.
    delta = certified driver-ball radius rho_alpha(U, U~) <= delta, so eps1=delta ell^alpha, eps2=delta^2 ell^{2 alpha}."""
    Sa = sewing_constant(alpha) if Sa is None else Sa
    eps1, eps2 = delta * ell**alpha, delta**2 * ell ** (2 * alpha)
    e0 = q1 + q2 + q1**2 + ell
    e3 = q2 + q1**2 + ell
    C0 = C2 = C3 = 2.0
    for _ in range(max_tries):
        # candidate radii; the box radius is r_in + m0, so iterate the coefficient bounds once or twice
        m0 = r_in + C0 * e0
        for _it in range(3):
            b = bounds_fn(m0)
            m = (r_in + C0 * e0, b.K[0] + 1.0, C2 * e0, C3 * e3)
            if m[0] <= m0 * (1 + 1e-12):
                break
            m0 = m[0]
        b = bounds_fn(m[0])
        Phi, aY, h2, rH = _phi(m, b.K, b.th, b.F0, ell, q1, q2, Sa, r_in)
        if np.all(np.array(m) > Phi):
            L = difference_matrix(m, b.K, b.th, b.F1, ell, q1, q2, Sa)
            w = np.array([1.0, 4 * (1 + b.K[1]), 4 * (1 + b.K[1]), 1.0])
            lam = float(np.max((L @ w) / w))
            if lam < 1.0:
                K0, K1, K2, K3 = b.K
                m0_, m1, m2, m3 = m
                D_drv = (2 * K2 * aY * m1 * q1 + 2 * K2 * m1**2 * q2 + rH) * eps1 + h2 * eps2
                f = np.array([r_in + K0 * eps1 + K1 * m1 * eps2 + Sa * D_drv, 0.0,
                              K1 * m1 * eps1, K1 * m1 * eps2 + Sa * D_drv])
                radius = float(np.max(f / w)) / (1.0 - lam)
                return BlockResult(True, m, lam, radius, L)
            # contraction fails -> constants are fine but ell too large
            return BlockResult(False, m, lam, math.inf, L, "contraction lambda_w>=1")
        # enlarge the constant that violates
        viol = np.array(m) <= Phi
        if viol[3]:
            C3 *= 2.0
        if viol[2]:
            C2 *= 2.0
        if viol[0] or viol[1]:
            C0 *= 2.0
        if max(C0, C2, C3) > 1e12:
            break
    return BlockResult(False, reason="invariance not attained")
