"""Deterministic logarithmic layer certificates M_c  (Paper 2 §8, Theorem 8.4) from model constants.

`M_c` must come from model constructions, never from observed samples.  Formulas (with delta=T/2):
  kbar=k+ e^{-2 Lb T}, R_uv=e^{Lb T}(|u|+B0 T)+|v|, B_v=(B0+Lb|v|)e^{Lb T},
  g_{d,r}(c)=pi^{d/2} c^{-(d+r)/2} Gamma((d+r)/2)/Gamma(d/2),
  K^E_r=(C+^2/p0~) delta^{-d/2} c_r {delta^{r/2} g_{d,r}(k+) + chi_r R_uv^r g_{d,0}(k+)},
  K^T_r=(C+^2/p0~) delta^{-d/2} c_r {g_{d,r}(kbar) + chi_r B_v^r delta^{r/2} g_{d,0}(kbar)},
  G-bound (62): |G|<=g0+g1|y|/h+g2|y|^2/h+g3|y|^3/h^2, then A_E,A_T,B_T, J_*=delta A_E+delta A_T+2 sqrt(delta) B_T,
  M_HK=2(Z_+ + J_*^2),  Z_+=(C+ T^{-d/2}/p0~) exp(-k+|theta_{0,T}(u)-v|^2/T).
Floating-point evaluation (documented weakening of 'certified outward' arithmetic); results are inflated by 1e-9.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


def g_dr(d: int, r: int, c: float) -> float:
    return math.exp(0.5 * d * math.log(math.pi) - 0.5 * (d + r) * math.log(c) + gammaln(0.5 * (d + r)) - gammaln(0.5 * d))


def p0_tilde(T: float, u, v, aT: np.ndarray) -> float:
    d = len(u)
    diff = np.asarray(v, float) - np.asarray(u, float)
    return float((2 * math.pi * T) ** (-d / 2) * np.linalg.det(aT) ** -0.5
                 * math.exp(-diff @ np.linalg.solve(aT, diff) / (2 * T)))


@dataclass
class HKConstants:
    """Assumption 8.1 data."""
    Cplus: float
    kplus: float
    Lb: float
    B0: float
    Sigma0: float
    Sigmax: float
    Sigmat: float


def ou_heat_kernel_constants(ou, T: float) -> HKConstants:
    """(70): C+=(2pi)^{-d/2}(lam e^{-2LT})^{-d/2}, k+=1/(2 lam_max e^{2LT}); L=‖B‖."""
    a = ou.a
    lam_lo, lam_hi = np.linalg.eigvalsh(a)[[0, -1]]
    L = float(np.linalg.norm(ou.B, 2))
    d = ou.d
    return HKConstants((2 * math.pi) ** (-d / 2) * (lam_lo * math.exp(-2 * L * T)) ** (-d / 2),
                       1.0 / (2 * lam_hi * math.exp(2 * L * T)), L, float(np.linalg.norm(ou.c)),
                       float(np.linalg.norm(ou.Sigma, 2)), 0.0, 0.0)


def layer_certificate(d: int, T: float, u, v, aT: np.ndarray, hk: HKConstants, theta_uT) -> dict:
    """Theorem 8.4.  `theta_uT` = theta_{0,T}(u) (deterministic drift flow at T)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    delta = T / 2
    p0 = p0_tilde(T, u, v, aT)
    kbar = hk.kplus * math.exp(-2 * hk.Lb * T)
    Ruv = math.exp(hk.Lb * T) * (np.linalg.norm(u) + hk.B0 * T) + np.linalg.norm(v)
    Bv = (hk.B0 + hk.Lb * np.linalg.norm(v)) * math.exp(hk.Lb * T)
    cr = {0: 1.0, 2: 2.0, 4: 8.0, 6: 32.0}
    chi = {0: 0.0, 2: 1.0, 4: 1.0, 6: 1.0}
    pref = hk.Cplus**2 / p0 * delta ** (-d / 2)
    KE = {r: pref * cr[r] * (delta ** (r / 2) * g_dr(d, r, hk.kplus) + chi[r] * Ruv**r * g_dr(d, 0, hk.kplus))
          for r in cr}
    KT = {r: pref * cr[r] * (g_dr(d, r, kbar) + chi[r] * Bv**r * delta ** (r / 2) * g_dr(d, 0, kbar)) for r in cr}
    At, Ax = 2 * hk.Sigma0 * hk.Sigmat, 2 * hk.Sigma0 * hk.Sigmax
    sT = float(np.linalg.norm(np.linalg.inv(aT), 2))
    Bstar = hk.B0 + hk.Lb * np.linalg.norm(v)
    g0 = 0.5 * d * sT * At
    g1 = sT * Bstar + 0.5 * d * sT * Ax
    g2 = sT * hk.Lb + 0.5 * sT**2 * At
    g3 = 0.5 * sT**2 * Ax
    AE = (g0 * math.sqrt(KE[0]) + g1 / delta * math.sqrt(KE[2]) + g2 / delta * math.sqrt(KE[4])
          + g3 / delta**2 * math.sqrt(KE[6]))
    AT = g0 * math.sqrt(KT[0]) + g2 * math.sqrt(KT[4])
    BT = g1 * math.sqrt(KT[2]) + g3 * math.sqrt(KT[6])
    Jstar = delta * AE + delta * AT + 2 * math.sqrt(delta) * BT
    Zplus = (hk.Cplus * T ** (-d / 2) / p0) * math.exp(-hk.kplus * float(np.sum((np.asarray(theta_uT) - v) ** 2)) / T)
    MHK = 2 * (Zplus + Jstar**2)
    return dict(M_c=MHK * (1 + 1e-9), Z_plus=Zplus, J_star=Jstar, p0_tilde=p0, KE=KE, KT=KT,
                g=(g0, g1, g2, g3))


def ou_endpoint_data(ou, T: float, u, v) -> dict:
    """Exact Gaussian quantities for the OU target (test/diagnostic only): Z=p(0,u;T,v)/p0~."""
    mu, Q = ou.mean_cov(T, np.asarray(u, float))
    d = ou.d
    diff = np.asarray(v, float) - mu
    p = (2 * math.pi) ** (-d / 2) * np.linalg.det(Q) ** -0.5 * math.exp(-0.5 * diff @ np.linalg.solve(Q, diff))
    return dict(p=p, Z=p / p0_tilde(T, u, v, ou.a), theta_uT=mu)


# ---------------------------------------------------------------------------------------------------------------
# Tight OU certificate (model-specific, deterministic; float quadrature)
# ---------------------------------------------------------------------------------------------------------------
def _graded_nodes(T: float, panels: int = 34, ngl: int = 8):
    """Composite Gauss-Legendre on [0,T] with geometric panels toward T: [T(1-2^-j), T(1-2^-(j+1))]."""
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    xg, wg = 0.5 * (xg + 1), 0.5 * wg
    t, w = [], []
    for j in range(panels):
        a, b = T * (1 - 2.0 ** -j), T * (1 - 2.0 ** -(j + 1))
        t.append(a + (b - a) * xg)
        w.append((b - a) * wg)
    return np.concatenate(t), np.concatenate(w)


def ou_bridge_J_second_moment(ou, T: float, u, v, panels: int = 34) -> dict:
    """E_P[J^2] and E_P[J] for the OU target bridge, J=int_0^T G(t,X_t)dt, G=-(BX+c)^T S (X-v)/(T-t).

    G is a quadratic form in the Gaussian bridge; E[G(s)G(t)] via Isserlis/Wick.  Quadrature error and the neglected
    last panel (length T 2^-panels, integrand ~ tau^{-1/2}) are NOT interval-certified."""
    from scipy.linalg import expm
    d = ou.d
    u, v = np.asarray(u, float), np.asarray(v, float)
    t, w = _graded_nodes(T, panels)
    N = len(t)
    S = np.linalg.inv(ou.a)
    M = ou.B.T @ S
    Ms = 0.5 * (M + M.T)
    wv = S @ ou.c - ou.B.T @ S @ v
    kappa = -float(ou.c @ S @ v)
    # unconditional OU moments
    par = [ou.transition_params(float(ti)) for ti in t]
    m = np.array([A @ u + m0 for A, m0, Q in par])
    Sig = np.array([Q for A, m0, Q in par])
    lam, V = np.linalg.eig(ou.B)
    Vi = np.linalg.inv(V)

    def Aexp(h):   # e^{B h} vectorised over array h
        E = np.einsum("ij,nj,jk->nik", V, np.exp(np.multiply.outer(h, lam)), Vi)
        return E.real

    AT = Aexp(T - t)                                        # e^{B(T-t)}
    _, m0T, QT = ou.transition_params(T)
    AT_full, _, _ = ou.transition_params(T)
    mT = AT_full @ u + m0T
    K_iT = np.einsum("nij,nkj->nik", Sig, AT)               # Cov(X_t,X_T)=Sig_t A(T-t)^T
    QTi = np.linalg.inv(QT)
    mu = m + np.einsum("nij,jk,k->ni", K_iT, QTi, v - mT)   # bridge mean
    # bridge covariance blocks C_ij (i<=j uses Sig_i A(t_j-t_i)^T)
    ii, jj = np.triu_indices(N)
    Aij = Aexp(t[jj] - t[ii])
    Kij = np.einsum("nab,ncb->nac", Sig[ii], Aij)           # Cov(X_i,X_j)
    Cij = Kij - np.einsum("nab,bc,ndc->nad", K_iT[ii], QTi, K_iT[jj])
    C = np.zeros((N, N, d, d))
    C[ii, jj] = Cij
    C[jj, ii] = np.transpose(Cij, (0, 2, 1))
    Cii = np.array([C[i, i] for i in range(N)])
    tau = T - t
    g = wv[None, :] + 2 * mu @ Ms.T
    h = np.einsum("ni,ij,nj->n", mu, Ms, mu) + mu @ wv + kappa
    trM = np.einsum("ab,nba->n", Ms, Cii)
    EqEq = (np.outer(h, h) + np.outer(h, trM) + np.outer(trM, h)
            + np.einsum("ia,ijab,jb->ij", g, C, g)
            + np.outer(trM, trM)
            + 2 * np.einsum("ab,ijbc,cd,ijad->ij", Ms, C, Ms, C, optimize=True))
    # note last term: tr(Ms C_ij Ms C_ji), C_ji = C_ij^T
    EG2 = EqEq / np.outer(tau, tau)
    Eq = (h + trM) / tau
    EJ = float(np.sum(w * (-Eq)))
    EJ2 = float(w @ EG2 @ w)
    return dict(EJ=EJ, EJ2=EJ2)


def ou_tight_certificate(ou, T: float, u, v, safety: float = 1.05) -> dict:
    """M_c = safety * Z (1+sqrt(E_P J^2))^2 >= E_Q[Psi (1+log+ Psi)^2]  (identity E_Q[Psi f(J)]=Z E_P f(J) and
    (1+|J|)^2, Cauchy-Schwarz).  Deterministic from model constants; numerical quadrature (not interval-certified)."""
    ed = ou_endpoint_data(ou, T, u, v)
    mm = ou_bridge_J_second_moment(ou, T, u, v)
    Mc = safety * ed["Z"] * (1 + math.sqrt(max(mm["EJ2"], 0.0))) ** 2
    return dict(M_c=Mc, Z=ed["Z"], EJ2=mm["EJ2"], EJ=mm["EJ"])
