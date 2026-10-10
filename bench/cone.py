#!/usr/bin/env python3
"""The intersection of the sphere of a SAT01 instance's equation wrapper with the surface
of its standard wrapper, in closed form (reports/CliqueWrapperAnchoring, Section 6.7).

The sphere is Sigma_A = {c + Q p: |p| = rho} of the equation wrapper H_A (jam.py's
equation_jam()), and the standard wrapper H_0 has z_i z_j on the edges and the diagonal
and 0 on the contradictions.  On the sphere x^T H_0 x - 1 is a quadric
p^T M p + 2 g^T p + h; in the eigenvectors V of M (eigenvalues l, u = V^T p, gamma =
V^T g) it equals, for a root sigma of

    F(sigma) = sum_i gamma_i^2/(l_i - sigma) - h - sigma rho^2,

the cone (u - v)^T (diag(l) - sigma I)(u - v) with vertex v_i = gamma_i/(sigma - l_i).
Its generators w_i = xi_i/sqrt(l_i - sigma) (l_i > sigma), eta_i/sqrt(sigma - l_i)
(l_i < sigma), |xi| = |eta| = 1, meet the sphere at v + t w, t a root of
|w|^2 t^2 + 2 v^T w t + |v|^2 - rho^2.  So two unit vectors and a sign give every point
of the intersection, and when |v| < rho the positive root alone gives each point once.

For each instance (after full propagation, runs/sat01/work/<inst>/full, as sat01.py
prepares it) the script reports the roots sigma found in the gaps of the spectrum of
M, the root whose vertex is nearest the centre, the share of random generators that
meet the sphere, and, around the generator of a solution (exact cover search), the
largest error of the formula and the negative part of the intersection points.

Usage: bench/cone.py [--samples 100000] [--seed 1] INSTANCE ...
"""
import argparse
import os

import numpy as np

import jam as J
import sat01 as S


def roots(lam, gamma, h, rho):
    """the roots of F in the gaps between consecutive distinct eigenvalues where F
    changes sign, by bisection"""
    F = lambda s: np.sum(gamma**2 / (lam - s)) - h - s * rho**2
    ev = np.unique(np.round(lam, 10))
    out = []
    for a, b in zip(ev[:-1], ev[1:]):
        lo, hi = a + (b - a) * 1e-9, b - (b - a) * 1e-9
        if not F(lo) < 0 < F(hi):
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if F(mid) < 0 else (lo, mid)
        out.append(0.5 * (lo + hi))
    return out, len(ev) - 1


def generators(a, xi, eta):
    """the generators w of the cone for the unit vectors xi (on l > sigma) and eta
    (on l < sigma), one per column"""
    P, N = a > 0, a < 0
    w = np.zeros((len(a), xi.shape[1]))
    w[P] = xi / np.sqrt(a[P])[:, None]
    w[N] = eta / np.sqrt(-a[N])[:, None]
    return w


def meet(v, w, rho):
    """the points v + t w on the sphere |u| = rho, both roots, for the columns of w that
    meet it"""
    B, A2, C = v @ w, (w * w).sum(0), v @ v - rho**2
    disc = B * B - A2 * C
    ok = disc >= 0
    pts = [v[:, None] + ((-B[ok] + sg * np.sqrt(disc[ok])) / A2[ok]) * w[:, ok] for sg in (1, -1)]
    return np.hstack(pts), ok


def unit(rng, k, n):
    x = rng.standard_normal((k, n))
    return x / np.linalg.norm(x, axis=0)


def main():
    ap = argparse.ArgumentParser(description="the sphere of H_A and the surface of H_0 in closed form")
    ap.add_argument("instances", nargs="+")
    ap.add_argument("--samples", type=int, default=100000, help="random generators tried")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    for inst in args.instances:
        rng = np.random.default_rng(args.seed)
        base = os.path.join(S.RUNS, "work", inst, "full", inst)
        adj, w, equ = S.load(base)
        n, m = len(w), len(equ)
        z = np.sqrt(w)
        c, rho, Q, _ = J.equation_jam(w, equ)
        k = Q.shape[1]
        H0 = np.outer(z, z) * (adj | np.eye(n, dtype=bool))
        lam, V = np.linalg.eigh(Q.T @ H0 @ Q)
        gamma = V.T @ (Q.T @ H0 @ c)
        h = c @ H0 @ c - 1.0
        rs, gaps = roots(lam, gamma, h, rho)
        print("%s: n=%d m=%d, the sphere has %d directions, rho=%.4g; F has a root in %d of %d gaps"
              % (inst, n, m, k, rho, len(rs), gaps))
        if not rs:
            continue
        sigma = min(rs, key=lambda s: np.linalg.norm(gamma / (s - lam)))
        v = gamma / (sigma - lam)
        a = lam - sigma
        P, N = a > 0, a < 0
        print("  vertex nearest the centre: sigma=%.6g, |v|/rho=%.4f, %d directions above sigma, %d below;"
              " %d roots with the vertex inside the ball"
              % (sigma, np.linalg.norm(v) / rho, P.sum(), N.sum(),
                 sum(np.linalg.norm(gamma / (s - lam)) < rho for s in rs)))
        hit = 0
        for b0 in range(0, args.samples, 10000):
            nb = min(10000, args.samples - b0)
            _, ok = meet(v, generators(a, unit(rng, P.sum(), nb), unit(rng, N.sum(), nb)), rho)
            hit += ok.sum()
        print("  random generators meeting the sphere: %d of %d" % (hit, args.samples))

        sols, _ = S.solutions(adj, equ, 1)
        if not sols:
            continue
        K = sols[0]
        xs = np.zeros(n)
        xs[K] = z[K] / m
        u = V.T @ (Q.T @ (xs - c))
        d = u - v
        sp, sn = np.linalg.norm(np.sqrt(a[P]) * d[P]), np.linalg.norm(np.sqrt(-a[N]) * d[N])
        print("  a solution (%d variables): |u|/rho=%.9f, x^T H_0 x=%.9f, on the cone: %.9g = %.9g"
              % (len(K), np.linalg.norm(u) / rho, xs @ H0 @ xs, sp, sn))
        xi0, eta0 = np.sqrt(a[P]) * d[P] / sp, np.sqrt(-a[N]) * d[N] / sn
        pts, _ = meet(v, generators(a, xi0[:, None], eta0[:, None]), rho)
        print("  its generator meets the sphere at the solution (distance %.1e) and %.4g rho from it"
              % tuple(sorted(np.linalg.norm(pts - u[:, None], axis=0) / np.array([1.0, rho]))))
        for eps in (1e-3, 1e-2, 1e-1):
            xi = xi0[:, None] + eps * rng.standard_normal((P.sum(), 1000))
            eta = eta0[:, None] + eps * rng.standard_normal((N.sum(), 1000))
            pts, ok = meet(v, generators(a, xi / np.linalg.norm(xi, axis=0), eta / np.linalg.norm(eta, axis=0)), rho)
            if not ok.any():
                print("  generators perturbed by %.0e: none of 1000 meets the sphere" % eps)
                continue
            X = c[:, None] + Q @ (V @ pts)
            err = max(np.abs((pts * pts).sum(0) / rho**2 - 1).max(),
                      np.abs(np.einsum("ij,ij->j", X, H0 @ X) - 1).max(), np.abs(z @ X - 1).max())
            neg = np.median(-np.minimum(X, 0).sum(0) / np.abs(X).sum(0))
            print("  generators perturbed by %.0e: %d of 1000 meet the sphere, largest error %.1e,"
                  " median negative share of the l1 mass %.3f" % (eps, ok.sum(), err, neg))


if __name__ == "__main__":
    main()
