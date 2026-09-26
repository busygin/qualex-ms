#!/usr/bin/env python3
"""Anchoring several cliques in one wrapper: numerical checks and a survey.

In tau-form a non-edge ij of the wrapper carries z_i z_j (1 - tau_ij), tau = 1 being
the standard wrapper and tau > 0 the entries below z_i z_j.  A maximal clique Q is
anchored at level gamma (Theorem 8 holds at its indicator, with mu* = gamma - w_min)
when every vertex i outside Q has

    sum over j in Q not adjacent to i of  w_j tau_ij  =  gamma.

H -> (1+t) H - t z z^T scales tau and gamma together and changes nothing else.  What
the checks confirm:

  level identity   two cliques anchored in one wrapper have
                   gamma_a W(Qb minus Qa) = gamma_b W(Qa minus Qb),
                   so all maximum cliques share one level
  collision        two cliques at one level make mu* an eigenvalue of hatA whose
                   eigenspace holds the difference of their indicators and is
                   orthogonal to hatb (the c = 0 degeneracy)
  curvature        from a clique Q anchored at gamma, hatA has Rayleigh quotient
                   mu* + gamma (W(Q') - W(Q)) / W(Q symdiff Q') along x^Q' - x^Q,
                   whatever the free entries
  holes            on odd holes and antiholes anchoring every maximum clique forces
                   gamma = 0; on even holes it forces tau = 0 at odd distances

Subcommands:

  bench/anchor_all.py checks               the above on small synthetic graphs
  bench/anchor_all.py dimacs               the DIMACS graphs with 2..20000 maximum
                                           cliques and n <= 1100
  bench/anchor_all.py random [N ...]       the unweighted random graphs of bench/ with
                                           a cliquer optimum (default n = 200)
  bench/anchor_all.py witness GRAPH OMEGA  a sparsest certificate that the maximum
                                           cliques of GRAPH force the level to 0

The survey anchors every maximum clique (listed by clen, $CLEN or ~/clen/clen) at one
level and reports: k maximum cliques, |U| their union, affdim the dimension of their
indicators' affine hull, rows(raw) the equations after dropping repeats, cols the
non-edges met, std-anchors whether the standard wrapper already anchors them all;
then either "gamma forced to 0", or, for the anchoring wrapper nearest the standard
one (level left free) and for a relative-interior point of the tau >= 0 solutions,
the multiplicity of mu* in hatA, the eigenvalues above it, |c| on its eigenspace and
the tau >= 0 margin and forced zeros.  DIMACS graphs are read from $QMS_DIMACS
(default ~/DIMACS).
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.linalg import lsmr

from wrapper import anchor, project, read_dimacs_bin, theorem8_residual, wrapper

BENCH = os.path.dirname(os.path.abspath(__file__))
CLEN = os.environ.get("CLEN", os.path.expanduser("~/clen/clen"))
DIMACS = os.environ.get("QMS_DIMACS", os.path.expanduser("~/DIMACS"))
CLEN_COUNTS = os.path.expanduser("~/clen/work/results/dimacs-maxcliques.txt")


# ---- graphs and cliques ------------------------------------------------------------

def gnp(n, p, rng):
    a = np.triu(rng.random((n, n)) < p, 1)
    return a | a.T


def cycle(n):
    adj = np.zeros((n, n), bool)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = True
    return adj


def complement(adj):
    return ~adj & ~np.eye(len(adj), dtype=bool)


def cliques(adj, w, maximal=False):
    """the maximum weight cliques (and every maximal one if asked), by Bron-Kerbosch;
    for small graphs only"""
    n = len(adj)
    nb = [frozenset(np.flatnonzero(adj[i]).tolist()) for i in range(n)]
    best = [-np.inf, []]
    found = []

    def bk(R, P, X, wR):
        if not P and not X:
            if maximal:
                found.append(np.array(sorted(R)))
            if wR > best[0] + 1e-9:
                best[0], best[1] = wR, [np.array(sorted(R))]
            elif wR > best[0] - 1e-9:
                best[1].append(np.array(sorted(R)))
            return
        if not maximal and wR + sum(w[v] for v in P) < best[0] - 1e-9:
            return
        u = max(P | X, key=lambda v: len(P & nb[v]))
        for v in list(P - nb[u]):
            bk(R | {v}, P & nb[v], X & nb[v], wR + w[v])
            P = P - {v}
            X = X | {v}

    bk(frozenset(), frozenset(range(n)), frozenset(), 0.0)
    return (best[1], found) if maximal else best[1]


def clen_max_cliques(path, omega, timeout=300):
    """every clique of omega vertices, from clen; 0-based vertex arrays"""
    fd, out = tempfile.mkstemp(suffix=".cl")
    os.close(fd)
    try:
        subprocess.run([CLEN, path, "-m%d" % omega, "-u", "-c" + out], check=True,
                       timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out) as f:
            return [np.array(sorted(int(v) - 1 for v in line.split(":")[2].split(",")))
                    for line in f if line.strip()]
    finally:
        os.remove(out)


def indicators(Qs, w):
    X = np.zeros((len(Qs), len(w)))
    for a, Q in enumerate(Qs):
        X[a, Q] = np.sqrt(w[Q]) / w[Q].sum()
    return X


def affine_dim(X):
    if len(X) < 2:
        return 0
    D = X[1:] - X[0]
    G = D.T @ D if len(D) > D.shape[1] else D
    return int(np.linalg.matrix_rank(G, tol=1e-9 * max(1.0, np.abs(G).max())))


# ---- the anchoring system ----------------------------------------------------------

def sparse_system(adj, w, Qs, common=True):
    """rows (clique a, vertex i outside it) holding w_j on the non-edges ij into the
    clique, repeats dropped; with common=False one extra column -1 per clique for its
    own level.  Returns S, the non-edge of each tau column, and the raw row count."""
    n = len(adj)
    colid = -np.ones((n, n), np.int64)
    ncol = 0
    rows, cols, vals, lev = [], [], [], []
    r0 = 0
    for a, Q in enumerate(Qs):
        inQ = np.zeros(n, bool)
        inQ[Q] = True
        out = np.flatnonzero(~inQ)
        rr, cc = np.nonzero(~adj[np.ix_(out, Q)])
        I, J = out[rr], Q[cc]
        lo, hi = np.minimum(I, J), np.maximum(I, J)
        new = colid[lo, hi] < 0
        if new.any():
            keys = np.unique(lo[new] * n + hi[new])
            colid[keys // n, keys % n] = ncol + np.arange(len(keys))
            ncol += len(keys)
        rows.append(r0 + rr)
        cols.append(colid[lo, hi])
        vals.append(w[J])
        lev.append(np.full(len(out), a))
        r0 += len(out)
    S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(r0, ncol))
    if not common:
        L = sp.csr_matrix((-np.ones(r0), (np.arange(r0), np.concatenate(lev))), shape=(r0, len(Qs)))
        S = sp.hstack([S, L]).tocsr()
    S.sum_duplicates()
    S.sort_indices()
    seen, keep = set(), []
    for r in range(S.shape[0]):
        a, b = S.indptr[r], S.indptr[r + 1]
        key = (S.indices[a:b].tobytes(), S.data[a:b].tobytes())
        if key not in seen:
            seen.add(key)
            keep.append(r)
    pairs = np.argwhere(colid >= 0)
    pairs = pairs[np.argsort(colid[pairs[:, 0], pairs[:, 1]])]
    return S[keep], pairs, r0


def build(adj, w, pairs, tau):
    """the wrapper A = H - w_min I with tau on the listed non-edges, 1 on the rest"""
    z = np.sqrt(w)
    A = wrapper(adj, w)
    A[pairs[:, 0], pairs[:, 1]] = z[pairs[:, 0]] * z[pairs[:, 1]] * (1.0 - tau)
    A[pairs[:, 1], pairs[:, 0]] = A[pairs[:, 0], pairs[:, 1]]
    return A


def perp_basis(w):
    z = np.sqrt(w)
    Q, _ = np.linalg.qr(np.column_stack([z, np.eye(len(w))]))
    return Q[:, 1:len(w)]


def spectral(A, w, X, mu):
    """hatA on the complement of z around mu: multiplicity of mu, eigenvalues above it,
    largest |c| on its eigenspace, and how far each indicator is from stationary at mu"""
    hatA, hatb = project(A, w)
    B = perp_basis(w)
    lam, V = np.linalg.eigh(B.T @ hatA @ B)
    c = V.T @ (B.T @ hatb)
    scale = max(1.0, np.abs(lam).max())
    E = np.abs(lam - mu) < 1e-7 * scale
    Y = X - np.sqrt(w) / w.sum()
    stat = np.abs((mu * Y - Y @ hatA - hatb) @ B).max()
    return dict(mult=int(E.sum()), above=int((lam > mu + 1e-7 * scale).sum()),
                cE=float(np.abs(c[E]).max()) if E.any() else 0.0, stat=stat)


def highs(c, A_eq, b_eq, bounds, A_ub=None, b_ub=None):
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                   method="highs", options=dict(time_limit=600))


def level_forced_zero(S):
    """True when S tau = 1 has no solution, i.e. only gamma = 0 anchors every clique"""
    b = np.ones(S.shape[0])
    x = lsmr(S, b, atol=1e-14, btol=1e-14, maxiter=5000)[0]
    if np.linalg.norm(S @ x - b) < 1e-8 * np.sqrt(len(b)):
        return False
    return highs(np.zeros(S.shape[1]), S, b, [(None, None)] * S.shape[1]).status == 2


def nearest(S):
    """the anchoring nearest the standard wrapper, level free: min |tau - 1| with
    S tau = gamma 1; returns (tau, gamma)"""
    std = S @ np.ones(S.shape[1])
    scale = 1e3                           # a cheap column, so gamma is barely penalized
    K = sp.hstack([S, sp.csr_matrix(-scale * np.ones((S.shape[0], 1)))]).tocsr()
    x = lsmr(K, -std, atol=1e-15, btol=1e-15, maxiter=50000)[0]
    tau, gamma = 1.0 + x[:-1], scale * x[-1]
    if gamma > 1e-3 and np.linalg.norm(S @ tau - gamma) < 1e-6 * np.sqrt(S.shape[0]):
        return tau, gamma
    x = lsmr(S, 1.0 - std, atol=1e-15, btol=1e-15, maxiter=50000)[0]
    return 1.0 + x, 1.0


def nonneg(S, max_rounds=6):
    """tau >= 0 at gamma = 1: feasible?, the largest min tau, the non-edges that are 0
    in every solution, and a point positive on all the others"""
    m = S.shape[1]
    b = np.ones(S.shape[0])
    res = highs(np.zeros(m), S, b, [(0, None)] * m)
    if res.status != 0:
        return dict(feasible=False if res.status == 2 else None)
    A_ub = sp.hstack([-sp.identity(m), sp.csr_matrix(np.ones((m, 1)))]).tocsr()
    r = highs(np.r_[np.zeros(m), -1.0], sp.hstack([S, sp.csr_matrix((S.shape[0], 1))]).tocsr(), b,
              [(0, None)] * m + [(None, 1.0)], A_ub=A_ub, b_ub=np.zeros(m))
    margin = r.x[-1] if r.status == 0 else float("nan")
    point = res.x.copy()
    free = point > 1e-9
    for _ in range(max_rounds):
        rest = np.flatnonzero(~free)
        if len(rest) == 0:
            break
        nr = len(rest)
        A_ub = sp.hstack([-sp.csr_matrix((np.ones(nr), (np.arange(nr), rest)), shape=(nr, m)),
                          sp.identity(nr)]).tocsr()
        r = highs(np.r_[np.zeros(m), -np.ones(nr)], sp.hstack([S, sp.csr_matrix((S.shape[0], nr))]).tocsr(),
                  b, [(0, None)] * m + [(0, 1)] * nr, A_ub=A_ub, b_ub=np.zeros(nr))
        if r.status != 0:
            break
        grown = r.x[:m] > 1e-9
        point = 0.5 * (point + r.x[:m])
        if not (grown & ~free).any():
            break
        free |= grown
    return dict(feasible=True, margin=margin, zeros=int((~free).sum()), point=point)


def analyse(name, adj, w, Qs, lp=True):
    """anchor all of Qs at one level and print what it takes and gives"""
    t0 = time.time()
    X = indicators(Qs, w)
    U = np.unique(np.concatenate(Qs))
    line = "%-16s n=%-4d k=%-5d |U|=%-4d affdim=%-4d" % (name, len(adj), len(Qs), len(U), affine_dim(X))
    if len(Qs) < 2:
        print(line + " (a single clique)")
        return
    S, pairs, raw = sparse_system(adj, w, Qs)
    m = S.shape[1]
    std = S @ np.ones(m)
    line += " rows=%d(%d) cols=%-6d std-anchors=%s" % (S.shape[0], raw, m, "yes" if np.ptp(std) < 1e-9 else "no")
    if level_forced_zero(S):
        print(line + " | gamma forced to 0  [%.0fs]" % (time.time() - t0))
        return
    tau, gamma = nearest(S)
    s = spectral(build(adj, w, pairs, tau), w, X, gamma - w.min())
    line += " | nearest (gamma %.3f): mult %d above %d |c_E| %.0e stat %.0e" % (
        gamma, s["mult"], s["above"], s["cE"], s["stat"])
    if lp:
        nn = nonneg(S)
        if nn["feasible"] is None:
            line += " | tau>=0: LP failed"
        elif not nn["feasible"]:
            line += " | tau>=0 infeasible"
        else:
            g = 1.37   # a generic level, so no untouched non-edge (tau = 1) sits at mu*
            s = spectral(build(adj, w, pairs, g * nn["point"]), w, X, g - w.min())
            line += " | tau>=0: margin %.3f zeros %d (%.0f%%) mult %d above %d" % (
                nn["margin"], nn["zeros"], 100.0 * nn["zeros"] / m, s["mult"], s["above"])
    print(line + "  [%.0fs]" % (time.time() - t0))


# ---- subcommands -------------------------------------------------------------------

def checks():
    rng = np.random.default_rng(20260926)

    print("level identity, pairs of maximal cliques of weighted G(40,.5):")
    worst, total, zero, agree = 0.0, 0, 0, 0
    for _ in range(30):
        adj = gnp(40, 0.5, rng)
        w = rng.integers(1, 11, 40).astype(float)
        _, maximal = cliques(adj, w, maximal=True)
        for _ in range(5):
            a, b = rng.choice(len(maximal), 2, replace=False)
            Qa, Qb = maximal[a], maximal[b]
            S, pairs, _ = sparse_system(adj, w, [Qa, Qb], common=False)
            S = S.toarray()
            m = len(pairs)
            _, s, vt = np.linalg.svd(S)
            N = vt[int((s > s.max() * max(S.shape) * 1e-12).sum()):].T
            G = N[m:, :]
            free = G.size > 0 and (np.linalg.svd(G, compute_uv=False) > 1e-9).any()
            total += 1
            Da, Db = np.setdiff1d(Qa, Qb), np.setdiff1d(Qb, Qa)
            if free:
                worst = max(worst, np.abs(w[Db].sum() * G[0] - w[Da].sum() * G[1]).max() / np.abs(G).max())
            else:
                zero += 1
            # predicted: 0 iff (an outside vertex is complete to the symmetric difference
            # and the weights differ) or a component of the non-adjacency between Qa\Qb and
            # Qb\Qa is out of balance
            sym = np.r_[Da, Db]
            outside = np.setdiff1d(np.arange(40), np.r_[Qa, Qb])
            complete = any(adj[i, sym].all() for i in outside) and abs(w[Qa].sum() - w[Qb].sum()) > 1e-9
            imbalance, seen = False, set()
            for s0 in Db:
                if s0 in seen:
                    continue
                comp, stack = {s0}, [s0]
                while stack:
                    v = stack.pop()
                    for u in (Da if v in Db else Db):
                        if u not in comp and not adj[u, v]:
                            comp.add(u)
                            stack.append(u)
                seen |= comp
                ca = w[[v for v in comp if v in Da]].sum()
                cb = w[[v for v in comp if v in Db]].sum()
                imbalance |= abs(cb * w[Da].sum() - ca * w[Db].sum()) > 1e-9
            agree += (not free) == (complete or imbalance)
    print("  %d pairs, %d force gamma = 0; identity off by at most %.0e; the zero test is right on %d"
          % (total, zero, worst, agree))

    print("collision, two maximum cliques of unweighted G(40,.5) anchored together:")
    shown = 0
    while shown < 4:
        adj = gnp(40, 0.5, rng)
        w = np.ones(40)
        Qs = cliques(adj, w)
        if len(Qs) >= 2:
            analyse("  G(40,.5)", adj, w, Qs[:2], lp=False)
            shown += 1

    print("curvature from a singly anchored clique toward the others, weighted G(40,.5):")
    worst, count = 0.0, 0
    for _ in range(10):
        adj = gnp(40, 0.5, rng)
        w = rng.integers(1, 11, 40).astype(float)
        _, maximal = cliques(adj, w, maximal=True)
        Q = maximal[rng.integers(len(maximal))]
        A = anchor(wrapper(adj, w), adj, w, Q)
        _, mu, _ = theorem8_residual(A, adj, w, Q)
        gamma = mu + w.min()
        hatA, _ = project(A, w)
        xQ = indicators([Q], w)[0]
        for Qp in maximal[:200]:
            if np.array_equal(Qp, Q):
                continue
            u = indicators([Qp], w)[0] - xQ
            pred = mu + gamma * (w[Qp].sum() - w[Q].sum()) / w[np.setxor1d(Q, Qp)].sum()
            worst = max(worst, abs(u @ hatA @ u / (u @ u) - pred) / max(1.0, abs(pred)))
            count += 1
    print("  %d cliques, largest relative error %.0e" % (count, worst))

    print("C5, one anchored edge over a grid of the free entries: never a global maximizer")
    adj, w = cycle(5), np.ones(5)
    B = perp_basis(w)
    pairs = np.array([(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)])
    gap = min(np.linalg.eigvalsh(B.T @ project(build(adj, w, pairs, np.array([1.0, s, 1 - s, 1.0, u])), w)[0] @ B)[-1]
              for s in np.linspace(-3, 4, 71) for u in np.linspace(-5, 10, 61))
    print("  smallest lambda_max(hatA) - mu* = %.4f" % gap)

    print("every maximum clique anchored, holes and antiholes:")
    for n in range(5, 11):
        analyse("  C%d" % n, cycle(n), np.ones(n), cliques(cycle(n), np.ones(n)))
    for n in (7, 9, 11):
        adj = complement(cycle(n))
        analyse("  co-C%d" % n, adj, np.ones(n), cliques(adj, np.ones(n)))

    print("random interval graphs (perfect): the colouring wrapper and the anchoring system")
    for _ in range(4):
        n = 30
        left = rng.random(n) * 10
        right = left + rng.random(n) * 2.5
        adj = (left[:, None] < right[None, :]) & (left[None, :] < right[:, None])
        np.fill_diagonal(adj, False)
        Qs = cliques(adj, np.ones(n))
        omega = len(Qs[0])
        colour = -np.ones(n, int)
        for v in np.argsort(left):
            used = set(colour[adj[v] & (colour >= 0)])
            colour[v] = min(c for c in range(n) if c not in used)
        classes = sum(np.outer(colour == c, colour == c) for c in range(colour.max() + 1)).astype(float)
        M = omega * np.eye(n) - (omega * classes - np.ones((n, n)))
        res = max(np.abs(M @ np.isin(np.arange(n), Q) - omega * np.isin(np.arange(n), Q)).max() for Q in Qs)
        print("  omega %d, %d colours: lambda_max of the colouring wrapper %.6f, anchoring residual %.0e"
              % (omega, colour.max() + 1, np.linalg.eigvalsh(M)[-1], res))
        analyse("  interval", adj, np.ones(n), Qs)


def dimacs():
    counts = {}
    if os.path.exists(CLEN_COUNTS):
        for line in open(CLEN_COUNTS):
            f = line.split()
            if len(f) == 4 and not line.startswith("#"):
                counts[f[0]] = (int(f[1]), int(f[2]))
    else:
        for line in open(os.path.join(BENCH, "dimacs_best.tsv")):
            f = line.split()
            if len(f) >= 2 and f[1].isdigit():
                counts[f[0]] = (int(f[1]), None)
    for g in sorted(counts, key=lambda g: (counts[g][1] or 0, g)):
        omega, k = counts[g]
        if k is not None and not 2 <= k <= 20000:
            continue
        path = os.path.join(DIMACS, g + ".clq.b")
        adj = read_dimacs_bin(path)
        if len(adj) > 1100:
            continue
        try:
            analyse(g, adj, np.ones(len(adj)), clen_max_cliques(path, omega))
        except subprocess.TimeoutExpired:
            print("%-16s clen timed out" % g)


def random_graphs(sizes):
    opt = {}
    for line in open(os.path.join(BENCH, "optima", "u.tsv")):
        f = line.split()
        if len(f) >= 2 and f[1].isdigit():
            opt[f[0]] = int(f[1])
    key = lambda g: tuple(int(s) for s in g[1:].split("_"))
    for g in sorted(opt, key=key):
        if key(g)[0] not in sizes:
            continue
        path = os.path.join(BENCH, "graphs", "rnd", g + ".clq.b")
        if not os.path.exists(path):
            sys.exit("%s is missing: bench/run.sh ru ... generates the random graphs" % path)
        adj = read_dimacs_bin(path)
        analyse(g, adj, np.ones(len(adj)), clen_max_cliques(path, opt[g]))


def witness(path, omega):
    """the sparsest y with y^T S = 0 and sum y = 1 over the rows of S: a certificate
    that every anchoring of all maximum cliques has gamma = 0"""
    adj = read_dimacs_bin(path)
    w = np.ones(len(adj))
    Qs = clen_max_cliques(path, omega)
    S, pairs, _ = sparse_system(adj, w, Qs)
    if not level_forced_zero(S):
        print("the maximum cliques can be anchored at a nonzero level")
        return
    # keep, for each deduplicated row, which (clique, vertex) produced it
    origin = {}
    for a, Q in enumerate(Qs):
        for i in np.setdiff1d(np.arange(len(adj)), Q):
            trace = tuple(Q[~adj[i, Q]])
            origin.setdefault((i, trace), a)
    m, r = S.shape[1], S.shape[0]
    A_eq = sp.vstack([sp.hstack([S.T, -S.T]), sp.csr_matrix(np.r_[np.ones(r), -np.ones(r)])]).tocsr()
    res = highs(np.ones(2 * r), A_eq, np.r_[np.zeros(m), 1.0], [(0, None)] * (2 * r))
    y = res.x[:r] - res.x[r:]
    used = np.flatnonzero(np.abs(y) > 1e-9)
    print("%d maximum cliques; the certificate combines %d equations:" % (len(Qs), len(used)))
    for t in used:
        cols = S.indices[S.indptr[t]:S.indptr[t + 1]]
        edges = pairs[cols]
        i = np.bincount(edges.ravel()).argmax() if len(edges) > 1 else None
        if i is None:   # a single non-edge: either end may be the outside vertex
            i = next(v for v in edges[0] if any(v not in Q and set(edges[0]) - {v} <= set(Q) for Q in Qs))
        trace = sorted(int(v) for v in edges.ravel() if v != i)
        a = origin.get((i, tuple(trace)))
        print("  %+g x [vertex %d outside clique %s: its non-neighbours there %s]"
              % (y[t], i + 1, "?" if a is None else a + 1, [v + 1 for v in trace]))
    for a, Q in enumerate(Qs[:12]):
        print("  clique %d: %s" % (a + 1, " ".join(str(v + 1) for v in Q)))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "checks":
        checks()
    elif what == "dimacs":
        dimacs()
    elif what == "random":
        random_graphs(tuple(int(s) for s in sys.argv[2:]) or (200,))
    elif what == "witness" and len(sys.argv) == 4:
        witness(sys.argv[2], int(sys.argv[3]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
