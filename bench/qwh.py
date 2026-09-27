#!/usr/bin/env python3
"""Quasigroup-with-holes instances: how degenerate are their Lovasz-optimal wrappers?

A QWH instance is a random latin square of order n (Jacobson-Matthews chain) with h
cells emptied at random; it is always satisfiable.  Its QCP graph G (Busygin and
Pasechnik, reports/qcp.tex) has the candidate triples (i,j,k) as vertices and joins two
of them when they share a cell, a row and symbol, or a column and symbol; the
completions are the independent sets of size h = alpha(G) = theta(G), and cells,
row-symbol lines and column-symbol lines are three partitions of G into h cliques.
For the independent sets of G the wrapper is the clique wrapper of the complement,
and the average of the three colouring wrappers (defect h/3 on every edge of G) is
the standard wrapper up to gauge.

For each instance the script lists the completions (exact cover, capped) and reports
  span   the dimension spanned by the completion indicators
  rank   the rank of the Lovasz optimum found by CSDP, the least top multiplicity of a
         Lovasz-optimal wrapper (range 1e-6..1e-4 when there is no clear gap)
  mult   the top multiplicity of CSDP's optimal wrapper
  dimL   the dimension of L = {x : all cell, row-symbol and column-symbol sums equal},
         the top eigenspace of the standard wrapper, whose top eigenvalue is checked
         to be h (lmax) and at which every completion is anchored
  hole   |V| - h + 1, the top multiplicity of the colouring wrapper of the cells
and the exact-cover search nodes to the first completion and to all of them.

Usage: bench/qwh.py [-n 10,15] [-p 0.3,0.4,0.5,0.6,0.7] [-r 4] [--cap 20000] [--sample N]

With --sample, an instance whose listing hits the cap also gets the completions of N
randomized descents, so that span estimates the dimension of all completions rather
than of the first ones depth-first search reaches.
CSDP is $CSDP (default ~/Csdp/solver/csdp); its files go to runs/qwh/work/.
"""
import argparse
import os
import re
import subprocess
import time

import numpy as np

import theta as T

RUNS = os.path.join(T.BENCH, "runs", "qwh")


def random_latin_square(n, rng, steps):
    """Jacobson-Matthews: +-1 moves on the incidence cube, ending on a proper cube"""
    M = np.zeros((n, n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            M[i, j, (i + j) % n] = 1
    improper = None
    t = 0
    while t < steps or improper is not None:
        if improper is None:
            while True:
                r, c, s = rng.integers(n, size=3)
                if M[r, c, s] == 0:
                    break
            r1 = np.flatnonzero(M[:, c, s] == 1)[0]
            c1 = np.flatnonzero(M[r, :, s] == 1)[0]
            s1 = np.flatnonzero(M[r, c, :] == 1)[0]
        else:
            r, c, s = improper
            r1 = rng.choice(np.flatnonzero(M[:, c, s] == 1))
            c1 = rng.choice(np.flatnonzero(M[r, :, s] == 1))
            s1 = rng.choice(np.flatnonzero(M[r, c, :] == 1))
        for (a, b, d), e in (((r, c, s), 1), ((r, c1, s1), 1), ((r1, c, s1), 1), ((r1, c1, s), 1),
                             ((r, c, s1), -1), ((r, c1, s), -1), ((r1, c, s), -1), ((r1, c1, s1), -1)):
            M[a, b, d] += e
        improper = (r1, c1, s1) if M[r1, c1, s1] < 0 else None
        t += 1
    assert M.min() == 0 and (M.sum(0) == 1).all() and (M.sum(1) == 1).all() and (M.sum(2) == 1).all()
    return M.argmax(axis=2)


def qcp_graph(P):
    """vertices (i,j,k) of the holes of the partial square P (-1 = hole), their
    adjacency, and the three partitions into lines (cell, row-symbol, column-symbol)"""
    n = len(P)
    row_has = [set(P[i][P[i] >= 0]) for i in range(n)]
    col_has = [set(P[:, j][P[:, j] >= 0]) for j in range(n)]
    V = [(i, j, k) for i in range(n) for j in range(n) if P[i, j] < 0
         for k in range(n) if k not in row_has[i] and k not in col_has[j]]
    lines = []
    for key in (lambda v: (v[0], v[1]), lambda v: (v[0], v[2]), lambda v: (v[1], v[2])):
        classes = {}
        for idx, v in enumerate(V):
            classes.setdefault(key(v), []).append(idx)
        lines.append(list(classes.values()))
    adj = np.zeros((len(V), len(V)), bool)
    for part in lines:
        for cls in part:
            adj[np.ix_(cls, cls)] = True
    np.fill_diagonal(adj, False)
    return V, adj, lines


def completions(lines, nV, cap):
    """all exact covers of the 3h lines by vertices (Algorithm X), at most cap of them;
    returns them, whether the cap was hit, and the search nodes to the first and to all"""
    X = {}
    Y = {v: [] for v in range(nV)}
    for t, part in enumerate(lines):
        for c, cls in enumerate(part):
            X[(t, c)] = set(cls)
            for v in cls:
                Y[v].append((t, c))
    sols, nodes, first = [], [0], [None]
    sol = []

    def search():
        if len(sols) >= cap:
            return
        nodes[0] += 1
        if not X:
            sols.append(list(sol))
            if first[0] is None:
                first[0] = nodes[0]
            return
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            sol.append(r)
            cols = []
            for j in Y[r]:
                for i in X[j]:
                    for k in Y[i]:
                        if k != j:
                            X[k].remove(i)
                cols.append(X.pop(j))
            search()
            for j in reversed(Y[r]):
                X[j] = cols.pop()
                for i in X[j]:
                    for k in Y[i]:
                        if k != j:
                            X[k].add(i)
            sol.pop()
            if len(sols) >= cap:
                return

    search()
    return sols, len(sols) >= cap, first[0], nodes[0]


class Cutoff(Exception):
    pass


def sample_completions(lines, nV, rng, count, cutoff=None, tries=100):
    """completions found by count randomized descents (random branching order, the
    first completion of each), for a span estimate where there are too many to list.
    Randomized backtracking on these instances has heavy-tailed run times, so a
    descent that visits more than cutoff nodes (default 20 h) is restarted, at most
    tries times per sample."""
    X0 = {}
    Y = {v: [] for v in range(nV)}
    for t, part in enumerate(lines):
        for c, cls in enumerate(part):
            X0[(t, c)] = set(cls)
            for v in cls:
                Y[v].append((t, c))
    found = set()
    if cutoff is None:
        cutoff = 20 * len(lines[0])
    for _ in range(count * tries):
        if count == 0:
            break
        X = {k: set(v) for k, v in X0.items()}
        sol = []
        nodes = [0]

        def descend():
            nodes[0] += 1
            if nodes[0] > cutoff:
                raise Cutoff
            if not X:
                return True
            size = min(len(X[c]) for c in X)
            cands = [c for c in X if len(X[c]) == size]
            c = cands[rng.integers(len(cands))]
            options = list(X[c])
            rng.shuffle(options)
            for r in options:
                sol.append(r)
                cols = []
                for j in Y[r]:
                    for i in X[j]:
                        for k in Y[i]:
                            if k != j:
                                X[k].remove(i)
                    cols.append(X.pop(j))
                if descend():
                    return True
                for j in reversed(Y[r]):
                    X[j] = cols.pop()
                    for i in X[j]:
                        for k in Y[i]:
                            if k != j:
                                X[k].add(i)
                sol.pop()
            return False

        try:
            if descend():
                found.add(tuple(sorted(sol)))
                count -= 1
        except Cutoff:
            pass
    return [list(s) for s in found]


def lovasz(adjF, work, threads):
    """theta of the complement of F (= theta(G) for F the complement of G) with CSDP,
    in the X-form (one constraint per edge of G); returns theta, wrapper, optimum"""
    os.makedirs(work, exist_ok=True)
    prob, sol, log = (os.path.join(work, s) for s in ("prob.dat-s", "prob.sol", "csdp.log"))
    m = T.write_sdpa(prob, adjF, "X")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    with open(log, "w") as f:
        subprocess.run([T.CSDP, prob, sol], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=work)
    out = open(log).read()
    pobj = float(re.search(r"Primal objective value:\s*(\S+)", out).group(1))
    dobj = float(re.search(r"Dual objective value:\s*(\S+)", out).group(1))
    y, Z, X = T.read_solution(sol, len(adjF), m)
    return (pobj + dobj) / 2, T.wrapper_from(adjF, "X", y, X), X


def cluster(values):
    """how many of the values, sorted down, come before the largest relative drop"""
    v = np.maximum(np.sort(np.abs(values))[::-1], 1e-300)
    return int(np.argmax(v[:-1] / v[1:])) + 1


def rank_range(values):
    """count at the largest relative gap, and within 1e-6..1e-4 of the largest"""
    v = np.sort(np.abs(values))[::-1]
    return cluster(v), int((v > 1e-6 * v[0]).sum()), int((v > 1e-4 * v[0]).sum())


def instance(n, p, rep, cap, threads, sample=0):
    rng = np.random.default_rng(1000003 * n + 1009 * round(100 * p) + rep)
    L = random_latin_square(n, rng, 30 * n ** 3)
    P = L.copy()
    h = round(p * n * n)
    holes = rng.choice(n * n, h, replace=False)
    P.flat[holes] = -1
    V, adj, lines = qcp_graph(P)
    nV = len(V)
    assert all(len(part) == h for part in lines)
    t0 = time.time()
    sols, capped, first, nodes = completions(lines, nV, cap)
    sampled = 0
    if capped and sample:
        seen = {tuple(sorted(x)) for x in sols}
        extra = [x for x in sample_completions(lines, nV, np.random.default_rng(rep), sample)
                 if tuple(sorted(x)) not in seen]
        sampled = len(extra)
        sols = sols + extra
    t_search = time.time() - t0
    S = np.zeros((len(sols), nV))
    for a, s in enumerate(sols):
        S[a, s] = 1.0
    gram = S.T @ S
    ev = np.linalg.eigvalsh(gram) if len(sols) else np.zeros(1)
    span = int((ev > 1e-9 * max(1.0, ev.max())).sum()) if len(sols) else 0
    # L: equal sums within each of the three partitions
    rows = []
    for part in lines:
        first_cls = np.zeros(nV)
        first_cls[part[0]] = 1
        for cls in part[1:]:
            r = -first_cls.copy()
            r[cls] += 1
            rows.append(r)
    dimL = nV - int(np.linalg.matrix_rank(np.array(rows), tol=1e-9))
    # the average of the three colouring wrappers = the standard wrapper up to gauge
    J = np.ones((nV, nV))
    Havg = J - (h / 3.0) * adj
    lam = np.linalg.eigvalsh(Havg)[::-1]
    lmax = lam[0]
    multL = int((lam > lmax - 1e-8 * max(1.0, lmax)).sum())
    anch = max((np.abs(Havg[:, s].sum(1) - h * np.isin(np.arange(nV), s)).max() for s in sols[:200]),
               default=0.0)
    # Lovasz optimum with CSDP (on the complement of G, whose cliques are the completions)
    adjF = ~adj & ~np.eye(nV, dtype=bool)
    name = "n%d_p%02d_%d" % (n, round(100 * p), rep)
    t0 = time.time()
    theta, H, X = lovasz(adjF, os.path.join(RUNS, "work", name), threads)
    t_sdp = time.time() - t0
    lamH = np.linalg.eigvalsh(H)[::-1]
    mult = 1 + cluster(1.0 / np.maximum(lamH[0] - lamH[1:], 1e-300))
    rk, rk6, rk4 = rank_range(np.linalg.eigvalsh(X))
    print("%-13s h=%-4d |V|=%-5d |E|=%-6d completions %s%-6d span %-5d%s | theta %-10.6f rank %d (%d..%d) mult %d | "
          "dimL %d lmax %.6f mult %d anchored %.0e | hole %d | nodes %s/%d  [%.0fs %.0fs]" % (
              name, h, nV, int(adj.sum()) // 2, ">=" if capped else "", len(sols) - sampled, span,
              " (+%d sampled)" % sampled if sampled else "", theta,
              rk, rk4, rk6, mult, dimL, lmax, multL, anch, nV - h + 1, first, nodes, t_search, t_sdp),
          flush=True)


def main():
    ap = argparse.ArgumentParser(description="Lovasz-optimal wrappers of QWH instances")
    ap.add_argument("-n", default="10,15")
    ap.add_argument("-p", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("-r", type=int, default=4)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("-t", "--threads", type=int, default=12)
    ap.add_argument("--sample", type=int, default=0,
                    help="when the listing is capped, add completions from this many random descents")
    args = ap.parse_args()
    for n in (int(s) for s in args.n.split(",")):
        for p in (float(s) for s in args.p.split(",")):
            for rep in range(args.r):
                instance(n, p, rep, args.cap, args.threads, args.sample)


if __name__ == "__main__":
    main()
