#!/usr/bin/env python3
"""The Lovasz theta of the complement of a DIMACS graph with CSDP, and the wrapper
that attains it.

theta(Gbar) = min lambda_max(H) over the wrappers H of G (1 on the diagonal and the
edges, anything on the non-edges), the upper bound on omega(G).  Two SDPs give it; the
one with fewer constraints is solved:

  X-form   max <J,X> s.t. tr X = 1, X_ij = 0 on the non-edges of G, X psd -- what
           CSDP's theta program solves when handed the complement of G; one
           constraint per non-edge.  The wrapper comes from the dual: H = J - sum y_ij
           (E_ij + E_ji), with lambda_max(H) <= y_0 = theta.
  Y-form   min t s.t. Y = tI - H psd: Y_ii = t - 1, Y_ij = -1 on the edges of G;
           one constraint per edge.  The wrapper is H = tI - Y.

Usage: bench/theta.py [-t THREADS] [-f] [--form X|Y] GRAPH ...

GRAPH is a DIMACS name (read from $QMS_DIMACS, default ~/DIMACS) or a path.  Each
run leaves prob.dat-s, prob.sol and csdp.log in runs/theta/work/<graph>/ and appends
"graph n omega theta form constraints seconds" to runs/theta/theta.tsv; a graph
already there is skipped unless -f.  omega comes from dimacs_best.tsv.  The report
also gives the multiplicity of the wrapper's top eigenvalue and the rank of the Lovasz
optimum X (complementarity puts its range into that eigenspace, so every optimal
wrapper has at least that multiplicity), both read off at their largest relative
jump, and, when theta = omega, from clen's list of the maximum cliques, how far each
is from being anchored at C = 0 (H z_Q = omega z_Q), which Lovasz optimality forces.
CSDP is $CSDP (default ~/Csdp/solver/csdp).
"""
import argparse
import os
import re
import subprocess
import time

import numpy as np

from anchor_all import clen_max_cliques
from wrapper import read_dimacs_bin

BENCH = os.path.dirname(os.path.abspath(__file__))
CSDP = os.environ.get("CSDP", os.path.expanduser("~/Csdp/solver/csdp"))
DIMACS = os.environ.get("QMS_DIMACS", os.path.expanduser("~/DIMACS"))
RUNS = os.path.join(BENCH, "runs", "theta")


def best_omega():
    omega = {}
    for line in open(os.path.join(BENCH, "dimacs_best.tsv")):
        f = line.split()
        if len(f) >= 2 and f[1].isdigit():
            omega[f[0]] = int(f[1])
    return omega


def write_sdpa(path, adj, form):
    n = len(adj)
    iu, ju = np.triu_indices(n, 1)
    edge = adj[iu, ju]
    with open(path, "w") as f:
        if form == "X":
            ni, nj = iu[~edge] + 1, ju[~edge] + 1
            m = 1 + len(ni)
            f.write("%d\n1\n%d\n" % (m, n))
            f.write("1.0 " + "0.0 " * (m - 1) + "\n")
            for i in range(1, n + 1):                      # C = J
                f.write("".join("0 1 %d %d 1.0\n" % (i, j) for j in range(i, n + 1)))
            f.write("".join("1 1 %d %d 1.0\n" % (i, i) for i in range(1, n + 1)))
            f.write("".join("%d 1 %d %d 1.0\n" % (k + 2, i, j) for k, (i, j) in enumerate(zip(ni, nj))))
        else:
            ei, ej = iu[edge] + 1, ju[edge] + 1
            m = n - 1 + len(ei)
            f.write("%d\n1\n%d\n" % (m, n))
            f.write("0.0 " * (n - 1) + "-1.0 " * len(ei) + "\n")
            f.write("0 1 1 1 -1.0\n")                       # max -Y_11
            for v in range(2, n + 1):                        # Y_vv - Y_11 = 0
                f.write("%d 1 1 1 -1.0\n%d 1 %d %d 1.0\n" % (v - 1, v - 1, v, v))
            f.write("".join("%d 1 %d %d 0.5\n" % (n + k, i, j) for k, (i, j) in enumerate(zip(ei, ej))))
    return m


def read_solution(path, n, m):
    """y, Z and X of CSDP's solution file"""
    with open(path) as f:
        y = np.array(f.readline().split(), float)
        ZX = np.zeros((2, n, n))
        for line in f:
            mat, blk, i, j, v = line.split()
            i, j = int(i) - 1, int(j) - 1
            ZX[int(mat) - 1, i, j] = ZX[int(mat) - 1, j, i] = float(v)
    assert len(y) == m
    return y, ZX[0], ZX[1]


def cluster(values):
    """how many of the values, sorted down, come before the largest relative drop"""
    v = np.maximum(np.abs(values[:300]), 1e-300)
    return int(np.argmax(v[:-1] / v[1:])) + 1


def wrapper_from(adj, form, y, X):
    n = len(adj)
    if form == "X":
        H = np.ones((n, n))
        iu, ju = np.triu_indices(n, 1)
        non = ~adj[iu, ju]
        H[iu[non], ju[non]] -= y[1:]
        H[ju[non], iu[non]] -= y[1:]
    else:
        t = np.diag(X).mean() + 1.0
        H = t * np.eye(n) - X
        H[adj] = 1.0                                          # exact on the edges
        np.fill_diagonal(H, 1.0)
    return H


def solve(name, path, omega, threads, force, done, form=None):
    if name in done and not force:
        print("%-16s already in theta.tsv: %s" % (name, " ".join(done[name])))
        return
    adj = read_dimacs_bin(path)
    n = len(adj)
    edges = int(adj.sum()) // 2
    work = os.path.join(RUNS, "work", name if form is None else "%s.%s" % (name, form))
    if form is None:
        form = "X" if n * (n - 1) // 2 - edges + 1 <= edges + n - 1 else "Y"
    os.makedirs(work, exist_ok=True)
    prob, sol, log = (os.path.join(work, s) for s in ("prob.dat-s", "prob.sol", "csdp.log"))
    m = write_sdpa(prob, adj, form)
    print("%-16s n=%d omega=%s %s-form, %d constraints ..." % (name, n, omega, form, m), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run([CSDP, prob, sol], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=work)
    secs = time.time() - t0
    out = open(log).read()
    status = next((l for l in out.splitlines() if "Success" in l or "Failure" in l), "no status line")
    pobj = float(re.search(r"Primal objective value:\s*(\S+)", out).group(1))
    dobj = float(re.search(r"Dual objective value:\s*(\S+)", out).group(1))
    theta = (pobj + dobj) / 2 if form == "X" else 1.0 - (pobj + dobj) / 2
    y, Z, X = read_solution(sol, n, m)
    H = wrapper_from(adj, form, y, X)
    lam = np.linalg.eigvalsh(H)[::-1]
    # the Lovasz optimum X (tr X = 1, zero on the non-edges) is CSDP's X in the X-form
    # and its Z in the Y-form; complementarity puts its range in H's top eigenspace
    L = X if form == "X" else Z
    rank = cluster(np.linalg.eigvalsh(L)[::-1])
    top = 1 + cluster(1.0 / np.maximum(lam[0] - lam[1:], 1e-300))   # gaps to lambda_max, largest jump
    line = "%-16s theta %.7f  omega %s  (%s, %.0fs)  wrapper lambda_max %.7f, top cluster %d, rank of the Lovasz optimum %d" % (
        name, theta, omega, status, secs, lam[0], top, rank)
    if omega is not None and theta - omega < 1e-4 * omega:
        # Lovasz optimality forces C = 0 on every maximum clique: H 1_Q = omega 1_Q
        Qs = clen_max_cliques(path, omega)
        res = max(np.abs(H[:, Q].sum(1) - omega * np.isin(np.arange(n), Q)).max() for Q in Qs)
        span = np.linalg.matrix_rank(np.array([np.isin(np.arange(n), Q) for Q in Qs], float), tol=1e-9)
        line += "\n%-16s theta = omega: %d maximum cliques anchored at C = 0 to %.1e, their span %d" % (
            "", len(Qs), res, span)
    print(line, flush=True)
    with open(os.path.join(RUNS, "theta.tsv"), "a") as f:
        f.write("%s\t%d\t%s\t%.7f\t%s\t%d\t%.0f\n" % (name, n, omega, theta, form, m, secs))


def main():
    ap = argparse.ArgumentParser(description="theta(complement of G) with CSDP")
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("-t", "--threads", type=int, default=12)
    ap.add_argument("-f", "--force", action="store_true", help="solve again what theta.tsv has")
    ap.add_argument("--form", choices=("X", "Y"), help="the formulation, instead of the smaller one")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    tsv = os.path.join(RUNS, "theta.tsv")
    done = {l.split("\t")[0]: l.split("\t")[1:] for l in open(tsv).read().splitlines()} if os.path.exists(tsv) else {}
    omega = best_omega()
    for g in args.graphs:
        path = g if os.path.exists(g) else os.path.join(DIMACS, g + ".clq.b")
        name = os.path.basename(path).replace(".clq.b", "")
        solve(name, path, omega.get(name), args.threads, args.force, done, args.form)


if __name__ == "__main__":
    main()
