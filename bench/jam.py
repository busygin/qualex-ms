#!/usr/bin/env python3
"""Finding a maximum weight clique inside the jammed top of a Lovasz-optimal wrapper.

When theta(Gbar, w) = omega(G, w), every feasible X of the theta program whose range lies
in the range R of a Lovasz optimum of maximal rank is optimal, and the rank-one points of
this optimal face are exactly the maximum weight cliques (reports/CliqueWrapperAnchoring,
Sections 6 and 6.5).  If i lies in a maximum clique Q, then z_Q vanishes off the closed
neighbourhood N[i], and no maximum clique avoiding i fits into N[i].  So
R_U = {x in R: x = 0 off U}, U the common closed neighbourhood of a clique C, contains
every maximum clique through C and no other.  The script reads a saved Lovasz optimum
and takes R as its eigenvectors up to the largest relative gap in its spectrum when that
gap exceeds 1e3 (clean optima, whose remaining eigenvalues are solver noise), and else
those above 1e-7 of the largest eigenvalue (jammed optima decay without a gap).  It tries

  decode   direct decoders: subspace decimation (grow C by leverage in R_U, depth first),
           the projections P_R e_i grown into greedy cliques, an l4 power iteration in R
           followed by greedy, and greedy in random orders as a baseline
  search   exact cover search, for instances whose cliques of weight theta are the exact
           covers of equations (SAT01 instances, QWH lines): Algorithm X with the equation
           of fewest candidates first and the conflicting vertices removed, in three modes,
           plain (candidates in index order), order (by leverage in R) and prune
           (candidates must keep mass in R_U, ordered by it)

R_U is always computed from the full basis: restricting the previous basis accumulates
error and was seen to prune true solutions.

Sources:
  theta:<graph>              runs/theta/work/<graph> (theta.py, X-form problems only)
  sat01:<inst>[:light|full]  runs/sat01/work/<inst>/<prep> (sat01.py)
  qwh:<name>                 runs/qwh/work/<name> (qwh.py), regenerated from qwh.py's
                             seeds for its lines and checked against the saved problem
  qwhnew:<n>,<p>,<rep>       a QWH instance generated with qwh.py's seeds; plain search
                             first, and theta with CSDP into runs/jam/work/<name> only
                             when plain search needs more than --hard nodes

Usage: bench/jam.py decode|search [--budget 200000] [--prune-budget 5000]
       [--decode-budget 3000] [--hard 10000] [-t 12] SOURCE ...

--budget bounds the plain and order searches, --prune-budget the prune search (each of
its nodes decomposes the restricted basis), --decode-budget subspace decimation.
--modes runs a subset of plain,order,prune, and --tag marks their rows (e.g. a rerun of
prune with a larger budget as prune-50k).

Rows are appended to runs/jam/jam.tsv:
  source method n target rank found nodes seconds
"""
import argparse
import os
import re
import sys
import time

import numpy as np

import qwh as Q
import sat01 as S
import theta as T

RUNS = os.path.join(T.BENCH, "runs", "jam")
TSV = os.path.join(RUNS, "jam.tsv")


class Budget(Exception):
    pass


# ---------------------------------------------------------------- loading


def from_sdpa(work):
    """an X-form problem of theta.py, qwh.py or sat01.py (objective z z^T, constraint 1
    the trace, one constraint per non-edge) and its solution: adjacency, weights, the
    Lovasz optimum and theta"""
    prob = os.path.join(work, "prob.dat-s")
    lines = open(prob).read().split("\n")
    m, n = int(lines[0]), int(lines[2])
    w = np.zeros(n)
    adj = np.ones((n, n), bool)
    np.fill_diagonal(adj, False)
    trace = 0
    for l in lines[4:]:
        f = l.split()
        if len(f) != 5:
            continue
        k, i, j = int(f[0]), int(f[2]) - 1, int(f[3]) - 1
        if k == 0 and i == j:
            w[i] = float(f[4])
        elif k == 1:
            trace += 1
        elif k >= 2:
            adj[i, j] = adj[j, i] = False
    if trace != n:
        raise SystemExit("%s is not an X-form problem" % prob)
    y, Z, X = T.read_solution(os.path.join(work, "prob.sol"), n, m)
    out = open(os.path.join(work, "csdp.log")).read()
    theta = float(re.search(r"Primal objective value:\s*(\S+)", out).group(1))
    return adj, w, X, theta


def qwh_instance(n, p, rep):
    """the QWH instance of qwh.instance(n, p, rep): its name, QCP graph and lines"""
    rng = np.random.default_rng(1000003 * n + 1009 * round(100 * p) + rep)
    L = Q.random_latin_square(n, rng, 30 * n ** 3)
    P = L.copy()
    h = round(p * n * n)
    P.flat[rng.choice(n * n, h, replace=False)] = -1
    V, adj, lines = Q.qcp_graph(P)
    return "n%d_p%02d_%d" % (n, round(100 * p), rep), adj, [cls for part in lines for cls in part]


def load(source, args):
    """adjacency of the clique graph, weights, Lovasz optimum (None if not computed),
    target weight, equations (None for graphs without them)"""
    kind, name = source.split(":", 1)
    if kind == "theta":
        adj, w, X, th = from_sdpa(os.path.join(T.BENCH, "runs", "theta", "work", name))
        return adj, w, X, T.best_omega()[name], None
    if kind == "sat01":
        inst, prep = (name.split(":") + ["full"])[:2]
        base = os.path.join(S.RUNS, "work", inst, prep, inst)
        adj, w, equ = S.load(base)
        H, X = S.dual_wrapper(base, adj, w)
        return adj, w, X, len(equ), equ
    if kind == "qwh":
        n, p, rep = re.fullmatch(r"n(\d+)_p(\d+)_(\d+)", name).groups()
        _, conflict, equ = qwh_instance(int(n), int(p) / 100, int(rep))
        adj, w, X, th = from_sdpa(os.path.join(T.BENCH, "runs", "qwh", "work", name))
        if not (adj == (~conflict & ~np.eye(len(conflict), dtype=bool))).all():
            raise SystemExit("regenerated %s differs from the saved problem" % name)
        return adj, w, X, round(th), equ
    if kind == "qwhnew":
        n, p, rep = name.split(",")
        name, conflict, equ = qwh_instance(int(n), float(p), int(rep))
        adj = ~conflict & ~np.eye(len(conflict), dtype=bool)
        return adj, np.ones(len(adj)), None, len(equ) // 3, equ
    raise SystemExit("unknown source " + source)


def lovasz_qwh(source, adj, threads):
    """theta and a Lovasz optimum of a generated QWH instance, with CSDP"""
    n, p, rep = source.split(":", 1)[1].split(",")
    name = "n%d_p%02d_%d" % (int(n), round(100 * float(p)), int(rep))
    work = os.path.join(RUNS, "work", name)
    if os.path.exists(os.path.join(work, "prob.sol")):
        a, w, X, th = from_sdpa(work)
        return th, X
    th, H, X = Q.lovasz(adj, work, threads)
    return th, X


# ---------------------------------------------------------------- subspaces


def basis(X, floor=1e-9, gap=1e3, rel=1e-7):
    """eigenvectors spanning the range of a Lovasz optimum: up to the largest relative gap
    in its spectrum above floor * the largest eigenvalue if that gap exceeds gap (a clean
    optimum, whose remaining eigenvalues are solver noise), else those above rel * the
    largest (jammed optima decay without a gap)"""
    lam, U = np.linalg.eigh(X)
    lam, U = lam[::-1], U[:, ::-1]
    top = lam[lam > floor * lam[0]]
    if len(top) > 1:
        ratio = top[:-1] / top[1:]
        k = int(np.argmax(ratio))
        if ratio[k] > gap:
            return U[:, :k + 1]
    return U[:, lam > rel * lam[0]]


def restrict(Bm, U, tol=1e-3):
    """orthonormal basis of {x in span(Bm): |x| off U <= tol |x|}, from the full basis.
    A unit Bm c has |x|^2 off U = 1 - |Bm[U] c|^2, so these are the singular directions of
    Bm[U] with s^2 > 1 - tol^2; they come from the eigendecomposition of the smaller Gram
    matrix (LAPACK's SVD failed to converge on one such matrix, eigh does not)"""
    inside = Bm[U]
    k, r = inside.shape
    if k == 0:
        return None
    if k < r:
        ev, W = np.linalg.eigh(inside @ inside.T)
        keep = ev > 1 - tol ** 2
        if not keep.any():
            return None
        V = inside.T @ W[:, keep] / np.sqrt(ev[keep])
    else:
        ev, V = np.linalg.eigh(inside.T @ inside)
        keep = ev > 1 - tol ** 2
        if not keep.any():
            return None
        V = V[:, keep]
    return Bm @ V


def is_target(C, adj, w, target):
    C = list(C)
    return (len(C) > 0 and abs(w[C].sum() - target) < 1e-6 * max(1.0, target)
            and all(adj[i, j] for a, i in enumerate(C) for j in C[a + 1:]))


def greedy(order, adj):
    C = []
    for v in order:
        if all(adj[v, u] for u in C):
            C.append(v)
    return C


# ---------------------------------------------------------------- decoders


def decimate(Bm, adj, w, target, budget, tol=1e-3):
    n = len(adj)
    idx = np.arange(n)
    nodes = [0]

    def rec(U, C):
        nodes[0] += 1
        if nodes[0] > budget:
            raise Budget
        if is_target(C, adj, w, target):
            return C
        Bc = restrict(Bm, U, tol)
        if Bc is None:
            return None
        if Bc.shape[1] == 1:
            x = Bc[:, 0] * np.sign(Bc[np.argmax(np.abs(Bc[:, 0])), 0])
            s = list(np.flatnonzero(x > 1e-2 * x.max()))
            return s if is_target(s, adj, w, target) else None
        lev = (Bc ** 2).sum(1)
        lev[C] = -1
        lev[~U] = -1
        for i in np.argsort(-lev):
            if lev[i] <= 1e-6:
                break
            r = rec(U & (adj[i] | (idx == i)), C + [int(i)])
            if r is not None:
                return r
        return None

    try:
        return rec(np.ones(n, bool), []) is not None, nodes[0]
    except Budget:
        return False, nodes[0]


def decode(adj, w, X, target, budget):
    """(method, found, tries or nodes) for the four decoders"""
    rng = np.random.default_rng(0)
    Bm = basis(X)
    out = [("decimate",) + decimate(Bm, adj, w, target, budget)]
    P = Bm @ Bm.T
    probes = np.argsort(-np.diag(P))[:50]
    hits = sum(is_target(greedy(np.argsort(-P[:, i], kind="stable"), adj), adj, w, target) for i in probes)
    out.append(("projections", hits > 0, "%d/%d" % (hits, len(probes))))
    hits = 0
    for s in range(20):
        x = Bm @ rng.standard_normal(Bm.shape[1])
        for _ in range(300):
            x = Bm @ (Bm.T @ x ** 3)
            x /= np.linalg.norm(x)
        hits += any(is_target(greedy(np.argsort(-sg * x), adj), adj, w, target) for sg in (1, -1))
    out.append(("l4", hits > 0, "%d/20" % hits))
    hits = sum(is_target(greedy(rng.permutation(len(adj)), adj), adj, w, target) for _ in range(50))
    out.append(("random-greedy", hits > 0, "%d/50" % hits))
    return out, Bm.shape[1]


# ---------------------------------------------------------------- search


def search(adj, equ, Bm, mode, budget, eps=1e-6, tol=1e-3):
    """Algorithm X over the equations with the conflicting vertices removed; returns the
    first exact cover found (or None) and the nodes"""
    n = len(adj)
    eq_of = [[] for _ in range(n)]
    for e, vs in enumerate(equ):
        for v in vs:
            eq_of[v].append(e)
    conflicts = [np.flatnonzero(~adj[v] & (np.arange(n) != v)) for v in range(n)]
    cand = {e: set(vs) for e, vs in enumerate(equ)}
    alive = np.ones(n, bool)
    static = (Bm ** 2).sum(1) if Bm is not None else None
    chosen = []
    nodes = [0]

    def rec():
        nodes[0] += 1
        if nodes[0] > budget:
            raise Budget
        if not cand:
            return list(chosen)
        if mode == "prune":
            U = alive.copy()
            U[chosen] = True
            Bc = restrict(Bm, U, tol)
            if Bc is None:
                return None
            lev = (Bc ** 2).sum(1)
            pick = min(cand, key=lambda e: sum(lev[v] > eps for v in cand[e]))
            order = sorted((v for v in cand[pick] if lev[v] > eps), key=lambda v: -lev[v])
        else:
            pick = min(cand, key=lambda e: len(cand[e]))
            order = sorted(cand[pick], key=(lambda v: -static[v]) if mode == "order" else None)
        for v in order:
            # choose v: cover its equations, remove v and every live vertex conflicting with it
            removed = [v] + [u for u in conflicts[v] if alive[u]]
            for u in removed:
                alive[u] = False
                for e in eq_of[u]:
                    if e in cand:
                        cand[e].discard(u)
            covered = [(e, cand.pop(e)) for e in eq_of[v] if e in cand]
            chosen.append(v)
            ok = all(cand[e] for e in cand)
            r = rec() if ok else None
            chosen.pop()
            for e, s in covered:
                cand[e] = s
            for u in removed:
                alive[u] = True
                for e in eq_of[u]:
                    if e in cand:
                        cand[e].add(u)
            if r is not None:
                return r
        return None

    try:
        return rec(), nodes[0]
    except Budget:
        return None, nodes[0]


# ---------------------------------------------------------------- main


def record(source, method, n, target, rank, found, nodes, secs):
    with open(TSV, "a") as f:
        f.write("%s\t%s\t%d\t%g\t%s\t%s\t%s\t%.1f\n" % (source, method, n, target, rank, found, nodes, secs))


def main():
    ap = argparse.ArgumentParser(description="maximum cliques in the jammed top of a Lovasz-optimal wrapper")
    ap.add_argument("what", choices=("decode", "search"))
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--budget", type=int, default=200000, help="nodes of the plain and order searches")
    ap.add_argument("--prune-budget", type=int, default=5000, help="nodes of the prune search")
    ap.add_argument("--decode-budget", type=int, default=3000, help="nodes of subspace decimation")
    ap.add_argument("--hard", type=int, default=10000,
                    help="qwhnew: compute theta only when plain search needs more nodes")
    ap.add_argument("-t", "--threads", type=int, default=12)
    ap.add_argument("--modes", default="plain,order,prune", help="search modes to run")
    ap.add_argument("--tag", default="", help="appended to the method names in jam.tsv")
    args = ap.parse_args()
    modes = args.modes.split(",")
    os.makedirs(RUNS, exist_ok=True)
    for source in args.sources:
        adj, w, X, target, equ = load(source, args)
        n = len(adj)
        if args.what == "decode":
            t = time.time()
            out, rank = decode(adj, w, X, target, args.decode_budget)
            for method, found, info in out:
                record(source, method, n, target, rank, found, info, time.time() - t)
            print("%-24s n=%-5d target %-6g rank %-4d | %s" % (source, n, target, rank, " | ".join(
                "%s %s %s" % (m, "yes" if f else "no", i) for m, f, i in out)), flush=True)
            continue
        if equ is None:
            raise SystemExit("%s has no equations to search over" % source)
        results = []
        hard = True
        if "plain" in modes:
            t = time.time()
            sol, nodes = search(adj, equ, None, "plain", args.budget)
            results.append(("plain", sol, nodes, time.time() - t))
            hard = sol is None or nodes > args.hard
        rank = "-"
        if X is None and hard:
            t = time.time()
            th, X = lovasz_qwh(source, adj, args.threads)
            print("%-24s theta %.6f (target %g), CSDP %.0fs" % (source, th, target, time.time() - t), flush=True)
        if X is not None:
            Bm = basis(X)
            rank = Bm.shape[1]
            for mode in ("order", "prune"):
                if mode not in modes:
                    continue
                t = time.time()
                sol, nodes = search(adj, equ, Bm, mode, args.prune_budget if mode == "prune" else args.budget)
                results.append((mode, sol, nodes, time.time() - t))
        line = []
        for mode, sol, nodes, secs in results:
            found = sol is not None and is_target(sol, adj, w, target)
            if sol is not None and not found:
                raise SystemExit("%s %s returned a non-solution" % (source, mode))
            record(source, mode + args.tag, n, target, rank, found, nodes, secs)
            line.append("%s%s %s %d nodes %.1fs" % (mode, args.tag, "found" if found else "not found", nodes, secs))
        print("%-24s n=%-5d target %-6g rank %-4s | %s" % (source, n, target, rank, " | ".join(line)), flush=True)


if __name__ == "__main__":
    main()
