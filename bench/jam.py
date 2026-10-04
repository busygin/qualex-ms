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
  project  nonnegative points of the jam: the global minimizers inside R form the sphere
           {x in R: z^T x = 1, |x|^2 = 1/theta}, whose nonnegative points are the maximum
           cliques for a proper Lovasz-optimal wrapper.  From the jam vectors c +- rho d_j
           over an orthonormal basis d_j of the sphere's directions (as QUALEX-MS steps
           onto each eigenvector of a cluster), QUALEX-MS's own MIN refinement (lib/refiner.cc,
           called through qmsmin.cc) on the start itself, after alternating projections
           (clip the negatives, project back) and after Douglas-Rachford

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

Usage: bench/jam.py decode|search|project [--budget 200000] [--prune-budget 5000]
       [--decode-budget 3000] [--hard 10000] [--starts 0] [--iters 300] [-t 12] SOURCE ...

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
        work = os.path.join(RUNS, "work", name)
        X = from_sdpa(work)[2] if os.path.exists(os.path.join(work, "prob.sol")) else None
        return adj, np.ones(len(adj)), X, len(equ) // 3, equ
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


# ---------------------------------------------------------------- nonnegative points of the jam


def jam_sphere(X, w, theta):
    """the global minimizers of the relaxed program inside R: the sphere
    {x in R: z^T x = 1, |x|^2 = 1/theta}, as its centre c (the least-norm point of
    R with z^T x = 1), radius and an orthonormal basis of its directions; also the
    eigenvectors of X spanning R, in decreasing order of their eigenvalues"""
    Bm = basis(X)
    z = np.sqrt(w)
    a = Bm.T @ z
    c = Bm @ a / (a @ a)
    _, _, vt = np.linalg.svd(a[None, :])
    D = Bm @ vt[1:].T
    return c, np.sqrt(max(1.0 / theta - c @ c, 0.0)), D, Bm


def equation_jam(w, equ):
    """the jam of the equation wrapper H_A of a SAT01 instance (no SDP): the sphere
    {x in L_A: z^T x = 1, |x|^2 = 1/m}, L_A = D^1/2 {u: Au in R1}, which holds the
    indicator z_S/m of every exact cover S; returned as jam_sphere() returns its sphere"""
    n, m = len(w), len(equ)
    z = np.sqrt(w)
    A = np.zeros((m, n))
    for e, vs in enumerate(equ):
        A[e, vs] = 1
    _, s, vt = np.linalg.svd(A - w[None, :] / m)          # (I - 1 1^T/m) A
    r = int((s > 1e-10 * s[0]).sum())
    Bm, _ = np.linalg.qr(z[:, None] * vt[r:].T)
    a = Bm.T @ z
    c = Bm @ a / (a @ a)
    _, _, vt2 = np.linalg.svd(a[None, :])
    return c, np.sqrt(max(1.0 / m - c @ c, 0.0)), Bm @ vt2[1:].T, Bm


def clause_projector(adj, equ):
    """the greedy projection onto the nonnegative vectors whose support has no 2-clause
    conflict (a contradiction between variables sharing no equation): clip the
    negatives, keep entries in decreasing order unless they 2-clash with a kept one"""
    n = len(adj)
    A = np.zeros((len(equ), n))
    for e, vs in enumerate(equ):
        A[e, vs] = 1
    two = ~adj & ~np.eye(n, dtype=bool) & (A.T @ A == 0)
    nbr = [np.flatnonzero(two[i]) for i in range(n)]

    def project(y):
        out = np.zeros(n)
        blocked = np.zeros(n, bool)
        for i in np.argsort(-y):
            if y[i] <= 0:
                break
            if not blocked[i]:
                out[i] = y[i]
                blocked[nbr[i]] = True
        return out
    return project


def project_sphere(Y, c, rho, D):
    """the nearest points of the sphere to the columns of Y"""
    U = D @ (D.T @ (Y - c[:, None]))
    nu = np.linalg.norm(U, axis=0)
    nu[nu == 0] = 1.0
    return c[:, None] + rho * U / nu


def negativity(Y):
    return np.linalg.norm(np.minimum(Y, 0), axis=0) / np.maximum(np.linalg.norm(Y, axis=0), 1e-300)


class MinRefiner:
    """QUALEX-MS's MIN refinement, refine_clique_MIN_w() of lib/refiner.cc, through ctypes:
    weight(a) is the weight of the clique it builds from the appealing vector a"""
    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            import ctypes
            lib_dir = os.path.join(os.path.dirname(T.BENCH), "lib")
            so = os.path.join(RUNS, "bin", "libqmsmin.so")
            # MIN is in the library's combinatorial core, which needs no BLAS
            srcs = [os.path.join(T.BENCH, "qmsmin.cc")] + [
                os.path.join(lib_dir, f) for f in ("refiner.cc", "greedy_clique.cc", "graph.cc", "bool_vector.cc")]
            deps = srcs + [os.path.join(lib_dir, f) for f in ("refiner.h", "greedy_clique.h", "graph.h", "bool_vector.h")]
            if not os.path.exists(so) or os.path.getmtime(so) < max(os.path.getmtime(d) for d in deps):
                os.makedirs(os.path.dirname(so), exist_ok=True)
                import subprocess
                subprocess.run(["g++", "-std=gnu++20", "-O3", "-fPIC", "-shared", "-w", "-I" + lib_dir,
                                *srcs, "-o", so], check=True)
            lib = ctypes.CDLL(so)
            lib.qms_new.restype = ctypes.c_void_p
            lib.qms_new.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            lib.qms_min.restype = ctypes.c_double
            lib.qms_min.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.qms_free.argtypes = [ctypes.c_void_p]
            cls._lib = lib
        return cls._lib

    def __init__(self, adj, w):
        self.adj = np.ascontiguousarray(adj, dtype=np.uint8)
        self.w = np.ascontiguousarray(w, dtype=float)
        self.h = self.lib().qms_new(len(adj), self.adj.ctypes.data, self.w.ctypes.data)

    def weight(self, a):
        a = np.ascontiguousarray(a, dtype=float)
        return self.lib().qms_min(self.h, a.ctypes.data)

    def __del__(self):
        if getattr(self, "h", None):
            self.lib().qms_free(self.h)


def nonneg_jam(adj, w, X, target, starts, iters, tol=1e-4, jam="lovasz", proj="orthant", equ=None):
    """MIN on the jam vectors c +- rho d_j, d_j an orthonormal basis of the directions
    of the sphere (QUALEX-MS's try_eigendir_points, which hands MIN the vector x o z),
    and on the same starts after alternating projections (AP: clip the negatives,
    project back onto the sphere) or Douglas-Rachford (DR: x += P+(2 P_S x - x) - P_S x,
    read at P_S x), all starts at once, each frozen when its negativity falls below tol
    (solutions computed from CSDP's optimum carry about 1e-6 of it).  starts = 0 takes
    every direction.  jam = "equation" takes the sphere of the equation wrapper H_A
    (equation_jam(), no SDP) instead of the one in the range of the Lovasz optimum;
    proj = "clause" replaces the clipping by the greedy projection onto nonnegative
    vectors whose support has no 2-clause conflict (clause_projector()), each start
    iterated on its own and stopped when that projection moves P_S x by less than tol.
    Per method: starts whose MIN clique has weight target, starts that reached a point
    of both sets, the seconds, and (clause only) starts whose projected point is itself
    a solution"""
    z = np.sqrt(w)
    c, rho, D, Bm = equation_jam(w, equ) if jam == "equation" else jam_sphere(X, w, target)
    k = D.shape[1] if starts <= 0 else min(starts, D.shape[1])
    Y0 = np.hstack([c[:, None] + rho * D[:, :k], c[:, None] - rho * D[:, :k]]) if k else c[:, None]
    refine = MinRefiner(adj, w)
    hit = lambda Y: sum(abs(refine.weight(Y[:, j] * z) - target) < 1e-6 * max(1.0, target) for j in range(Y.shape[1]))
    out = {}
    t = time.time()
    out["jam"] = (hit(Y0), 0, time.time() - t, None)
    if proj == "clause":
        P = clause_projector(adj, equ)
        sphere = lambda y: project_sphere(y[:, None], c, rho, D)[:, 0]
        for method in ("ap", "dr"):
            t = time.time()
            hits = conv = exact = 0
            for j in range(Y0.shape[1]):
                y = Y0[:, j].copy()
                for _ in range(iters):
                    e = sphere(y)
                    pe = P(e)
                    if np.linalg.norm(pe - e) < tol * np.linalg.norm(e):
                        conv += 1
                        break
                    y = pe if method == "ap" else y + P(2 * e - y) - e
                hits += abs(refine.weight(e * z) - target) < 1e-6 * max(1.0, target)
                exact += is_target(np.flatnonzero(pe > 0), adj, w, target)
            out[method] = (hits, conv, time.time() - t, exact)
        return out, Y0.shape[1], Bm.shape[1]
    for method in ("ap", "dr"):
        t = time.time()
        Y = Y0.copy()
        est = Y0.copy()
        live = np.ones(Y.shape[1], bool)
        for _ in range(iters):
            L = np.flatnonzero(live)
            if not len(L):
                break
            if method == "ap":
                Y[:, L] = project_sphere(np.maximum(Y[:, L], 0), c, rho, D)
                est[:, L] = Y[:, L]
            else:
                ps = project_sphere(Y[:, L], c, rho, D)
                Y[:, L] = Y[:, L] + np.maximum(2 * ps - Y[:, L], 0) - ps
                est[:, L] = project_sphere(Y[:, L], c, rho, D)
            live[L[negativity(est[:, L]) < tol]] = False
        out[method] = (hit(est), int((~live).sum()), time.time() - t, None)
    return out, Y0.shape[1], Bm.shape[1]


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
    ap.add_argument("what", choices=("decode", "search", "project"))
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--budget", type=int, default=200000, help="nodes of the plain and order searches")
    ap.add_argument("--prune-budget", type=int, default=5000, help="nodes of the prune search")
    ap.add_argument("--decode-budget", type=int, default=3000, help="nodes of subspace decimation")
    ap.add_argument("--hard", type=int, default=10000,
                    help="qwhnew: compute theta only when plain search needs more nodes")
    ap.add_argument("-t", "--threads", type=int, default=12)
    ap.add_argument("--starts", type=int, default=0, help="project: jam directions, two starts each (0: all)")
    ap.add_argument("--iters", type=int, default=300, help="project: AP and DR iterations")
    ap.add_argument("--jam", choices=("lovasz", "equation"), default="lovasz",
                    help="project: the sphere in the range of the Lovasz optimum, or that of H_A")
    ap.add_argument("--proj", choices=("orthant", "clause"), default="orthant",
                    help="project: clip the negatives, or the greedy 2-clause projection")
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
        if args.what == "project":
            out, nstarts, rank = nonneg_jam(adj, w, X, target, args.starts, args.iters,
                                            jam=args.jam, proj=args.proj, equ=equ)
            for method, (hits, conv, secs, exact) in out.items():
                record(source, method + args.tag, n, target, rank, hits > 0,
                       "%d/%d conv %d%s" % (hits, nstarts, conv, "" if exact is None else " exact %d" % exact), secs)
            print("%-24s n=%-5d target %-6g rank %-4d starts %-3d | %s" % (source, n, target, rank, nstarts, " | ".join(
                "%s %d hits%s%s %.0fs" % (m, h, "" if m == "jam" else ", %d reached" % cv,
                                          "" if ex is None else ", %d exact" % ex, s)
                for m, (h, cv, s, ex) in out.items())), flush=True)
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
