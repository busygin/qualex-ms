#!/usr/bin/env python3
"""The smallest jam of a satisfiable SAT01 instance (reports/CliqueWrapperAnchoring,
Section 6.8): the subspace R* spanned by all Lovasz optima, which the top eigenspace of
no Lovasz-optimal wrapping matrix can undercut, with a certificate of its minimality.

When theta = m the equation wrapper H_A is Lovasz-optimal, so every optimum is
X = B Y B^T, B an orthonormal basis of its top eigenspace L_A, with Y >= 0, tr Y = 1 and
<S_ij, Y> = 0 on every contradiction, S_ij = (b_i b_j^T + b_j b_i^T)/2 for the rows b_i
of B: the optima form a spectrahedron F in the symmetric d x d matrices, cut by the
linear space V of those Y.  A PSD W orthogonal to V has every optimum in its kernel, so
a Y in F and such a W of complementary ranks prove R* = B range(Y) minimal.

The steps, on an instance after full propagation (runs/sat01/work/<inst>/full, as
sat01.py builds it):
  structure  a sparse rational basis N of {u: Au in R1} modulo P = 2^61 - 1 (in it the
             constraints read n_i^T Y n_j = 0, the factor D^1/2 dropping out) and the
             exact rank of the constraint rows by srank.cc: dim V, and the set G' of
             contradictions whose constraints are independent and imply the others
  select     a G' of the same exact rank that is also well conditioned: a pivoted
             Cholesky factorization of the Gram matrix of G' and further pairs, with
             entries (Pi_ik Pi_jl + Pi_il Pi_jk)/2, Pi = B B^T; needed on factoring
             instances, where the eliminated G' can be nearly dependent
  cond       how well conditioned the eliminated G' is as it stands: an unpivoted
             Cholesky factorization of its Gram matrix in packed storage
  ipm        the pair (max t: Y - tI >= 0, <S_ij,Y> = 0 on G', tr Y = 1) and (min y0:
             W = sum y_ij S_ij + y0 I >= 0, tr W = 1), both strictly feasible, by a
             primal-dual interior point method (HKM direction, Mehrotra's predictor-
             corrector, infeasible start); the Schur matrix has order |G'| + 1 and,
             the S_ij having rank two, entries made of those of B Z B^T and
             B W^-1 B^T; --packed keeps it in rectangular full packed storage
  dense      the same pair over a basis of V from a dense eigendecomposition of the
             Gram matrix of the constraints, d(d+1)/2 unknowns (small d only)
  save       R* (orthonormal, in R^n), the optimum B Y B^T and the certificate
             B W B^T into runs/sat01/work/<inst>/full/<inst>.jam.npz; the cut is the
             largest gap in the spectrum of Z = Y - tI unless --dim gives it, and
             the eigenvalues of Z and the weights u^T W u of its eigenvectors are
             printed around it
and the experiments of the section:
  dr         Douglas-Rachford from the 2(k-1) corners c +- rho d_j of the sphere
             {x in R: z^T x = 1, |x|^2 = 1/m}, R = L_A and R = R*, towards the orthant
             or the greedy 2-clause projection (clause.cc), as lib/dr.cc iterates;
             MIN (jam.py's MinRefiner) on the last point of every start
  bm         the factored search Y = C C^T / |C|^2 of Burer and Monteiro, minimizing
             sum <S_ij, Y>^2 by L-BFGS: decode (checkpoints, MIN on the top
             eigenvectors), starts (more seeds), combine (their mean), floor
             (Y >= delta I, the residual as a would-be certificate)

Everything in jam coordinates goes through one basis B per instance, cached: the basis
an SVD returns for a null space depends on the BLAS thread count.  Intermediate files go
to the ignored runs/sat01/work/minjam/; the logs of the report's runs are in
runs/sat01/minjam/.

Usage: bench/minjam.py structure f60491 cubic20_1
       bench/minjam.py select f60491; bench/minjam.py ipm f60491; bench/minjam.py save f60491
       bench/minjam.py ipm --packed cubic30_1
       bench/minjam.py bm decode f14471 --checkpoints 200,400,800,1600
"""
import argparse
import ctypes
import json
import os
import struct
import subprocess
import time

import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
from scipy.linalg import lapack

import jam as J
import sat01 as S

P = (1 << 61) - 1
BENCH = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(S.RUNS, "work", "minjam")
BIN = os.path.join(S.RUNS, "bin")


# ---------------------------------------------------------------- common


def load(inst):
    """the instance after full propagation: adjacency (non-contradiction), weights, equations"""
    return S.load(os.path.join(S.RUNS, "work", inst, "full", inst))


def work(inst, suffix):
    os.makedirs(WORK, exist_ok=True)
    return os.path.join(WORK, inst + suffix)


def jam_basis(inst, w, equ):
    """an orthonormal basis B of L_A, computed once per instance and kept, so that saved
    jam coordinates stay valid whatever the thread count of the BLAS"""
    f = work(inst, ".B.npy")
    if os.path.exists(f):
        return np.load(f)
    _, _, _, B = J.equation_jam(w, equ)
    np.save(f, B)
    return B


def contradictions(adj):
    """the contradiction pairs i < j, as two index arrays"""
    n = len(adj)
    return np.nonzero(np.triu(~adj & ~np.eye(n, dtype=bool), 1))


def solutions(inst, adj, equ, cap=200):
    """the solutions by sat01.py's exact cover search, up to cap, kept as a list"""
    f = work(inst, ".sols.json")
    if not os.path.exists(f):
        sols, _ = S.solutions(adj, equ, cap)
        with open(f, "w") as out:
            json.dump([sorted(int(v) for v in K) for K in sols], out)
    with open(f) as inp:
        return json.load(inp)


def indicators(sols, z, m):
    """the indicators x^S = z_S / m of the solutions"""
    for K in sols:
        x = np.zeros(len(z))
        x[K] = z[K] / m
        yield x


def build(target, sources, flags):
    """compile bench sources into runs/sat01/bin unless the target is newer"""
    out = os.path.join(BIN, target)
    srcs = [os.path.join(BENCH, s) for s in sources]
    if not os.path.exists(out) or os.path.getmtime(out) < max(os.path.getmtime(s) for s in srcs):
        os.makedirs(BIN, exist_ok=True)
        subprocess.run(["g++", "-std=gnu++20", "-O3", "-march=native", *flags, *srcs, "-o", out], check=True)
    return out


def srank(inst, pairs, out):
    """srank on the pairs (two index arrays) in the instance's sparse basis: the exact rank,
    and the pivot rows written to out"""
    with open(work(inst, ".sub.bin"), "wb") as f:
        f.write(struct.pack("<i", len(pairs[0])))
        f.write(np.ascontiguousarray(np.stack(pairs, 1).astype("<i4")).tobytes())
    p = subprocess.run([build("srank", ["srank.cc"], []), work(inst, ".basis.bin"), work(inst, ".sub.bin"), out],
                       capture_output=True, text=True, check=True)
    return int(p.stdout.split()[1]), p.stdout.strip()


def independent(inst, prefer_selected=True):
    """the indices into contradictions() of G': select's set if there is one, else structure's"""
    for suffix in ((".indep2.bin", ".indep.bin") if prefer_selected else (".indep.bin",)):
        f = work(inst, suffix)
        if os.path.exists(f):
            raw = np.fromfile(f, dtype="<i4")
            return np.sort(raw[1:1 + raw[0]]), suffix
    raise SystemExit("%s: run structure first" % inst)


def gram(Pi, i, j, rows):
    """rows of the Gram matrix <S_ij, S_kl> = (Pi_ik Pi_jl + Pi_il Pi_jk)/2 of the pairs (i, j)"""
    return 0.5 * (Pi[i[rows]][:, i] * Pi[j[rows]][:, j] + Pi[i[rows]][:, j] * Pi[j[rows]][:, i])


# ---------------------------------------------------------------- structure


def sparse_basis(n, equ):
    """a basis of N = {u: A u in R1} modulo P: Gaussian elimination of [A | 1] with a
    Markowitz pivot choice (fewest (row length - 1)(column length - 1)), reduced to echelon
    form, then one null vector e_f - sum_p R_pf e_p per free column f and the particular
    solution u0 (free variables 0) as the last column; returned as its rows n_i, each a list
    of (column, value), with rank A"""
    m = len(equ)
    RHS = n
    rows = [{**{int(v): 1 for v in vs}, RHS: 1} for vs in equ]
    colrows = {}
    for r, row in enumerate(rows):
        for c in row:
            colrows.setdefault(c, set()).add(r)
    live = set(range(m))
    pivots = {}
    order = []
    while True:
        best = None
        for r in live:
            row = rows[r]
            lr = len(row) - (RHS in row)
            if lr == 0:
                continue
            for c in row:
                if c == RHS:
                    continue
                cost = (lr - 1) * (len(colrows[c]) - 1)
                if best is None or cost < best[0]:
                    best = (cost, r, c)
                    if cost == 0:
                        break
            if best is not None and best[0] == 0:
                break
        if best is None:
            break
        _, r, c = best
        inv = pow(rows[r][c], P - 2, P)
        prow = {k: v * inv % P for k, v in rows[r].items()}
        rows[r] = prow
        live.discard(r)
        for s in list(colrows[c]):
            if s == r:
                continue
            row = rows[s]
            f = row[c]
            for k, v in prow.items():
                nv = (row.get(k, 0) - f * v) % P
                if nv:
                    if k not in row:
                        colrows.setdefault(k, set()).add(s)
                    row[k] = nv
                elif k in row:
                    del row[k]
                    colrows[k].discard(s)
        for k in prow:
            colrows[k].discard(r)
        pivots[c] = r
        order.append(c)
    # back substitution: each pivot row keeps no other pivot column
    for c in reversed(order):
        prow = rows[pivots[c]]
        for c2 in order:
            if c2 == c:
                continue
            row = rows[pivots[c2]]
            if c in row:
                f = row[c]
                for k, v in prow.items():
                    nv = (row.get(k, 0) - f * v) % P
                    if nv:
                        row[k] = nv
                    elif k in row:
                        del row[k]
    for r in live:
        assert rows[r].get(RHS, 0) == 0 and all(k == RHS for k in rows[r]), "A u = 1 is inconsistent"
    free = [c for c in range(n) if c not in pivots]
    d = len(free) + 1
    fidx = {f: a for a, f in enumerate(free)}
    basis = [[] for _ in range(n)]
    for f in free:
        basis[f].append((fidx[f], 1))
    for c, r in pivots.items():
        assert rows[r][c] == 1 and all(k == c or k == RHS or k in fidx for k in rows[r]), "not reduced"
        for k, v in rows[r].items():
            if k == RHS:
                basis[c].append((d - 1, v))
            elif k != c:
                basis[c].append((fidx[k], (P - v) % P))
    return basis, d, len(pivots)


def cmd_structure(args):
    for inst in args.instances:
        adj, w, equ = load(inst)
        n, m = len(w), len(equ)
        t0 = time.time()
        basis, d, rankA = sparse_basis(n, equ)
        B = jam_basis(inst, w, equ)
        assert B.shape[1] == d, "the sparse basis and L_A differ in dimension"
        with open(work(inst, ".basis.bin"), "wb") as f:
            f.write(struct.pack("<ii", n, d))
            for row in basis:
                f.write(struct.pack("<i", len(row)))
                for c, v in row:
                    f.write(struct.pack("<iQ", c, v))
        nnz = np.array([len(row) for row in basis])
        I, Jc = contradictions(adj)
        t1 = time.time()
        rank, out = srank(inst, (I, Jc), work(inst, ".indep.bin"))
        N = d * (d + 1) // 2
        print("%s: n=%d m=%d rank A=%d, dim L_A=%d; rows of N: %.1f nonzeros on average, at most %d (%.1fs)"
              % (inst, n, m, rankA, d, nnz.mean(), nnz.max(), t1 - t0))
        print("  %d contradictions, %d independent (%.0f%%); dim V = %d of %d (%.0f%%); srank %.1fs"
              % (len(I), rank, 100.0 * rank / len(I), N - rank, N, 100.0 * (N - rank) / N, time.time() - t1), flush=True)


# ---------------------------------------------------------------- the set G'


def cmd_select(args):
    for inst in args.instances:
        adj, w, equ = load(inst)
        B = jam_basis(inst, w, equ)
        Pi = B @ B.T
        I, Jc = contradictions(adj)
        sel, _ = independent(inst, prefer_selected=False)
        rank = len(sel)
        rest = np.setdiff1d(np.arange(len(I)), sel)
        extra = min(args.extra, len(rest))
        cand = np.concatenate([sel, np.random.default_rng(1).choice(rest, extra, replace=False)])
        q = len(cand)
        t0 = time.time()
        K = np.empty((q, q), order="F")
        i, j = I[cand], Jc[cand]
        for a in range(0, q, 2000):
            K[a:a + 2000] = gram(Pi, i, j, slice(a, a + 2000))
        print("%s: Gram matrix of %d candidates (%d independent + %d further pairs), %.1f GB, %.0fs"
              % (inst, q, rank, extra, q * q * 8 / 1e9, time.time() - t0), flush=True)
        diag0 = K.diagonal().max()
        c, piv, r, info = lapack.dpstrf(K, tol=-1.0, lower=1, overwrite_a=1)
        piv = piv - 1
        res = np.abs(np.diagonal(c))[:min(r, rank)] ** 2 / diag0
        print("  pivoted Cholesky %.0fs: numerical rank %d; residual at pivot %d %.2e of the largest diagonal"
              % (time.time() - t0, r, min(r, rank), res[-1]), flush=True)
        del K, c
        new = np.sort(cand[piv[:rank]])
        er, _ = srank(inst, (I[new], Jc[new]), work(inst, ".subindep.bin"))
        print("  exact rank of the %d chosen pairs: %d (needs %d)%s"
              % (len(new), er, rank, "; written" if er == rank else "; not written"), flush=True)
        if er == rank:
            with open(work(inst, ".indep2.bin"), "wb") as f:
                f.write(struct.pack("<i", len(new)))
                f.write(np.ascontiguousarray(new.astype("<i4")).tobytes())


def rfp_put(AR, k, j, col):
    """column j (from the diagonal down) of a lower triangle into RFP storage (TRANSR N,
    UPLO L, even order 2k): A[r, c] is AR[r + 1, c] for c < k and AR[c - k, r - k] else"""
    if j < k:
        AR[j + 1:, j] = col
    else:
        AR[j - k, j - k:] = col


def rfp_diag(AR, k, j):
    return (j + 1, j) if j < k else (j - k, j - k)


def cmd_cond(args):
    for inst in args.instances:
        adj, w, equ = load(inst)
        B = jam_basis(inst, w, equ)
        Pi = B @ B.T
        I, Jc = contradictions(adj)
        sel, _ = independent(inst, prefer_selected=False)
        i, j = I[sel], Jc[sel]
        p = len(sel)
        N = p + (p % 2)
        k = N // 2
        t0 = time.time()
        AR = np.zeros((N + 1, k), order="F")
        diag = np.empty(p)
        for s0 in range(0, p, 1000):
            rows = 0.5 * (Pi[i[s0:s0 + 1000]][:, i[s0:]] * Pi[j[s0:s0 + 1000]][:, j[s0:]]
                          + Pi[i[s0:s0 + 1000]][:, j[s0:]] * Pi[j[s0:s0 + 1000]][:, i[s0:]])
            for c in range(s0, min(p, s0 + 1000)):
                col = rows[c - s0, c - s0:]
                if N > p:
                    col = np.append(col, 0.0)
                diag[c] = col[0]
                rfp_put(AR, k, c, col)
        if N > p:
            AR[rfp_diag(AR, k, N - 1)] = 1.0
        print("%s: Gram matrix of the %d eliminated pairs in packed storage, %.1f GB, %.0fs"
              % (inst, p, AR.nbytes / 1e9, time.time() - t0), flush=True)
        cf, info = lapack.dpftrf(N, AR.ravel(order="F"), transr="N", uplo="L", overwrite_a=1)
        print("  Cholesky %.0fs, info %d" % (time.time() - t0, info), flush=True)
        if info == 0:
            C = cf.reshape((N + 1, k), order="F")
            L = np.array([C[rfp_diag(C, k, c)] for c in range(p)])
            r = L**2 / diag.max()
            o = np.argsort(r)
            print("  squared pivots / largest diagonal entry: smallest %s; below 1e-8/1e-6/1e-4: %d/%d/%d; median %.2e"
                  % (" ".join("%.1e" % x for x in r[o[:6]]), (r < 1e-8).sum(), (r < 1e-6).sum(), (r < 1e-4).sum(),
                     np.median(r)), flush=True)


# ---------------------------------------------------------------- the interior point method


def maxstep(M, D):
    """the largest step a with M + a D positive semidefinite"""
    Li = np.linalg.inv(np.linalg.cholesky(M))
    lam = np.linalg.eigvalsh(Li @ D @ Li.T).min()
    return np.inf if lam >= 0 else -1.0 / lam


def report(inst, Z, W, t, y0, B, adj, w, equ):
    """the spectra of Z = Y - tI and W, the largest gap of Z, the solutions against it,
    and the optimum on every contradiction; the result goes to <inst>.ipm.npz"""
    n, m = len(w), len(equ)
    d = B.shape[1]
    ez, Uz = np.linalg.eigh(Z)
    ez, Uz = ez[::-1], Uz[:, ::-1]
    ew = np.linalg.eigvalsh(W)[::-1]
    cuts = (1e-4, 1e-6, 1e-8, 1e-10)
    print("t = %.3e, y0 = %.3e (0 when no optimum is positive definite)" % (t, y0))
    print("Z: eigenvalues/largest above 1e-4/6/8/10: %s" % "/".join(str(int((ez > c * ez[0]).sum())) for c in cuts))
    print("W: eigenvalues/largest above 1e-4/6/8/10: %s" % "/".join(str(int((ew > c * ew[0]).sum())) for c in cuts))
    gz = int(np.argmax(np.log(np.maximum(ez[:-1], 1e-300)) - np.log(np.maximum(ez[1:], 1e-300)))) + 1
    print("largest gap in Z after %d: %.2e -> %.2e; W around %d: %s"
          % (gz, ez[gz - 1], ez[gz], d - gz, " ".join("%.2e" % e for e in ew[max(0, d - gz - 2):d - gz + 2])))
    face = Uz[:, :gz]
    for x in indicators(solutions(inst, adj, equ), np.sqrt(w), m):
        c = B.T @ x
        print("a solution: share outside the face %.1e; c^T W c / |c|^2 = %.1e"
              % (np.linalg.norm(c - face @ (face.T @ c)) / np.linalg.norm(c), c @ W @ c / (c @ c)))
    Y = Z + t * np.eye(d)
    X = B @ (Y / np.trace(Y)) @ B.T
    contr = ~adj & ~np.eye(n, dtype=bool)
    print("the optimum on all %d contradictions: max |X_ij| %.1e" % (contr.sum() // 2, np.abs(X[contr]).max()), flush=True)
    np.savez(work(inst, ".ipm.npz"), Z=Z, W=W, t=t, y0=y0)


def cmd_ipm(args):
    inst = args.instance
    adj, w, equ = load(inst)
    n = len(w)
    B = jam_basis(inst, w, equ)
    d = B.shape[1]
    I, Jc = contradictions(adj)
    nc = len(I)
    sel, which = independent(inst)
    I, Jc = I[sel], Jc[sel]
    p = len(I) + 1                                 # the pairs of G', then the trace
    Pi = B @ B.T
    h = np.concatenate([Pi[I, Jc], [float(d)]])    # <A_k, I>
    a = np.zeros(p)
    a[-1] = 1.0
    NR = p + (p % 2)                               # packed storage wants an even order: a dummy
    KR = NR // 2
    print("%s: d=%d, %d independent contradiction constraints (%s, of %d), Schur %d x %d (%.1f GB%s)"
          % (inst, d, len(I), "selected" if which == ".indep2.bin" else "as eliminated", nc, p, p,
             (p * (p + 1) * 4 if args.packed else p * p * 8) / 1e9, " packed" if args.packed else ""), flush=True)

    def opA(M):
        # <A_k, M> for a d x d M: (B M_sym B^T)_ij, and tr M for the last
        Ms = B @ (0.5 * (M + M.T)) @ B.T
        return np.concatenate([Ms[I, Jc], [np.trace(M)]])

    def opAt(y):
        # sum_k y_k A_k = B^T M B + y0 I, M with y_ij / 2 at (i,j) and (j,i)
        Mm = np.zeros((n, n))
        np.add.at(Mm, (I, Jc), 0.5 * y[:-1])
        return B.T @ (Mm + Mm.T) @ B + y[-1] * np.eye(d)

    def schur_rows(Zh, Wh):
        # the blocks of rows s:e, columns s: of the Schur matrix among the pairs of G'
        # (its lower triangle, read by columns), from Zh = B Z B^T and Wh = B W^-1 B^T
        ZA, ZB, WA, WB = Zh[:, I], Zh[:, Jc], Wh[:, I], Wh[:, Jc]
        blk = 1000 if args.packed else 4000
        for s0 in range(0, p - 1, blk):
            e = min(p - 1, s0 + blk)
            Ib, Jb = I[s0:e], Jc[s0:e]
            yield s0, e, 0.25 * (ZA[Ib, s0:] * WB[Jb, s0:] + ZB[Ib, s0:] * WA[Jb, s0:]
                                 + ZA[Jb, s0:] * WB[Ib, s0:] + ZB[Jb, s0:] * WA[Ib, s0:])

    def trace_row(Z, Wi):
        # the row of the trace constraint: <I, Z S_ab W^-1> = ((B W^-1 Z B^T)_ba + (..)_ab) / 2
        ZWh = B @ (Wi @ Z) @ B.T
        return 0.5 * (ZWh[Jc, I] + ZWh[I, Jc])

    def factor_full(Z, Wi):
        O = np.empty((p, p), order="F")            # Fortran order: the Cholesky works in place
        for s0, e, rows in schur_rows(B @ Z @ B.T, B @ Wi @ B.T):
            O[s0:p - 1, s0:e] = rows.T
        O[-1, :p - 1] = trace_row(Z, Wi)
        O[-1, -1] = np.trace(Z @ Wi)
        O[np.diag_indices(p)] += 1e-14 * np.trace(O) / p
        cf = sl.cholesky(O, lower=True, overwrite_a=True, check_finite=False)
        return lambda v: sl.cho_solve((cf, True), v, check_finite=False)

    def factor_packed(Z, Wi):
        AR = np.zeros((NR + 1, KR), order="F")
        diag_sum = 0.0
        tr_row = trace_row(Z, Wi)
        for s0, e, rows in schur_rows(B @ Z @ B.T, B @ Wi @ B.T):
            for j in range(s0, e):
                col = np.empty(NR - j)
                col[:p - 1 - j] = rows[j - s0, j - s0:]
                col[p - 1 - j] = tr_row[j]
                if NR > p:
                    col[-1] = 0.0
                diag_sum += col[0]
                rfp_put(AR, KR, j, col)
        last = np.array([np.trace(Z @ Wi)] + ([0.0] if NR > p else []))
        diag_sum += last[0]
        rfp_put(AR, KR, p - 1, last)
        if NR > p:
            rfp_put(AR, KR, p, np.array([1.0]))
        dl = 1e-14 * diag_sum / p
        for j in range(p):
            AR[rfp_diag(AR, KR, j)] += dl
        cf, info = lapack.dpftrf(NR, AR.ravel(order="F"), transr="N", uplo="L", overwrite_a=1)
        if info != 0:
            raise np.linalg.LinAlgError("dpftrf info %d" % info)

        def solve(v):
            bpad = np.zeros((NR, 1))
            bpad[:p, 0] = v
            x, _ = lapack.dpftrs(NR, cf, bpad, transr="N", uplo="L")
            return x[:p, 0]
        return solve

    Z = np.eye(d)                                  # Y = Z + t I = I/d, with room inside the cone
    t = 1.0 / d - 1.0
    y = np.zeros(p)
    y[-1] = 1.0 / d
    W = opAt(y)
    t0 = time.time()
    ap_last = ad_last = 0.0
    for it in range(100):
        rp = a - opA(Z) - h * t
        Rd = opAt(y) - W
        rt = 1.0 - h @ y
        mu = np.vdot(Z, W) / d
        pinf = np.linalg.norm(rp) / (1 + np.linalg.norm(a))
        dinf = (np.linalg.norm(Rd) + abs(rt)) / (1 + np.sqrt(d))
        print("  it %3d %7.1fs: t %.6e y0 %.6e <Z,W> %.1e pinf %.1e dinf %.1e  steps %.2f %.2f"
              % (it, time.time() - t0, t, y[-1], np.vdot(Z, W), pinf, dinf, ap_last, ad_last), flush=True)
        np.savez(work(inst, ".ipm_state.npz"), Z=Z, W=W, t=t, y=y, it=it)
        # the Schur solve's accuracy floors the primal residual near 1e-10 on the largest instances
        if pinf < 1e-9 and dinf < 1e-10 and np.vdot(Z, W) < 1e-11:
            break
        try:
            Wi = np.linalg.inv(W)
            Wi = 0.5 * (Wi + Wi.T)
            solveO = (factor_packed if args.packed else factor_full)(Z, Wi)
            Oh = solveO(h)

            def direction(Rc):
                # dZ W + Z dW = Rc,  dW = Rd + A*(dy),  A(dZ) + h dt = rp,  h^T dy = rt
                Oq = solveO(opA(Rc @ Wi - Z @ Rd @ Wi) - rp)
                dt = (rt - h @ Oq) / (h @ Oh)
                dy = Oq + Oh * dt
                dW = Rd + opAt(dy)
                dZ = (Rc - Z @ dW) @ Wi
                return 0.5 * (dZ + dZ.T), dt, dy, dW
            dZa, _, _, dWa = direction(-Z @ W)
            ap, ad = min(1.0, maxstep(Z, dZa)), min(1.0, maxstep(W, dWa))
            sigma = min(1.0, (np.vdot(Z + ap * dZa, W + ad * dWa) / d / mu) ** 3)
            dZ, dt, dy, dW = direction(sigma * mu * np.eye(d) - Z @ W - dZa @ dWa)
            ap, ad = min(1.0, 0.95 * maxstep(Z, dZ)), min(1.0, 0.95 * maxstep(W, dW))
            ap_last, ad_last = ap, ad
            Z = Z + ap * dZ
            Z = 0.5 * (Z + Z.T)
            t = t + ap * dt
            y = y + ad * dy
            W = W + ad * dW
            W = 0.5 * (W + W.T)
            solveO = None                          # frees the factor before the next one
        except np.linalg.LinAlgError as e:
            print("  it %3d: %s, keeping the last iterate" % (it, e), flush=True)
            break
    report(inst, Z, W, t, y[-1], B, adj, w, equ)


def cmd_dense(args):
    inst = args.instance
    adj, w, equ = load(inst)
    B = jam_basis(inst, w, equ)
    d = B.shape[1]
    I, Jc = contradictions(adj)
    iu = np.triu_indices(d)
    scale = np.where(iu[0] == iu[1], 0.5, 1.0 / np.sqrt(2.0))
    N = len(iu[0])
    t0 = time.time()
    G = np.zeros((N, N))
    for s in range(0, len(I), 1000):
        O = B[I[s:s + 1000]][:, :, None] * B[Jc[s:s + 1000]][:, None, :]
        rows = (O + O.transpose(0, 2, 1))[:, iu[0], iu[1]] * scale
        G += rows.T @ rows
    del O, rows
    print("%s: d=%d, %d contradictions, %d unknowns; Gram matrix %.1fs" % (inst, d, len(I), N, time.time() - t0), flush=True)
    t0 = time.time()
    ev, U = sl.eigh(G, driver="evd", overwrite_a=True)
    k = int((ev < args.cut * ev[-1]).sum())
    print("eigh %.1fs; null space dimension %d at %.0e; eigenvalues/largest around the cut: %s | %s"
          % (time.time() - t0, k, args.cut, " ".join("%.1e" % (e / ev[-1]) for e in ev[max(0, k - 3):k]),
             " ".join("%.1e" % (e / ev[-1]) for e in ev[k:k + 3])), flush=True)
    V = np.ascontiguousarray(U[:, :k])
    del G, U
    # the pair over V: max t s.t. Z = Y0 + sum y_l V0_l - t I >= 0 (V0 the trace-free part of
    # V, Y0 its least-norm point of trace 1) and the certificate W as the primal variable
    sc = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))

    def svec(M):
        return M[iu] * sc

    def smat(v):
        M = np.zeros((d, d))
        M[iu] = v / sc
        return M + np.triu(M, 1).T
    tau = svec(np.eye(d))
    tv = V.T @ tau
    Y0 = V @ tv / (tv @ tv)
    q, _ = np.linalg.qr(np.hstack([tv[:, None], np.eye(k)[:, :k - 1]]))
    Amat = np.hstack([V @ q[:, 1:], tau[:, None]])
    b = np.zeros(k)
    b[-1] = 1.0
    C = smat(Y0)
    Atens = np.stack([smat(Amat[:, i]) for i in range(k)])
    opA = lambda M: Amat.T @ svec(0.5 * (M + M.T))
    opAt = lambda y: smat(Amat @ y)
    print("%s: dim V=%d, %d constraints" % (inst, k, k), flush=True)

    def step(X, y, Z, rp, Rd, mu):
        Zi = np.linalg.inv(Z)
        Zi = 0.5 * (Zi + Zi.T)
        # the Schur matrix O_ij = <A_i, X A_j Z^-1> = <L^T A_i R, L^T A_j R>, X = L L^T, Z^-1 = R R^T
        Lx, Rz = np.linalg.cholesky(X), np.linalg.cholesky(Zi)
        Mt = np.einsum("ab,ibc,cd->iad", Lx.T, Atens, Rz, optimize=True).reshape(k, d * d)
        O = Mt @ Mt.T
        cf = np.linalg.cholesky(O + 1e-14 * np.trace(O) / k * np.eye(k))

        def solve(rc):
            dy = np.linalg.solve(cf.T, np.linalg.solve(cf, rp - opA(rc @ Zi) + opA(X @ Rd @ Zi)))
            dZ = Rd - opAt(dy)
            dX = (rc - X @ dZ) @ Zi
            return 0.5 * (dX + dX.T), dy, dZ
        dXa, _, dZa = solve(-X @ Z)
        ap, ad = min(1.0, maxstep(X, dXa)), min(1.0, maxstep(Z, dZa))
        sigma = min(1.0, (np.vdot(X + ap * dXa, Z + ad * dZa) / d / mu) ** 3)
        dX, dy, dZ = solve(sigma * mu * np.eye(d) - X @ Z - dXa @ dZa)
        ap, ad = min(1.0, 0.95 * maxstep(X, dX)), min(1.0, 0.95 * maxstep(Z, dZ))
        X = X + ap * dX
        Z = Z + ad * dZ
        return 0.5 * (X + X.T), y + ad * dy, 0.5 * (Z + Z.T)

    X = np.eye(d) / d
    y = np.zeros(k)
    y[-1] = np.linalg.eigvalsh(C).min() - 1.0
    Z = C - opAt(y)
    t0 = time.time()
    for it in range(200):
        rp = b - opA(X)
        Rd = C - opAt(y) - Z
        pinf, dinf = np.linalg.norm(rp) / (1 + np.linalg.norm(b)), np.linalg.norm(Rd) / (1 + np.linalg.norm(C))
        stop = pinf < 1e-10 and dinf < 1e-10 and np.vdot(X, Z) < 1e-11
        if it % 5 == 0 or stop:
            print("  it %3d  %.1fs: primal %.6e dual %.6e  <X,Z> %.1e  pinf %.1e dinf %.1e"
                  % (it, time.time() - t0, np.vdot(C, X), b @ y, np.vdot(X, Z), pinf, dinf), flush=True)
        if stop:
            break
        try:
            X, y, Z = step(X, y, Z, rp, Rd, np.vdot(X, Z) / d)
        except np.linalg.LinAlgError:
            print("  it %3d: a factorization failed, keeping the last iterate" % it, flush=True)
            break
    # in this form the dual slack is Z = Y - tI (t the last multiplier) and the primal
    # variable is the certificate, whose objective <Y0, W> plays the part of y0
    report(inst, Z, X, y[-1], np.vdot(C, X), B, adj, w, equ)


def cmd_save(args):
    inst = args.instance
    adj, w, equ = load(inst)
    n, m = len(w), len(equ)
    B = jam_basis(inst, w, equ)
    r = np.load(work(inst, ".ipm.npz"))
    Z, W, t = r["Z"], r["W"], float(r["t"])
    d = B.shape[1]
    ez, U = np.linalg.eigh(Z)
    ez, U = ez[::-1], U[:, ::-1]
    gap = int(np.argmax(np.log(np.maximum(ez[:-1], 1e-300)) - np.log(np.maximum(ez[1:], 1e-300)))) + 1
    cut = args.dim or gap
    wz = np.einsum("ij,jk,ki->i", U.T, W, U) / np.linalg.eigvalsh(W).max()
    print("%s: largest gap of Z after %d, cut at %d; index: Z eigenvalue / largest, u^T W u / largest W eigenvalue" % (inst, gap, cut))
    print("  " + "  ".join("%d:%.1e/%.1e" % (i + 1, ez[i] / ez[0], wz[i]) for i in range(max(0, cut - 5), min(d, cut + 5))))
    basis = B @ U[:, :cut]
    Y = Z + t * np.eye(d)
    X = B @ (Y / np.trace(Y)) @ B.T
    contr = ~adj & ~np.eye(n, dtype=bool)
    outs = [np.linalg.norm(x - basis @ (basis.T @ x)) / np.linalg.norm(x)
            for x in indicators(solutions(inst, adj, equ), np.sqrt(w), m)]
    out = os.path.join(S.RUNS, "work", inst, "full", inst + ".jam.npz")
    np.savez(out, basis=basis, X=X, W=B @ W @ B.T)
    print("  R* of dimension %d of %d; |X| on the contradictions %.1e; %d solutions, largest share outside %.1e; %s"
          % (cut, d, np.abs(X[contr]).max(), len(outs), max(outs), out))


# ---------------------------------------------------------------- Douglas-Rachford


def cmd_dr(args):
    for inst in args.instances:
        adj, w, equ = load(inst)
        n, m = len(w), len(equ)
        z = np.sqrt(w)
        B = jam_basis(inst, w, equ)
        refine = J.MinRefiner(adj, w)
        if args.proj == "clause":
            lib = ctypes.CDLL(build("libclause.so", ["clause.cc"], ["-fopenmp", "-fPIC", "-shared"]))
            A = np.zeros((m, n))
            for e, vs in enumerate(equ):
                A[e, vs] = 1
            two = ~adj & ~np.eye(n, dtype=bool) & (A.T @ A == 0)
            ptr = np.concatenate([[0], np.cumsum(two.sum(1))]).astype(np.int32)
            idx = np.concatenate([np.flatnonzero(two[i]) for i in range(n)]).astype(np.int32)

            def project(Y):
                Y = np.asfortranarray(Y)
                lib.clause_rows(n, Y.shape[1], Y.ctypes.data_as(ctypes.c_void_p),
                                ptr.ctypes.data_as(ctypes.c_void_p), idx.ctypes.data_as(ctypes.c_void_p))
                return Y
        else:
            project = lambda Y: np.maximum(Y, 0)
        sols = solutions(inst, adj, equ)
        for name, R in (("L_A", B), ("R*", np.load(os.path.join(S.RUNS, "work", inst, "full", inst + ".jam.npz"))["basis"])):
            a = R.T @ z
            c = R @ a / (a @ a)
            rho = np.sqrt(1.0 / m - c @ c)
            _, _, vt = np.linalg.svd(a[None, :])
            D = R @ vt[1:].T
            outside = max(np.linalg.norm(x - R @ (R.T @ x)) / np.linalg.norm(x) for x in indicators(sols, z, m))

            def sphere(Y):
                Pq = D @ (D.T @ (Y - c[:, None]))
                return c[:, None] + rho * Pq / np.maximum(np.linalg.norm(Pq, axis=0), 1e-300)
            V = np.hstack([c[:, None] + rho * D, c[:, None] - rho * D])
            E = sphere(V)
            live = np.ones(V.shape[1], bool)
            t0, steps = time.time(), 0
            for _ in range(args.iters):
                L = np.flatnonzero(live)
                if not len(L):
                    break
                steps += len(L)
                E[:, L] = sphere(V[:, L])
                V[:, L] += project(2 * E[:, L] - V[:, L]) - E[:, L]
                gapn = np.linalg.norm(project(E[:, L].copy()) - E[:, L], axis=0) / np.linalg.norm(E[:, L], axis=0)
                live[L[gapn <= 1e-9]] = False
            t1 = time.time()
            wts = np.array([refine.weight(E[:, j] * z) for j in range(E.shape[1])])
            print("%s %s on %s (dimension %d, solutions outside up to %.1e): %d starts, %d stopped, mean %.1f steps,"
                  " %.1fs; best MIN %g of %d, hits %d"
                  % (inst, args.proj, name, R.shape[1], outside, V.shape[1], int((~live).sum()), steps / V.shape[1],
                     t1 - t0, wts.max(), m, int((wts >= m - 1e-9).sum())), flush=True)


# ---------------------------------------------------------------- the factored search


def factored(B, adj, delta=0.0):
    """f(C) = sum over the contradictions of <S_ij, Y>^2 and its gradient for
    Y = delta I + (1 - d delta) C C^T/|C|^2 (C square, flattened)"""
    n, d = B.shape
    contr = ~adj & ~np.eye(n, dtype=bool)
    if not delta:
        def fg0(x):
            C = x.reshape(d, d)
            A = B @ C
            Phi = (A @ A.T) * contr
            Pq = 0.5 * np.vdot(Phi, Phi)
            s = x @ x
            G = 2.0 * (B.T @ (Phi @ A))
            return Pq / s**2, (G.ravel() / s**2 - 4.0 * Pq * x / s**3)
        return fg0
    Pi = delta * (B @ B.T) * contr
    a = 1.0 - d * delta

    def fg(x):
        C = x.reshape(d, d)
        s = x @ x
        A = B @ C
        R = Pi + (a / s) * ((A @ A.T) * contr)
        RA = R @ A
        g = (2.0 * a / s) * (B.T @ RA) - (2.0 * a * np.vdot(RA, A) / s**2) * C
        return 0.5 * np.vdot(R, R), g.ravel()
    return fg


def lbfgs(fg, x, iters):
    return so.minimize(fg, x, jac=True, method="L-BFGS-B",
                       options={"maxiter": iters, "maxfun": 3 * iters, "ftol": 0, "gtol": 0, "maxcor": 20})


def cmd_bm(args):
    inst = args.instance
    adj, w, equ = load(inst)
    n, m = len(w), len(equ)
    z = np.sqrt(w)
    B = jam_basis(inst, w, equ)
    d = B.shape[1]
    coords = [B.T @ x for x in indicators(solutions(inst, adj, equ), z, m)]
    cuts = (1e-2, 1e-4, 1e-6, 1e-8)
    if args.mode == "decode":
        # at each checkpoint the top three eigenvectors of Y (both signs), mapped by B, go to
        # MIN, and the shares of the solutions outside Y's top eigenspaces are printed
        refine = J.MinRefiner(adj, w)
        fg = factored(B, adj)
        x = np.random.default_rng(0).standard_normal(d * d)
        done, t0 = 0, time.time()
        print("%s: n=%d m=%d, dim L_A=%d, %d known solutions" % (inst, n, m, d, len(coords)), flush=True)
        for cp in (int(c) for c in args.checkpoints.split(",")):
            res = lbfgs(fg, x, cp - done)
            x, done = res.x, cp
            C = x.reshape(d, d)
            ev, U = np.linalg.eigh(C @ C.T / (x @ x))
            ev, U = ev[::-1], U[:, ::-1]
            best = max(refine.weight(sg * (B @ U[:, j]) * z) for j in range(3) for sg in (1, -1))
            counts = [int((ev > c * ev[0]).sum()) for c in cuts]
            line = ("  %6d iterations, %6.1fs: f=%.2e, top eigenvalues %s; above 1e-2/4/6/8 of the largest: %s;"
                    " best MIN weight %g of %d%s" % (cp, time.time() - t0, res.fun, " ".join("%.3g" % e for e in ev[:4]),
                                                      "/".join(map(str, counts)), best, m, "  SOLVED" if best >= m - 1e-9 else ""))
            for c in coords:
                p = U.T @ c
                line += "; a solution outside: " + "/".join(
                    "%.1e" % np.sqrt(max(0.0, 1.0 - (p[:k] @ p[:k]) / (c @ c))) for k in counts)
            print(line, flush=True)
        np.save(work(inst, ".C0.npy"), x.reshape(d, d))
    elif args.mode == "starts":
        fg = factored(B, adj)
        for seed in (int(s) for s in args.seeds.split(",")):
            t0 = time.time()
            res = lbfgs(fg, np.random.default_rng(seed).standard_normal(d * d), args.iters)
            C = res.x.reshape(d, d)
            ev = np.linalg.eigvalsh(C @ C.T / (res.x @ res.x))[::-1]
            np.save(work(inst, ".C%d.npy" % seed), C)
            print("%s seed %d: %d iterations, %.1fs, f=%.2e; eigenvalues above 1e-2/4/6/8 of the largest: %s"
                  % (inst, seed, res.nit, time.time() - t0, res.fun,
                     "/".join(str(int((ev > c * ev[0]).sum())) for c in cuts)), flush=True)
    elif args.mode == "combine":
        # every saved start is a point of the face to its accuracy, and so is their mean
        contr = ~adj & ~np.eye(n, dtype=bool)
        files = sorted(f for f in os.listdir(WORK) if f.startswith(inst + ".C") and f.endswith(".npy"))
        Ys = []
        for f in files:
            C = np.load(os.path.join(WORK, f))
            Ys.append(C @ C.T / np.sum(C * C))
        for name, Y in [(f[len(inst) + 1:-4], Y) for f, Y in zip(files, Ys)] + [("mean of %d" % len(Ys), sum(Ys) / len(Ys))]:
            X = B @ Y @ B.T
            ev, U = np.linalg.eigh(Y)
            ev, U = ev[::-1], U[:, ::-1]
            counts = [int((ev > c * ev[0]).sum()) for c in (1e-4, 1e-6, 1e-8, 1e-10)]
            line = "%-12s |A(Y)| %.1e; eigenvalues above 1e-4/6/8/10: %s; smallest %.1e" % (
                name, np.sqrt(0.5 * np.sum((X * contr) ** 2)), "/".join(map(str, counts)), ev[-1] / ev[0])
            for c in coords:
                p = U.T @ c
                line += "; a solution outside: " + "/".join(
                    "%.0e" % np.sqrt(max(0.0, 1 - (p[:k] @ p[:k]) / (c @ c))) for k in counts)
            print(line, flush=True)
    else:
        # floor: at a minimum of |A(Y)|^2 over Y >= delta I the KKT conditions make
        # W = A*(r)/|r| PSD up to O(delta); a stopped run shows whether it got there
        for delta in (float(s) for s in args.delta.split(",")):
            fg = factored(B, adj, delta)
            t0 = time.time()
            res = lbfgs(fg, np.random.default_rng(0).standard_normal(d * d), args.iters)
            C = res.x.reshape(d, d)
            s = res.x @ res.x
            contr = ~adj & ~np.eye(n, dtype=bool)
            A = B @ C
            R = delta * (B @ B.T) * contr + ((1.0 - d * delta) / s) * ((A @ A.T) * contr)
            nr = np.sqrt(0.5 * np.vdot(R, R))
            ew = np.linalg.eigvalsh(B.T @ R @ B / nr)[::-1]
            print("%s delta=%.0e: %d iterations, %.1fs, f=%.3e (|r| %.2e, |r|/delta %.3g); W/|r|: largest %.3g,"
                  " most negative relative %.1e, above 1e-2/1e-4/1e-6 of the largest: %s"
                  % (inst, delta, res.nit, time.time() - t0, res.fun, nr, nr / delta, ew[0], ew[-1] / ew[0],
                     "/".join(str(int((ew > c * ew[0]).sum())) for c in (1e-2, 1e-4, 1e-6))), flush=True)


def main():
    ap = argparse.ArgumentParser(description="the smallest jam of a satisfiable SAT01 instance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("structure", help="dim V and an independent G' by exact elimination")
    p.add_argument("instances", nargs="+")
    p.set_defaults(fn=cmd_structure)
    p = sub.add_parser("select", help="a well conditioned G' of the same exact rank")
    p.add_argument("instances", nargs="+")
    p.add_argument("--extra", type=int, default=6000, help="further pairs offered to the pivoted Cholesky")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("cond", help="the conditioning of the eliminated G'")
    p.add_argument("instances", nargs="+")
    p.set_defaults(fn=cmd_cond)
    p = sub.add_parser("ipm", help="the interior point method in the constraints of G'")
    p.add_argument("instance")
    p.add_argument("--packed", action="store_true", help="the Schur matrix in rectangular full packed storage")
    p.set_defaults(fn=cmd_ipm)
    p = sub.add_parser("dense", help="the interior point method over a dense basis of V")
    p.add_argument("instance")
    p.add_argument("--cut", type=float, default=1e-10, help="null space cut, relative to the largest eigenvalue")
    p.set_defaults(fn=cmd_dense)
    p = sub.add_parser("save", help="R*, the optimum and the certificate into <inst>.jam.npz")
    p.add_argument("instance")
    p.add_argument("--dim", type=int, help="the dimension of R*, if not the largest gap of Z")
    p.set_defaults(fn=cmd_save)
    p = sub.add_parser("dr", help="Douglas-Rachford on the spheres of L_A and of R*")
    p.add_argument("instances", nargs="+")
    p.add_argument("--proj", choices=("clause", "orthant"), default="clause")
    p.add_argument("--iters", type=int, default=300)
    p.set_defaults(fn=cmd_dr)
    p = sub.add_parser("bm", help="the factored search")
    p.add_argument("mode", choices=("decode", "starts", "combine", "floor"))
    p.add_argument("instance")
    p.add_argument("--checkpoints", default="200,400,800,1600,3200,6400,12800,25600")
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--iters", type=int, default=6400)
    p.add_argument("--delta", default="1e-5,1e-6")
    p.set_defaults(fn=cmd_bm)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
