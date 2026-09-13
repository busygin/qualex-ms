#!/usr/bin/env python3
"""Dense numpy versions of the wrapper constructions in main.cc.

For checking an idea on a graph or two before touching the C++; slow, but easy
to take apart.  The matrix A is the solver's A^(w) = H - w_min I, the wrapper H
less w_min on the diagonal, with z = sqrt(w).

  adj = read_dimacs_bin(path)          the adjacency matrix of a binary DIMACS file
  A = wrapper(adj, w)                  build_wrapper()
  A = anchor(A, adj, w, Q, theta)      anchor_wrapper(): Theorem 8 exact on Q
  A, sigma = lovasz_step(A, adj, w, Q) ice_step() in mode "lovasz", re-anchoring
                                       on Q when Q is given
  hatA, hatb = project(A, w)           init_projected_matrices()
  invariant_spread(A, w)               (lambda_max - lambda_min) / |hatb| of hatA,
                                       which H -> (1+t) H - t z z^T leaves alone
  theorem8_residual(A, adj, w, Q)      how far the indicator of Q is from being
                                       stationary, at mu* = W(Q) - w_min - C

Run as a script it summarizes these for one graph, anchoring on the clique of a
.sol file (default: a greedy clique):

  bench/wrapper.py GRAPH.clq.b [-w WEIGHTS] [-q CLIQUE.sol]
"""
import argparse

import numpy as np

SIGMAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)  # as in ice_step()


def read_dimacs_bin(path):
    with open(path, "rb") as f:
        preamble = f.read(int(f.readline())).decode()
        n = next(int(l.split()[2]) for l in preamble.splitlines() if l.startswith("p"))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    adj = np.zeros((n, n), dtype=bool)
    offset = 0
    for i in range(n):
        size = (i >> 3) + 1
        adj[i, :i] = np.unpackbits(data[offset:offset + size])[:i].astype(bool)
        offset += size
    return adj | adj.T


def read_weights(path, n):
    return np.ones(n) if path is None else np.loadtxt(path)[:n]


def read_sol(path):
    """the vertices of a .sol file written without +1"""
    return [int(line[1:]) for line in open(path) if line.startswith("v")]


def greedy_clique(adj, w):
    """a maximal clique, taking the candidate with the heaviest candidate neighbourhood"""
    candidates = np.ones(len(w), bool)
    clique = []
    while candidates.any():
        score = np.where(candidates, adj[:, candidates].astype(float) @ w[candidates] + w, -1.0)
        v = int(np.argmax(score))
        clique.append(v)
        candidates &= adj[v]
        candidates[v] = False
    return clique


def wrapper(adj, w):
    z = np.sqrt(w)
    A = np.where(adj, np.outer(z, z), 0.0)
    np.fill_diagonal(A, w - w.min())
    return A


def project(A, w):
    z = np.sqrt(w)
    P = np.eye(len(w)) - np.outer(z, z) / w.sum()
    return P @ A @ P, P @ A @ z / w.sum()


def anchor(A, adj, w, Q, theta=1.0):
    z = np.sqrt(w)
    inside = np.zeros(len(w), bool)
    inside[Q] = True
    out = np.where(~inside)[0]
    missing = ~adj[np.ix_(out, Q)]
    attachment = (A[np.ix_(out, Q)] * z[Q]).sum(1) / z[out]
    missing_weight = (missing * w[Q]).sum(1)
    if missing_weight.min() <= 0:
        raise ValueError("the clique is not maximal")
    t = theta * (attachment.min() - attachment) / missing_weight
    E = np.where(missing, t[:, None] * np.outer(z[out], z[Q]), 0.0)
    B = A.copy()
    B[np.ix_(out, Q)] += E
    B[np.ix_(Q, out)] += E.T
    return B


def theorem8_residual(A, adj, w, Q):
    """(max |A x - mu x - nu z|, mu*, spread of the attachments) at the indicator x of Q"""
    z = np.sqrt(w)
    inside = np.zeros(len(w), bool)
    inside[Q] = True
    attachment = (A[np.ix_(~inside, inside)] @ z[inside]) / z[~inside]
    wq = w[inside].sum()
    mu = wq - w.min() - attachment.min()
    x = np.where(inside, z / wq, 0.0)
    r = A @ x - mu * x
    nu = (r @ z) / w.sum()
    return np.abs(r - nu * z).max(), mu, attachment.max() - attachment.min()


def lovasz_step(A, adj, w, Q=None, theta=1.0):
    """one line-searched step on lambda_max of the wrapper along 2 u_i u_j on the
    non-edges, clipped at 0 and re-anchored on Q if given; returns the matrix and
    the sigma the line search took (0: no step)"""
    n = len(w)
    nonedge = ~adj & ~np.eye(n, dtype=bool)
    lam, U = np.linalg.eigh(A)
    u = U[:, -1]
    d = np.where(nonedge, -2.0 * np.outer(u, u), 0.0)
    norm2 = (d[np.triu_indices(n, 1)] ** 2).sum()
    best, B, sigma = lam[-1], A, 0.0
    for s in SIGMAS:
        T = np.where(nonedge, np.minimum(A + s * lam[-1] / norm2 * d, 0.0), A)
        if Q is not None:
            T = anchor(T, adj, w, Q, theta)
        top = np.linalg.eigvalsh(T)[-1]
        if top < best:
            best, B, sigma = top, T, s
    return B, sigma


def invariant_spread(A, w):
    hatA, hatb = project(A, w)
    lam = np.linalg.eigvalsh(hatA)
    norm = np.linalg.norm(hatb)
    if norm <= 1e-12 * max(1.0, np.abs(lam).max()):
        return float("inf")
    return (lam[-1] - lam[0]) / norm


def main():
    ap = argparse.ArgumentParser(description="Summarize the wrapper constructions for one graph.")
    ap.add_argument("graph", help="a binary DIMACS file")
    ap.add_argument("-w", "--weights", help="vertex weights, one per line")
    ap.add_argument("-q", "--clique", help="a .sol file with the clique to anchor on")
    args = ap.parse_args()
    adj = read_dimacs_bin(args.graph)
    n = len(adj)
    w = read_weights(args.weights, n)
    Q = read_sol(args.clique) if args.clique else greedy_clique(adj, w)
    inside = np.zeros(n, bool)
    inside[Q] = True
    attachment = (adj[np.ix_(~inside, inside)] * w[inside]).sum(1)
    print("n=%d, |Q|=%d, W(Q)=%g, attachments W(N(i) cap Q) %g..%g" % (
        n, len(Q), w[Q].sum(), attachment.min(), attachment.max()))

    def show(label, M, extra=""):
        lam = np.linalg.eigvalsh(project(M, w)[0])
        print("  %-24s projected lambda %9.3f .. %-9.3f spread/|hatb| %8.3f  %s" % (
            label, lam[0], lam[-1], invariant_spread(M, w), extra))

    A = wrapper(adj, w)
    nonedge = ~adj & ~np.eye(n, dtype=bool)
    u = np.linalg.eigh(A)[1][:, -1]
    g = np.outer(u, u)[nonedge]
    zz = np.outer(np.sqrt(w), np.sqrt(w))[nonedge]
    show("standard wrapper", A, "lambda_max gradient vs uniform: cosine %.4f" % (
        g @ zz / np.linalg.norm(g) / np.linalg.norm(zz)))
    Aa = anchor(A, adj, w, Q)
    residual, mu, spread = theorem8_residual(Aa, adj, w, Q)
    above = int((np.linalg.eigvalsh(project(Aa, w)[0]) > mu * (1 + 1e-12)).sum())
    show("anchored on Q", Aa, "mu* %g, residual %.1e, eigenvalues above mu* %d" % (mu, residual, above))
    for label, M, Qs in (("literal Lovasz step", A, None), ("anchored, then the step", Aa, Q)):
        B, sigma = lovasz_step(M, adj, w, Qs)
        show(label, B, "sigma %g, lambda_max(A) %.3f -> %.3f" % (
            sigma, np.linalg.eigvalsh(M)[-1], np.linalg.eigvalsh(B)[-1]))


if __name__ == "__main__":
    main()
