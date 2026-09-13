#!/usr/bin/env python3
"""Generate uniform random graphs G(n,p) for benchmarking QUALEX-MS.

Every pair of distinct vertices is joined independently with probability p.
A graph g<n>_<100p>_<rep> is written as

  g<n>_<100p>_<rep>.clq.b  binary DIMACS, the solver's input
  g<n>_<100p>_<rep>.w      integer vertex weights drawn uniformly from 1..10,
                           for the weighted problem (qualex-ms -w<file>)
  g<n>_<100p>_<rep>.clq    ASCII DIMACS with 'n' weight lines, for checking
                           optima with cliquer (only with --cliquer)

together with a list n<n>.lst of the graph names of every size.  For every
(n,p) it prints k1, the largest k for which the expected number of k-cliques
is at least 1, the first moment estimate of the clique number.

A graph's seed depends on (n, p, rep) alone, so restricting --sizes,
--densities or --reps reproduces the same graphs.  Without options this
writes the 132 graphs of the uniform random graph section of benchmarks.md.
They were made with numpy 2.3.5; numpy does not promise the same stream from
Generator.random() and Generator.integers() in every release.

Usage: random_graphs.py [--sizes 200,500] [--densities 0.5,0.9] [--reps N]
                        [--cliquer] <outdir>
"""
import argparse
import math
import os

import numpy as np

SIZES = (200, 300, 500, 1000, 2000)
DENSITIES = (0.25, 0.5, 0.7, 0.8, 0.9, 0.95)
REPS = {2000: 2}   # graphs per (n,p) where it is not the default 5


def generate(n, p, rep):
    """the edges as a strictly lower triangular boolean matrix, and weights"""
    rng = np.random.default_rng(1000003*n + 101*int(round(100*p)) + rep)
    lower = np.tril(rng.random((n, n)) < p, k=-1)
    weights = rng.integers(1, 11, size=n)
    return lower, weights


def write_graph(path, lower, weights, cliquer):
    n = lower.shape[0]
    preamble = "c uniform random graph G(n,p)\np edge %d %d\n" % (
        n, int(lower.sum()))
    # the binary format stores row i as the bits of j = 0..i, most significant
    # bit first, in (i>>3)+1 bytes
    bits = np.zeros((n, 8*((n >> 3) + 1)), dtype=bool)
    bits[:, :n] = lower
    with open(path + ".clq.b", "wb") as f:
        f.write(("%d\n" % len(preamble)).encode())
        f.write(preamble.encode())
        for i in range(n):
            f.write(np.packbits(bits[i, :8*((i >> 3) + 1)]).tobytes())
    with open(path + ".w", "w") as f:
        f.write("".join("%d\n" % x for x in weights))
    if cliquer:
        rows, cols = np.nonzero(lower)
        with open(path + ".clq", "w") as f:
            f.write(preamble)
            f.write("".join("n %d %d\n" % (i + 1, x)
                            for i, x in enumerate(weights)))
            f.write("".join("e %d %d\n" % (i + 1, j + 1)
                            for i, j in zip(rows, cols)))


def first_moment_clique_number(n, p):
    """the largest k with E[#k-cliques] = C(n,k) p^(k(k-1)/2) >= 1"""
    def log_expected(k):
        return (math.lgamma(n + 1) - math.lgamma(k + 1)
                - math.lgamma(n - k + 1) + k*(k - 1)/2*math.log(p))
    k = 1
    while k < n and log_expected(k + 1) >= 0.0:
        k += 1
    return k


def main():
    ap = argparse.ArgumentParser(
        description="Generate uniform random graphs G(n,p) for QUALEX-MS.")
    ap.add_argument("outdir")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)),
                    help="comma-separated numbers of vertices")
    ap.add_argument("--densities", default=",".join(map(str, DENSITIES)),
                    help="comma-separated edge probabilities")
    ap.add_argument("--reps", type=int,
                    help="graphs per (n,p), by default 5 (2 for n=2000)")
    ap.add_argument("--cliquer", action="store_true",
                    help="also write ASCII DIMACS files with the weights")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    densities = [float(x) for x in args.densities.split(",")]
    if any(not 0.0 < p < 1.0 for p in densities):
        ap.error("densities must lie strictly between 0 and 1")

    os.makedirs(args.outdir, exist_ok=True)
    for n in sizes:
        names = []
        for p in densities:
            for rep in range(args.reps or REPS.get(n, 5)):
                name = "g%d_%02d_%d" % (n, int(round(100*p)), rep)
                lower, weights = generate(n, p, rep)
                write_graph(os.path.join(args.outdir, name), lower, weights,
                            args.cliquer)
                names.append(name)
            print("n=%d p=%.2f k1=%d" % (n, p, first_moment_clique_number(n, p)))
        with open(os.path.join(args.outdir, "n%d.lst" % n), "w") as f:
            f.write("".join(name + "\n" for name in names))


if __name__ == "__main__":
    main()
