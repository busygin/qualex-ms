#!/usr/bin/env python3
"""SAT01 instances as weighted clique problems: when does theta see that alpha < m?

SAT01 (Busygin, ~/SAT01/sat01theta.tex): find x in {0,1}^n with Ax = 1, A the 0/1
equation-variable matrix, and x_i x_j = 0 on the contradiction graph Gamma, which holds
every pair of variables sharing an equation.  With the weights w = A^T 1, the number of
equations of each variable, the equations cover Gamma by m cliques, so

    alpha(Gamma, w) <= theta(Gamma, w) <= m,   alpha = m iff the instance is satisfiable.

theta < m refutes an instance in polynomial time; theta = m on an unsatisfiable one is a
gap no clique wrapper can see, the best relaxed program value over the wrappers being
1/theta.  sat012clique -w writes the complement of Gamma, whose weighted clique number
is alpha.  For each instance the script
  - builds it with the SAT01 tools: f<N> factors N (factor2sat01), the graph names
    below are Hamiltonian cycle problems (hcp2sat01);
  - runs the SAT01 solver for the answer (sat, unsat, or timeout) and its guesses;
  - converts it with sat012clique -w after light preprocessing (the assigned variables
    eliminated) and after the solver's full preprocessing (-p), whose contradiction
    analysis adds derived contradictions: alpha stays, theta can only drop;
  - computes theta(Gamma, w) of each graph that is left with CSDP: the X-form of
    theta.py with the objective z z^T, z = sqrt(w), one constraint per contradiction
    pair, skipped above --max-pairs (CSDP holds a pairs x pairs matrix).

HCP graphs (hcp2sat01 fixes vertex 1 at position 0, so every weight is 2, m = 2(n - 1)):
  gp<n>_<k>     generalized Petersen graph GP(n, k); gp5_2 is the Petersen graph,
                gp10_2 the dodecahedron; GP(n, 2) with n = 5 mod 6 is not Hamiltonian
  flower<k>     flower snark J_k, k odd, not Hamiltonian
  coxeter       the Coxeter graph, not Hamiltonian
  cubic<n>_<s>  random connected cubic graph on n vertices, seed s
  bip<k>_<s>    random connected bipartite graph with sides k and k + 1, degrees 3 and 4,
                not Hamiltonian by parity

Usage: bench/sat01.py [-f] [--prep light|full] [--timeout 600] [--max-pairs 50000] [-t 12] INSTANCE ...
       bench/sat01.py --span [--cap 20000] INSTANCE ...
       bench/sat01.py --qms [-f] [-t 12] INSTANCE ...

Rows are appended to runs/sat01/sat01.tsv, one per instance and preprocessing:
  inst answer guesses prep n m pairs blind theta lmax gap rank seconds
where n = 0 when preprocessing decided the instance, pairs counts the contradiction
pairs, blind is the share of them sharing no equation (invisible to the wrapper
H_A = z z^T + m (I - D^-1/2 A^T A D^-1/2), D = diag(w)), lmax is lambda_max of the
wrapper CSDP's dual gives, a certified upper bound on theta, gap = m - theta, and rank
that of the Lovasz optimum (eigenvalues above 1e-5 of the largest).  A pair already there is skipped unless -f.  The tools are
built by the Makefile of $SAT01 (default ~/SAT01) against this repository's lib/ and
copied into runs/sat01/bin; instances, solver runs and CSDP files go to
runs/sat01/work/<inst>/, both ignored.

With --span, satisfiable instances already solved get a second pass: their solutions
are listed (exact cover search, up to --cap), and runs/sat01/span.tsv gets
  inst prep sols span rank top dimLA resid nodes seconds
the number of solutions (+ at the cap), the dimension they span, the rank of the Lovasz
optimum, the top multiplicity of CSDP's optimal wrapper (within 1e-5 lambda_max), that of H_A (dim null A + 1),
the largest residual of H (z o 1_K) = m (z o 1_K) over the solutions K (Lovasz optimality
anchors every solution at the top), and the search nodes.

With --qms, QUALEX-MS runs on each instance instead, after the solver's full propagation
and without search (SAT01's sat01qms), every QMS_* switch unset but those of the
configuration: on the equation wrapper H_A or on the standard clique wrapper
("equation", "standard"), and each with its stationary points around the radius of a
clique one w_min heavier than the incumbent, as QUALEX-MS has them, or at the radius of
a clique of weight m with the method of QUALEX-MS 1.2 ("-m"), and that with the Meta-NBIW
stage on the two best multipliers ("-m-meta", QMS_META_N=2); and on H_A with QUALEX-MS's
Douglas-Rachford stage (QMS_DR=300) using the greedy 2-clause projection ("equation-dr",
sat01qms -D) or, as the control, the orthant ("equation-dr-orthant"), and each of those
driving the points onto the surface of the standard wrapper as well ("-w", sat01qms -W),
DR running between the product of the sphere and that surface and the diagonal of the
2-clause set or the orthant, or ("-concur", QMS_DR_CONCUR) in the symmetric product space
of all the sets, which is also run without the surface as the control, and onto the
surface alone, without the orthant ("-w-only", sat01qms -N); the DR lines
(QMS_STATS) go to the configuration's qms.log, and runs/sat01/qms.tsv gets
  inst answer guesses config n m preselected left greedy weight solution verified prop qms
the size propagation leaves (n = 0 when it decides the instance), the vertices QUALEX-MS's
preprocessing preselects and leaves, the weight of the clique Meta-NBIW and then
QUALEX-MS find (m is a solution, which sat01qms checks against the instance), and the
seconds of propagation and of QUALEX-MS.
"""
import argparse
import filecmp
import functools
import os
import re
import shutil
import subprocess
import time

import numpy as np

import theta as T
from wrapper import read_dimacs_bin

SAT01 = os.environ.get("SAT01", os.path.expanduser("~/SAT01"))
RUNS = os.path.join(T.BENCH, "runs", "sat01")
TSV = os.path.join(RUNS, "sat01.tsv")
QMS_TSV = os.path.join(RUNS, "qms.tsv")
# configuration -> the flags of sat01qms and the QMS_* settings
CONFIGS = {
    "equation": ([], {}),
    "standard": (["-s"], {}),
    "equation-m": (["-m"], {}),
    "standard-m": (["-s", "-m"], {}),
    "equation-m-meta": (["-m"], {"QMS_META_N": "2"}),
    "standard-m-meta": (["-s", "-m"], {"QMS_META_N": "2"}),
    "equation-dr": (["-D"], {"QMS_DR": "300", "QMS_STATS": "1"}),
    "equation-dr-orthant": ([], {"QMS_DR": "300", "QMS_STATS": "1"}),
    "equation-dr-w": (["-D", "-W"], {"QMS_DR": "300", "QMS_STATS": "1"}),
    "equation-dr-orthant-w": (["-W"], {"QMS_DR": "300", "QMS_STATS": "1"}),
    "equation-dr-concur": (["-D"], {"QMS_DR": "300", "QMS_DR_CONCUR": "1", "QMS_STATS": "1"}),
    "equation-dr-orthant-concur": ([], {"QMS_DR": "300", "QMS_DR_CONCUR": "1", "QMS_STATS": "1"}),
    "equation-dr-w-concur": (["-D", "-W"], {"QMS_DR": "300", "QMS_DR_CONCUR": "1", "QMS_STATS": "1"}),
    "equation-dr-orthant-w-concur": (["-W"], {"QMS_DR": "300", "QMS_DR_CONCUR": "1", "QMS_STATS": "1"}),
    "equation-dr-w-only": (["-N"], {"QMS_DR": "300", "QMS_STATS": "1"}),
    "equation-dr-w-only-concur": (["-N"], {"QMS_DR": "300", "QMS_DR_CONCUR": "1", "QMS_STATS": "1"}),
}


@functools.lru_cache(maxsize=None)
def tool(name):
    """a SAT01 tool (sat01, sat012clique, factor2sat01, hcp2sat01), built by $SAT01's
    Makefile against this repository's lib/ and copied into runs/sat01/bin when it changed"""
    subprocess.run(["make", "-s", "-C", SAT01, "QMS=" + os.path.dirname(T.BENCH), name],
                   check=True, stdout=subprocess.DEVNULL)
    src, exe = os.path.join(SAT01, name), os.path.join(RUNS, "bin", name)
    if not os.path.exists(exe) or not filecmp.cmp(src, exe, shallow=False):
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        shutil.copy2(src, exe)
    return exe


def cycle(vs):
    vs = list(vs)
    return [(vs[i], vs[(i + 1) % len(vs)]) for i in range(len(vs))]


def connected(n, edges):
    nbr = [[] for _ in range(n)]
    for u, v in edges:
        nbr[u].append(v)
        nbr[v].append(u)
    seen, todo = {0}, [0]
    while todo:
        for v in nbr[todo.pop()]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return len(seen) == n


def hcp_graph(name):
    """the number of vertices and the edges of a named graph (see the docstring)"""
    name = {"petersen": "gp5_2", "dodecahedron": "gp10_2"}.get(name, name)
    if m := re.fullmatch(r"gp(\d+)_(\d+)", name):
        n, k = map(int, m.groups())
        return 2 * n, cycle(range(n)) + [(i, n + i) for i in range(n)] + [(n + i, n + (i + k) % n) for i in range(n)]
    if m := re.fullmatch(r"flower(\d+)", name):
        k = int(m.group(1))
        a, b, c, d = (list(range(o * k, (o + 1) * k)) for o in range(4))
        return 4 * k, [(a[i], x[i]) for i in range(k) for x in (b, c, d)] + cycle(b) + cycle(c + d)
    if name == "coxeter":
        a, b, c, d = (list(range(o * 7, (o + 1) * 7)) for o in range(4))
        return 28, ([(a[i], a[(i + 1) % 7]) for i in range(7)] + [(b[i], b[(i + 2) % 7]) for i in range(7)]
                    + [(c[i], c[(i + 3) % 7]) for i in range(7)] + [(d[i], x[i]) for i in range(7) for x in (a, b, c)])
    if m := re.fullmatch(r"cubic(\d+)_(\d+)", name):
        n, s = map(int, m.groups())
        rng = np.random.default_rng(s)
        while True:
            p = rng.permutation(np.repeat(np.arange(n), 3)).reshape(-1, 2).tolist()
            E = {(min(u, v), max(u, v)) for u, v in p if u != v}
            if len(E) == 3 * n // 2 and connected(n, E):
                return n, sorted(E)
    if m := re.fullmatch(r"bip(\d+)_(\d+)", name):
        k, s = map(int, m.groups())
        rng = np.random.default_rng(s)
        # side A = 0..k-1 with three vertices of degree 4, side B = k..2k of degree 3
        sa = np.concatenate([np.repeat(np.arange(k), 3), rng.choice(k, 3, replace=False)]).tolist()
        sb = np.repeat(np.arange(k, 2 * k + 1), 3)
        while True:
            E = set(zip(sa, rng.permutation(sb).tolist()))
            if len(E) == len(sa) and connected(2 * k + 1, E):
                return 2 * k + 1, sorted(E)
    raise SystemExit("unknown instance " + name)


def build(inst):
    """the instance's work directory, holding <inst>.sat01"""
    work = os.path.join(RUNS, "work", inst)
    sat = os.path.join(work, inst + ".sat01")
    if not os.path.exists(sat):
        os.makedirs(work, exist_ok=True)
        if m := re.fullmatch(r"f(\d+)", inst):
            subprocess.run([tool("factor2sat01"), m.group(1)], cwd=work, stdout=subprocess.DEVNULL, check=True)
            os.rename(os.path.join(work, m.group(1) + ".sat01"), sat)
        else:
            n, edges = hcp_graph(inst)
            # hcp2sat01 names its output after the input up to the first dot
            with open(os.path.join(work, inst + ".hcp"), "w") as f:
                f.write("NAME : %s\nTYPE : HCP\nDIMENSION : %d\nEDGE_DATA_FORMAT : EDGE_LIST\n"
                        "EDGE_DATA_SECTION\n" % (inst, n))
                f.write("".join("%d %d\n" % (u + 1, v + 1) for u, v in edges) + "-1\nEOF\n")
            subprocess.run([tool("hcp2sat01"), inst + ".hcp"], cwd=work, stdout=subprocess.DEVNULL, check=True)
    return work


def answer(work, inst, timeout):
    """the SAT01 solver's verdict (sat, unsat or timeout) and its number of guesses, run
    in a directory of the instance's own"""
    d = os.path.join(work, "solve")
    rec = os.path.join(d, "answer")
    if os.path.exists(rec):
        return open(rec).read().split()
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(work, inst + ".sat01"), d)
    try:
        with open(os.path.join(d, "solve.log"), "w") as f:
            subprocess.run([tool("sat01"), inst + ".sat01"], cwd=d, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return ["timeout", "-"]
    out = open(os.path.join(d, inst + ".out")).read()
    m = re.search(r"(\d+) heuristic guesses", out)
    res = ["unsat" if "No solution" in out else "sat", m.group(1) if m else "0"]
    with open(rec, "w") as f:
        f.write(" ".join(res) + "\n")
    return res


def convert(work, inst, prep):
    """sat012clique -w, -p for the full preprocessing: the base path of the graph files,
    or 'solved' / 'refuted' when preprocessing decided the instance"""
    d = os.path.join(work, prep)
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(work, inst + ".sat01"), d)
    flags = ["-w"] + (["-p"] if prep == "full" else [])
    out = subprocess.run([tool("sat012clique"), *flags, inst + ".sat01"], cwd=d, capture_output=True, text=True).stdout
    if "solution has been found" in out:
        return "solved"
    if "no solution exists" in out:
        return "refuted"
    if not re.search(r"\d+ vertices, \d+ edges, required clique weight \d+", out):
        raise SystemExit("sat012clique on %s (%s): %s" % (inst, prep, out))
    return os.path.join(d, inst)


def load(base):
    adj = read_dimacs_bin(base + ".clq.b")
    w = np.loadtxt(base + ".w", ndmin=1)
    equ = [[int(v) - 1 for v in l.split()] for l in open(base + ".equ")]
    return adj, w, equ


def blind_share(adj, equ):
    """the share of contradiction pairs that share no equation"""
    n = len(adj)
    A = np.zeros((len(equ), n))
    for e, vs in enumerate(equ):
        A[e, vs] = 1
    pairs = ~adj & ~np.eye(n, dtype=bool)
    return ((A.T @ A == 0) & pairs).sum() / max(pairs.sum(), 1)


def wtheta(base, adj, w, threads):
    """weighted theta with CSDP: theta, lambda_max of the dual wrapper, rank of the optimum"""
    n = len(adj)
    z = np.sqrt(w)
    iu, ju = np.triu_indices(n, 1)
    non = ~adj[iu, ju]
    pi, pj = iu[non], ju[non]
    k = 1 + len(pi)
    prob, sol, log = base + ".dat-s", base + ".sol", base + ".csdp.log"
    with open(prob, "w") as f:
        f.write("%d\n1\n%d\n" % (k, n))
        f.write("1.0 " + "0.0 " * (k - 1) + "\n")
        for i in range(n):                                   # C = z z^T
            f.write("".join("0 1 %d %d %.17g\n" % (i + 1, j + 1, z[i] * z[j]) for j in range(i, n)))
        f.write("".join("1 1 %d %d 1.0\n" % (i, i) for i in range(1, n + 1)))
        f.write("".join("%d 1 %d %d 1.0\n" % (c + 2, i + 1, j + 1) for c, (i, j) in enumerate(zip(pi, pj))))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    with open(log, "w") as f:
        subprocess.run([T.CSDP, prob, sol], stdout=f, stderr=subprocess.STDOUT, env=env)
    out = open(log).read()
    pobj = float(re.search(r"Primal objective value:\s*(\S+)", out).group(1))
    dobj = float(re.search(r"Dual objective value:\s*(\S+)", out).group(1))
    H, X = dual_wrapper(base, adj, w)
    lmax = np.linalg.eigvalsh(H)[-1]
    return (pobj + dobj) / 2, lmax, rank(X)


def dual_wrapper(base, adj, w):
    """the wrapper of CSDP's dual solution (lambda_max(H) >= theta) and the Lovasz optimum X"""
    n = len(adj)
    iu, ju = np.triu_indices(n, 1)
    non = ~adj[iu, ju]
    pi, pj = iu[non], ju[non]
    y, Z, X = T.read_solution(base + ".sol", n, 1 + len(pi))
    z = np.sqrt(w)
    H = np.outer(z, z)
    H[pi, pj] -= y[1:]
    H[pj, pi] -= y[1:]
    return H, X


def solutions(adj, equ, cap):
    """the solutions (exact covers of the equations by pairwise non-contradicting
    vertices), depth first with the most constrained equation first, up to cap, and
    the number of search nodes"""
    n = len(adj)
    nbr = [sum(1 << int(j) for j in np.flatnonzero(adj[i])) for i in range(n)]
    eqmask = [sum(1 << v for v in vs) for vs in equ]
    eq_of = [[] for _ in range(n)]
    for e, vs in enumerate(equ):
        for v in vs:
            eq_of[v].append(e)
    sols, nodes = [], [0]

    def rec(allowed, uncovered, chosen):
        nodes[0] += 1
        if not uncovered:
            sols.append(list(chosen))
            return
        best = None
        for e in uncovered:
            c = eqmask[e] & allowed
            k = bin(c).count("1")
            if best is None or k < best[1]:
                best = c, k
                if k <= 1:
                    break
        c = best[0]
        while c and len(sols) < cap:
            v = (c & -c).bit_length() - 1
            c &= c - 1
            chosen.append(v)
            rec(allowed & nbr[v], uncovered.difference(eq_of[v]), chosen)
            chosen.pop()

    rec((1 << n) - 1, frozenset(range(len(equ))), [])
    return sols, nodes[0]


def rank(X):
    """the rank of a Lovasz optimum: its eigenvalues above 1e-5 of the largest; when theta >
    alpha they decay without a gap and the largest relative jump (theta.py's cluster) can
    stop after the first"""
    ev = np.linalg.eigvalsh(X)[::-1]
    return int((ev > 1e-5 * ev[0]).sum())


def span(inst, prep, cap):
    """for a satisfiable instance whose theta CSDP has solved: its solutions, their span,
    the rank of the Lovasz optimum, the top multiplicity of CSDP's optimal wrapper and of
    H_A (dim null A + 1), and how far the solutions are from being anchored at the top
    of CSDP's wrapper, H (z o 1_K) = m (z o 1_K), which Lovasz optimality forces"""
    base = os.path.join(RUNS, "work", inst, prep, inst)
    adj, w, equ = load(base)
    n, m = len(adj), len(equ)
    t0 = time.time()
    sols, nodes = solutions(adj, equ, cap)
    S = np.zeros((len(sols), n))
    for r, K in enumerate(sols):
        S[r, K] = 1
    H, X = dual_wrapper(base, adj, w)
    lam = np.linalg.eigvalsh(H)[::-1]
    top = int((lam[0] - lam <= 1e-5 * lam[0]).sum())     # CSDP splits a degenerate top by ~1e-6
    A = np.zeros((m, n))
    for e, vs in enumerate(equ):
        A[e, vs] = 1
    z = np.sqrt(w)
    resid = max((np.abs(H[:, K] @ z[K] - m * np.isin(np.arange(n), K) * z).max() for K in sols), default=np.nan)
    return [inst, prep, "%d%s" % (len(sols), "+" if len(sols) >= cap else ""), np.linalg.matrix_rank(S) if sols else 0,
            rank(X), top, n - np.linalg.matrix_rank(A) + 1, "%.1e" % resid, nodes,
            "%.0f" % (time.time() - t0)]


def qms(work, inst, config, threads):
    """sat01qms on the instance in one of CONFIGS: the fields of its RESULT line"""
    d = os.path.join(work, "qms-" + config)
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(work, inst + ".sat01"), d)
    flags, settings = CONFIGS[config]
    env = {k: v for k, v in os.environ.items() if not k.startswith("QMS_")}
    env.update(settings, OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    p = subprocess.run([tool("sat01qms"), *flags, inst + ".sat01"], cwd=d, env=env,
                       capture_output=True, text=True, check=True)
    out = p.stdout
    with open(os.path.join(d, "qms.log"), "w") as f:
        f.write(out + p.stderr)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    return dict(kv.split("=", 1) for kv in line.split()[2:])


def main():
    ap = argparse.ArgumentParser(description="weighted theta of SAT01 instances with CSDP")
    ap.add_argument("instances", nargs="+")
    ap.add_argument("-f", "--force", action="store_true", help="solve again what sat01.tsv has")
    ap.add_argument("--timeout", type=float, default=600, help="seconds for the SAT01 solver")
    ap.add_argument("--max-pairs", type=int, default=50000, help="skip theta above this many pairs")
    ap.add_argument("-t", "--threads", type=int, default=12, help="12 physical cores: hyperthreads slow the BLAS")
    ap.add_argument("--prep", choices=("light", "full"), help="only this preprocessing, instead of both")
    ap.add_argument("--span", action="store_true", help="the solution span pass instead (satisfiable instances)")
    ap.add_argument("--cap", type=int, default=20000, help="the most solutions --span lists")
    ap.add_argument("--qms", action="store_true", help="QUALEX-MS on the propagated instances instead")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    if args.qms:
        done = set()
        if os.path.exists(QMS_TSV):
            done = {tuple(l.split("\t")[0:4:3]) for l in open(QMS_TSV).read().splitlines()}
        for inst in args.instances:
            work = build(inst)
            ans, guesses = answer(work, inst, args.timeout)
            for config in CONFIGS:
                if (inst, config) in done and not args.force:
                    print("%-14s %-28s already in qms.tsv" % (inst, config))
                    continue
                r = qms(work, inst, config, args.threads)
                if "decided" in r:
                    row = [inst, ans, guesses, config, 0] + ["-"] * 7 + [r["propagation"].rstrip("s"), "-"]
                    print("%-14s %-28s decided by propagation (%s)" % (inst, config, r["decided"]), flush=True)
                else:
                    row = [inst, ans, guesses, config] + [r[k] for k in (
                        "n", "m", "preselected", "left", "greedy", "weight", "solution", "verified")] + [
                        r["propagation"].rstrip("s"), r["qms"].rstrip("s")]
                    print("%-14s %-28s n=%s m=%s left %s: Meta-NBIW %s, QUALEX-MS %s%s  (%ss; solver: %s, %s guesses)"
                          % (inst, config, r["n"], r["m"], r["left"], r["greedy"], r["weight"],
                             "  SOLUTION (verified %s)" % r["verified"] if r["solution"] == "1" else "",
                             r["qms"].rstrip("s"), ans, guesses), flush=True)
                with open(QMS_TSV, "a") as f:
                    f.write("\t".join(map(str, row)) + "\n")
        return
    if args.span:
        for inst in args.instances:
            for prep in [args.prep] if args.prep else ("light", "full"):
                if os.path.exists(os.path.join(RUNS, "work", inst, prep, inst + ".sol")):
                    row = span(inst, prep, args.cap)
                    print("%-14s %-5s %s solutions spanning %d, rank of the Lovasz optimum %d, top multiplicity %d"
                          " (H_A: %d), anchored at the top to %s  (%d nodes, %ss)" % tuple(row), flush=True)
                    with open(os.path.join(RUNS, "span.tsv"), "a") as f:
                        f.write("\t".join(map(str, row)) + "\n")
        return
    done = set()
    if os.path.exists(TSV):
        done = {tuple(l.split("\t")[0:4:3]) for l in open(TSV).read().splitlines()}
    for inst in args.instances:
        work = build(inst)
        ans, guesses = answer(work, inst, args.timeout)
        for prep in [args.prep] if args.prep else ("light", "full"):
            if (inst, prep) in done and not args.force:
                print("%-14s %-5s already in sat01.tsv" % (inst, prep))
                continue
            t0 = time.time()
            base = convert(work, inst, prep)
            if base in ("solved", "refuted"):
                row = [inst, ans, guesses, prep, 0] + ["-"] * 8
                print("%-14s %-5s %s by preprocessing (solver: %s, %s guesses)" % (inst, prep, base, ans, guesses),
                      flush=True)
            else:
                adj, w, equ = load(base)
                n, m = len(adj), len(equ)
                pairs = (n * (n - 1) - int(adj.sum())) // 2
                row = [inst, ans, guesses, prep, n, m, pairs, "%.3f" % blind_share(adj, equ)]
                if pairs > args.max_pairs:
                    row += ["-"] * 5
                    print("%-14s %-5s n=%d m=%d pairs=%d: over --max-pairs, theta skipped" % (inst, prep, n, m, pairs),
                          flush=True)
                else:
                    th, lmax, rk = wtheta(base, adj, w, args.threads)
                    row += ["%.7f" % th, "%.7f" % lmax, "%.2e" % (m - th), rk, "%.0f" % (time.time() - t0)]
                    print("%-14s %-5s n=%d m=%d pairs=%d blind %s  theta %.7f  (wrapper lambda_max %.7f)  m - theta %.2e"
                          "  rank %d  solver: %s, %s guesses" % (inst, prep, n, m, pairs, row[7], th, lmax, m - th,
                                                                 rk, ans, guesses), flush=True)
            with open(TSV, "a") as f:
                f.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
