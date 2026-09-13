#!/usr/bin/env python3
"""Compare solver variants on one benchmark suite, from bench/runs results.

usage: bench/compare.py [--run DIR]... [--base NAME] [--pairs A:B,...]
                        [--list] [--diag] [--moves] SUITE VARIANTS

  SUITE      dimacs, dimacs-big, ru, rw, ru2000 or rw2000
  VARIANTS   comma-separated; the baseline is --base, else head if listed,
             else the first
  --run DIR  a run directory (default: the one under bench/runs with the most
             recent results); repeat it to take variants from several runs,
             the first directory holding a variant winning
  --pairs    head-to-head counts, e.g. --pairs lzw:lzws,ancw:lzw
  --list     the graphs where the variants differ (always on for DIMACS)
  --diag     what the ANCHOR, THM8, PASS and ICE lines in the logs say
  --moves    for every improvement over the baseline, how many vertices the
             clique dropped and gained (from the .sol files)

The reference value of a graph is its best known clique on DIMACS
(bench/dimacs_best.tsv), its cliquer optimum where bench/optima records one,
and otherwise the best value any listed variant found.  When head is listed,
its results are checked against benchmarks.md.
"""
import argparse
import glob
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)
EPS = 1e-9
RANDOM_MODE = {"ru": "u", "rw": "w", "ru2000": "u", "rw2000": "w"}
REPS = {2000: 2}  # graphs per (n, p) cell, as tools/random_graphs.py makes them
DENSITIES = ("p<=0.5", "p=0.7", "p>=0.8")

NUM = r"([-+0-9.eEinfa]+)"
ANCHOR = re.compile(r"ANCHOR \|Q\|=\d+ W\(Q\)=\S+ C_i=\S+ theta=\S+ t_min=%s touched=(\d+)" % NUM)
THM8 = re.compile(r"THM8 \|Q\|=\d+ W\(Q\)=\S+ att=\S+ mu\*=\S+ lam_max=\S+ above=(\d+)/(\d+) "
                  r"maxdev=%s \(indicator entries ~%s\)" % (NUM, NUM))
PASS = re.compile(r"PASS \d+ anchored=(\d) %s -> %s" % (NUM, NUM))
ICE = re.compile(r"ICE mode=\w+ cluster=\d+ objective %s -> %s sigma=%s changed=\d+ t_min=\S+ "
                 r"projected \S+ -> \S+ spread/\|hatb\| %s -> %s" % ((NUM,) * 5))


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def read_values(path):
    """'name value ...' lines; comments and non-numeric values are skipped"""
    out = {}
    if os.path.exists(path):
        for line in open(path):
            fields = line.split("#", 1)[0].split()
            if len(fields) >= 2:
                try:
                    out[fields[0]] = float(fields[1])
                except ValueError:
                    pass
    return out


def default_run():
    best = None
    for run in glob.glob(os.path.join(BENCH, "runs", "*")):
        files = glob.glob(os.path.join(run, "res", "*", "*.res"))
        if files:
            t = max(os.path.getmtime(f) for f in files)
            if best is None or t > best[0]:
                best = (t, run)
    if best is None:
        sys.exit("no results under " + os.path.join(BENCH, "runs"))
    return best[1]


def load(runs, suite, variant):
    for run in runs:
        files = glob.glob(os.path.join(run, "res", suite, variant + ".*.res"))
        if files:
            res = {}
            for f in files:
                for line in open(f):
                    fields = line.split()
                    if len(fields) >= 3:
                        res[fields[0]] = (float(fields[1]), float(fields[2]))
            return run, res
    return None, {}


def cell(name):
    m = re.match(r"g(\d+)_(\d+)_\d+$", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def density(name):
    p = cell(name)[1]
    return DENSITIES[0] if p <= 50 else (DENSITIES[1] if p == 70 else DENSITIES[2])


def recorded_head(suite):
    """what benchmarks.md records for head (c430841): per graph on DIMACS, per
    (n, p) cell mean on the random graphs"""
    path = os.path.join(REPO, "benchmarks.md")
    if not os.path.exists(path):
        return {}
    lines = open(path).read().splitlines()
    if suite in ("dimacs", "dimacs-big"):
        row = re.compile(r"\|\s*([A-Za-z][\w.\-]*)\s*\|\s*\d+\s*\|\s*(\d+)[^|]*\|\s*\d+\s*\|\s*[\d.]+%\s*\|")
        return {m.group(1): float(m.group(2)) for m in map(row.match, lines) if m}
    mode, cells, section = RANDOM_MODE[suite], {}, None
    for line in lines:
        if line.startswith("#"):
            section = {"### Unweighted": "u", "### Weighted": "w"}.get(line.strip())
        elif section == mode and line.startswith("| "):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if cols[0].isdigit():
                cells[(int(cols[0]), int(round(100 * float(cols[1]))))] = \
                    float(cols[5] if mode == "u" else cols[4])
    return cells


def check_head(suite, head):
    rec = recorded_head(suite)
    if not rec:
        return
    if suite in ("dimacs", "dimacs-big"):
        names = [nm for nm in head if nm in rec]
        bad = [nm for nm in names if abs(rec[nm] - head[nm][0]) > EPS]
        print("head against benchmarks.md: %d graphs, %d differ%s" % (
            len(names), len(bad), ": " + ", ".join(bad) if bad else ""))
        return
    cells = defaultdict(list)
    for nm, (value, _) in head.items():
        if cell(nm):
            cells[cell(nm)].append(value)
    full = {c: v for c, v in cells.items() if c in rec and len(v) == REPS.get(c[0], 5)}
    bad = ["n=%d p=%.2f: %.2f against %.1f" % (c[0], c[1] / 100.0, statistics.mean(v), rec[c])
           for c, v in sorted(full.items()) if abs(statistics.mean(v) - rec[c]) > 0.0501]
    print("head against benchmarks.md: %d cell means, %d differ%s" % (
        len(full), len(bad), ": " + "; ".join(bad) if bad else ""))


def references(suite, names, results):
    """the reference value of each graph, and the graphs whose reference is known
    independently of the variants (best known on DIMACS, cliquer optimum)"""
    found = {nm: max(r[nm][0] for r in results.values()) for nm in names}
    if suite in ("dimacs", "dimacs-big"):
        best = read_values(os.path.join(BENCH, "dimacs_best.tsv"))
        return {nm: best.get(nm, found[nm]) for nm in names}, {nm for nm in names if nm in best}
    optima = read_values(os.path.join(BENCH, "optima", RANDOM_MODE[suite] + ".tsv"))
    ref, proven = {}, set()
    for nm in names:
        if nm in optima:
            if found[nm] > optima[nm] + EPS:
                print("warning: %s: %g found, above the recorded optimum %g" % (nm, found[nm], optima[nm]))
            ref[nm] = optima[nm]
            proven.add(nm)
        else:
            ref[nm] = found[nm]
    return ref, proven


def sol_vertices(path):
    if not os.path.exists(path):
        return None
    return {int(line[1:]) for line in open(path) if line.startswith("v")}


def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("suite")
    ap.add_argument("variants")
    ap.add_argument("--run", action="append")
    ap.add_argument("--base")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--moves", action="store_true")
    args = ap.parse_args()
    suite = args.suite
    if suite not in ("dimacs", "dimacs-big") and suite not in RANDOM_MODE:
        sys.exit("unknown suite " + suite)
    random_suite = suite in RANDOM_MODE

    runs = args.run or [default_run()]
    variants = [v for v in args.variants.split(",") if v]
    base = args.base or ("head" if "head" in variants else variants[0])
    pairs = [p.split(":") for p in args.pairs.split(",") if p]
    for v in [base] + [x for p in pairs for x in p]:
        if v not in variants:
            variants.append(v)

    run_of, results = {}, {}
    for v in variants:
        run_of[v], results[v] = load(runs, suite, v)
        if not results[v]:
            sys.exit("no results for %s on %s in %s" % (v, suite, " ".join(runs)))
    names = sorted(set.intersection(*(set(r) for r in results.values())))
    print("%s from %s" % (suite, ", ".join(sorted({os.path.relpath(r) for r in run_of.values()}))))
    print("%d graphs with every variant (%s)" % (
        len(names), ", ".join("%s %d" % (v, len(results[v])) for v in variants)))
    if "head" in results:
        check_head(suite, results["head"])
    if not names:
        return
    ref, proven = references(suite, names, results)

    def val(v, nm):
        return results[v][nm][0]

    print()
    header = "%-8s %9s" % ("variant", "at ref")
    if random_suite:
        header += " %9s" % "proven"
    print(header + " %9s %9s %6s %13s" % ("mean gap", "time (s)", "x base", "vs base +/-"))
    t_base = sum(results[base][nm][1] for nm in names)
    for v in variants:
        line = "%-8s %4d/%-4d" % (v, sum(val(v, nm) >= ref[nm] - EPS for nm in names), len(names))
        if random_suite:
            line += " %4d/%-4d" % (sum(val(v, nm) >= ref[nm] - EPS for nm in proven), len(proven))
        t = sum(results[v][nm][1] for nm in names)
        line += " %8.3f%% %9.0f %6.2f %8d/%-4d" % (
            statistics.mean(100.0 * (ref[nm] - val(v, nm)) / ref[nm] for nm in names), t,
            t / t_base if t_base else float("nan"),
            sum(val(v, nm) > val(base, nm) + EPS for nm in names),
            sum(val(v, nm) < val(base, nm) - EPS for nm in names))
        print(line)

    def head_to_head(a, b):
        wins = sum(val(a, nm) > val(b, nm) + EPS for nm in names)
        losses = sum(val(a, nm) < val(b, nm) - EPS for nm in names)
        text = "%-8s vs %-8s better %3d, worse %3d" % (a, b, wins, losses)
        if random_suite:
            by = defaultdict(lambda: [0, 0])
            for nm in names:
                by[density(nm)][0] += val(a, nm) > val(b, nm) + EPS
                by[density(nm)][1] += val(a, nm) < val(b, nm) - EPS
            text += "  (%s)" % ", ".join("%s %d/%d" % (d, by[d][0], by[d][1]) for d in DENSITIES)
        return text

    comparisons = [(v, base) for v in variants if v != base and random_suite] + pairs
    if comparisons:
        print()
        for a, b in comparisons:
            print(head_to_head(a, b))

    if args.list or not random_suite:
        differ = [nm for nm in names
                  if max(val(v, nm) for v in variants) - min(val(v, nm) for v in variants) > EPS]
        if differ:
            print("\ngraphs where the variants differ (* = reference is the best found here):")
            print("%-18s %8s " % ("graph", "ref") + " ".join("%8s" % v for v in variants))
            for nm in differ:
                print("%-18s %8s " % (nm, "%g%s" % (ref[nm], "" if nm in proven else "*")) +
                      " ".join("%8g" % val(v, nm) for v in variants))

    if args.diag:
        print()
        for v in variants:
            anchors, exact, passes, ice = [], [], [], []
            for nm in names:
                path = os.path.join(run_of[v], "log", suite, v, nm + ".out")
                if not os.path.exists(path):
                    continue
                text = open(path, errors="replace").read()
                found = ANCHOR.findall(text)
                if found:
                    anchors.append((float(found[0][0]), int(found[0][1])))
                # a pass prints its THM8 line before its PASS line, and only an
                # anchored pass meets Theorem 8 by construction
                run_passes, previous = [], 0
                for m in PASS.finditer(text):
                    if m.group(1) == "1":
                        for above, k, dev, entry in THM8.findall(text, previous, m.start()):
                            if float(dev) >= 0:
                                exact.append((int(above), int(k), float(dev) / float(entry)))
                    run_passes.append((float(m.group(2)), float(m.group(3))))
                    previous = m.end()
                if run_passes:
                    passes.append(run_passes)
                found = ICE.findall(text)
                if found:
                    ice.append(tuple(map(float, found[0])))
            lines = []
            if anchors:
                lines.append("anchored %d runs; t_min median %.3g, least %.3g; entries touched median %d" % (
                    len(anchors), median([a[0] for a in anchors]), min(a[0] for a in anchors),
                    median([a[1] for a in anchors])))
            if exact:
                lines.append("anchored passes meeting Theorem 8 %d, on the outer branch %d; eigenvalues above "
                             "mu* median %g of %g; worst |x - indicator| / entry %.2g" % (
                                 len(exact), sum(e[0] == 0 for e in exact), median([e[0] for e in exact]),
                                 median([e[1] for e in exact]), max(e[2] for e in exact)))
            if passes:
                hist = Counter(len(p) for p in passes)
                summary = "passes per run %s" % dict(sorted(hist.items()))
                if max(hist) >= 2:
                    summary += "; the second pass improved in %d" % sum(
                        len(p) >= 2 and p[1][1] > p[1][0] + EPS for p in passes)
                if max(hist) >= 3:
                    summary += "; later passes improved further in %d, the last still improving in %d" % (
                        sum(len(p) >= 3 and p[-1][1] > p[1][1] + EPS for p in passes),
                        sum(len(p) == max(hist) and p[-1][1] > p[-1][0] + EPS for p in passes))
                lines.append(summary)
            if ice:
                ratios = [x[4] / x[3] for x in ice if 0 < x[3] < 1e12 and 0 < x[4] < 1e12]
                lines.append("lambda_max step taken %d of %d; objective x%.3f median; spread/|hatb| x%.3f "
                             "median (%d finite)" % (
                                 sum(x[2] > 0 for x in ice), len(ice),
                                 median([x[1] / x[0] for x in ice if x[0] > 0]), median(ratios), len(ratios)))
            if lines:
                print("%s: %s" % (v, ("\n" + " " * (len(v) + 2)).join(lines)))

    if args.moves:
        print()
        for v in variants:
            if v == base:
                continue
            rows = []
            for nm in names:
                if val(v, nm) > val(base, nm) + EPS:
                    qa = sol_vertices(os.path.join(run_of[base], "work", suite, base, nm + ".sol"))
                    qb = sol_vertices(os.path.join(run_of[v], "work", suite, v, nm + ".sol"))
                    if qa and qb:
                        rows.append((len(qa), len(qa - qb), len(qb - qa)))
            if rows:
                print("%s over %s: %d improvements; dropped median %g (most %d), gained median %g "
                      "(most %d), share of the %s clique kept median %.2f" % (
                          v, base, len(rows), median([r[1] for r in rows]), max(r[1] for r in rows),
                          median([r[2] for r in rows]), max(r[2] for r in rows), base,
                          median([1 - r[1] / r[0] for r in rows])))


if __name__ == "__main__":
    main()
