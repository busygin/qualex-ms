# DIMACS Benchmark Results

QUALEX-MS performance on 80 DIMACS maximum clique benchmark instances.

`before` is the method as published: one trust region radius, taken from
Proposition 7 applied to the clique the greedy stage found, and eigenvector
directions only where the linear form vanishes.  `after` adds the multiplier
selection described in `qualex.cc` -- a geometric scan of radii around that
anchor, a Meta-NBIW pass at the two multipliers the scan ranked highest, and
the eigenvector directions of every cluster rather than only the degenerate
ones.

| Benchmark | before | after | Best Known | % Missing |
|-----------|--------|-------|------------|-----------|
| brock200_1 | 21 | 21 | 21 | 0.0% |
| brock200_2 | 12 | 12 | 12 | 0.0% |
| brock200_3 | 15 | 15 | 15 | 0.0% |
| brock200_4 | 17 | 17 | 17 | 0.0% |
| brock400_1 | 27 | 27 | 27 | 0.0% |
| brock400_2 | 29 | 29 | 29 | 0.0% |
| brock400_3 | 31 | 31 | 31 | 0.0% |
| brock400_4 | 33 | 33 | 33 | 0.0% |
| brock800_1 | 23 | 23 | 23 | 0.0% |
| brock800_2 | 24 | 24 | 24 | 0.0% |
| brock800_3 | 25 | 25 | 25 | 0.0% |
| brock800_4 | 26 | 26 | 26 | 0.0% |
| C125.9 | 34 | 34 | 34 | 0.0% |
| C250.9 | 44 | 44 | 44 | 0.0% |
| C500.9 | 55 | 55 | 57 | 3.5% |
| C1000.9 | 64 | 64 | 68 | 5.9% |
| C2000.5 | 16 | 16 | 16 | 0.0% |
| C2000.9 | 72 | 72 | 80 | 10.0% |
| C4000.5 | 17 | 17 | 18 | 5.6% |
| c-fat200-1 | 12 | 12 | 12 | 0.0% |
| c-fat200-2 | 24 | 24 | 24 | 0.0% |
| c-fat200-5 | 58 | 58 | 58 | 0.0% |
| c-fat500-1 | 14 | 14 | 14 | 0.0% |
| c-fat500-2 | 26 | 26 | 26 | 0.0% |
| c-fat500-5 | 64 | 64 | 64 | 0.0% |
| c-fat500-10 | 126 | 126 | 126 | 0.0% |
| DSJC500.5 | 13 | 13 | 13 | 0.0% |
| DSJC1000.5 | 14 | 14 | 15 | 6.7% |
| gen200_p0.9_44 | 42 | 44 **+2** | 44 | 0.0% |
| gen200_p0.9_55 | 55 | 55 | 55 | 0.0% |
| gen400_p0.9_55 | 51 | 53 **+2** | 55 | 3.6% |
| gen400_p0.9_65 | 65 | 65 | 65 | 0.0% |
| gen400_p0.9_75 | 75 | 75 | 75 | 0.0% |
| hamming6-2 | 32 | 32 | 32 | 0.0% |
| hamming6-4 | 4 | 4 | 4 | 0.0% |
| hamming8-2 | 128 | 128 | 128 | 0.0% |
| hamming8-4 | 16 | 16 | 16 | 0.0% |
| hamming10-2 | 512 | 512 | 512 | 0.0% |
| hamming10-4 | 40 | 40 | 40 | 0.0% |
| johnson8-2-4 | 4 | 4 | 4 | 0.0% |
| johnson8-4-4 | 14 | 14 | 14 | 0.0% |
| johnson16-2-4 | 8 | 8 | 8 | 0.0% |
| johnson32-2-4 | 16 | 16 | 16 | 0.0% |
| keller4 | 11 | 11 | 11 | 0.0% |
| keller5 | 26 | 26 | 27 | 3.7% |
| keller6 | 52 | 53 **+1** | 59 | 10.2% |
| MANN_a9 | 16 | 16 | 16 | 0.0% |
| MANN_a27 | 125 | 126 **+1** | 126 | 0.0% |
| MANN_a45 | 342 | 342 | 345 | 0.9% |
| MANN_a81 | 1096 | 1096 | 1100 | 0.4% |
| p_hat300-1 | 8 | 8 | 8 | 0.0% |
| p_hat300-2 | 25 | 25 | 25 | 0.0% |
| p_hat300-3 | 35 | 36 **+1** | 36 | 0.0% |
| p_hat500-1 | 9 | 9 | 9 | 0.0% |
| p_hat500-2 | 36 | 36 | 36 | 0.0% |
| p_hat500-3 | 48 | 49 **+1** | 50 | 2.0% |
| p_hat700-1 | 11 | 11 | 11 | 0.0% |
| p_hat700-2 | 44 | 44 | 44 | 0.0% |
| p_hat700-3 | 62 | 62 | 62 | 0.0% |
| p_hat1000-1 | 10 | 10 | 10 | 0.0% |
| p_hat1000-2 | 45 | 46 **+1** | 46 | 0.0% |
| p_hat1000-3 | 65 | 66 **+1** | 68 | 2.9% |
| p_hat1500-1 | 12 | 12 | 12 | 0.0% |
| p_hat1500-2 | 64 | 65 **+1** | 65 | 0.0% |
| p_hat1500-3 | 91 | 93 **+2** | 94 | 1.1% |
| san200_0.7_1 | 30 | 30 | 30 | 0.0% |
| san200_0.7_2 | 18 | 18 | 18 | 0.0% |
| san200_0.9_1 | 70 | 70 | 70 | 0.0% |
| san200_0.9_2 | 60 | 60 | 60 | 0.0% |
| san200_0.9_3 | 40 | 44 **+4** | 44 | 0.0% |
| san400_0.5_1 | 13 | 13 | 13 | 0.0% |
| san400_0.7_1 | 40 | 40 | 40 | 0.0% |
| san400_0.7_2 | 30 | 30 | 30 | 0.0% |
| san400_0.7_3 | 18 | 22 **+4** | 22 | 0.0% |
| san400_0.9_1 | 100 | 100 | 100 | 0.0% |
| san1000 | 15 | 15 | 15 | 0.0% |
| sanr200_0.7 | 18 | 18 | 18 | 0.0% |
| sanr200_0.9 | 41 | 41 | 42 | 2.4% |
| sanr400_0.5 | 13 | 13 | 13 | 0.0% |
| sanr400_0.7 | 20 | 20 | 21 | 4.8% |

## Summary

|  | before | after |
|--|--------|-------|
| Optimal / best known | 58/80 | 65/80 |
| Average % missing | 1.42% | 0.79% |
| Improved instances | -- | 12 |
| Regressions | -- | 0 |

Total wall time over the suite roughly doubles (495 s to 1107 s on an RTX 2080 Ti
host), essentially all of it in the two added Meta-NBIW passes, which cost the
same O(n^3) as the Meta-NBIW the solver already runs once on the plain vertex
weights.  `QMS_META_N=0` switches that stage off; on the 70 instances outside
the largest ten, the radius scan on its own then still improves 2 of them
(`gen200_p0.9_44` 42 -> 43, `p_hat300-3` 35 -> 36) for 79 s against the
original 78 s.  Everything beyond those two needs the Meta-NBIW stage.

### Instances improved

| Benchmark | before | after | Best Known |
|-----------|--------|-------|------------|
| gen200_p0.9_44 | 42 | 44 | 44 |
| gen400_p0.9_55 | 51 | 53 | 55 |
| keller6 | 52 | 53 | 59 |
| MANN_a27 | 125 | 126 | 126 |
| p_hat300-3 | 35 | 36 | 36 |
| p_hat500-3 | 48 | 49 | 50 |
| p_hat1000-2 | 45 | 46 | 46 |
| p_hat1000-3 | 65 | 66 | 68 |
| p_hat1500-2 | 64 | 65 | 65 |
| p_hat1500-3 | 91 | 93 | 94 |
| san200_0.9_3 | 40 | 44 | 44 |
| san400_0.7_3 | 18 | 22 | 22 |

Of these, 7 reach the best known value for the first time: `gen200_p0.9_44`, `MANN_a27`, `p_hat300-3`, `p_hat1000-2`, `p_hat1500-2`, `san200_0.9_3`, `san400_0.7_3`.

### Remaining hardest instances

| Benchmark | % Missing |
|-----------|-----------|
| keller6 | 10.2% |
| C2000.9 | 10.0% |
| DSJC1000.5 | 6.7% |
| C1000.9 | 5.9% |
| C4000.5 | 5.6% |
| sanr400_0.7 | 4.8% |
| keller5 | 3.7% |
| gen400_p0.9_55 | 3.6% |

## Weighted instances

There is no standard weighted test suite, so, following the protocol of the
QUALEX-MS paper, 56 random graphs were generated (n = 150 and 200, edge
densities 0.2 to 0.9, `normal` and `irregular` construction, integer vertex
weights drawn uniformly from 1..10) and solved exactly with `cliquer` for
reference.

|  | before | after |
|--|--------|-------|
| Solved exactly | 32/56 | 39/56 |
| Average % of optimum | 98.63% | 99.44% |
| Regressions | -- | 0 |

## Uniform random graphs

The DIMACS graphs are a fixed collection from the 1993 challenge, most of it
structured or built around planted cliques.  To check that the changes are not
tuned to it, the comparison was repeated on uniform random graphs G(n,p), in
which every pair of vertices is joined independently with probability p.
`tools/random_graphs.py` generates the set: five graphs for each of n = 200,
300, 500 and 1000 and two for n = 2000, at each of p = 0.25, 0.5, 0.7, 0.8, 0.9
and 0.95, 132 graphs in all.  Each was solved once unweighted and once with
integer vertex weights drawn uniformly from 1..10.  A graph's seed depends only
on its (n, p, replicate), so any one of them can be regenerated on its own.

`before` is `8a185a1` and `after` is `c430841`.  The reference is the optimum
where `cliquer` 1.21 proved it -- 50 unweighted graphs (n = 200 with p <= 0.8,
n = 300 with p <= 0.7, n = 500 with p <= 0.5, n = 1000 with p = 0.25) and 70
weighted ones (n = 200 with p <= 0.9, n = 300 with p <= 0.8, n = 500 with
p <= 0.7, n = 1000 with p <= 0.5) -- and elsewhere the best clique found by any
of the five versions in the attribution table below, marked `*`, which only
bounds the optimum from below.  k1 is the largest k for which the expected
number of k-cliques is at least 1, the first moment estimate of the clique
number.  Values are means over the graphs of a cell; `at reference` counts the
graphs whose clique reaches the reference, and `better` those where `after`
found a larger clique than `before` -- it never found a smaller one.  Times are
wall clock per graph, including about half a second of CUDA start-up.  All
versions ran back to back on each graph on the same RTX 2080 Ti, so the ratios
are paired, but `cliquer` was loading the CPU at the same time, which inflates
the absolute figures.

### Unweighted

| n | p | k1 | reference | before | after | at reference | better | time per graph (s) |
|---|---|----|-----------|--------|-------|--------------|--------|--------------------|
| 200 | 0.25 | 6 | 6.4 | 6.4 | 6.4 | 5 -> 5 | 0 | 0.6 -> 0.6 |
| 200 | 0.50 | 11 | 11.0 | 11.0 | 11.0 | 5 -> 5 | 0 | 0.6 -> 0.7 |
| 200 | 0.70 | 18 | 17.6 | 17.6 | 17.6 | 5 -> 5 | 0 | 0.6 -> 0.7 |
| 200 | 0.80 | 26 | 25.4 | 24.8 | 24.8 | 3 -> 3 | 0 | 0.6 -> 0.7 |
| 200 | 0.90 | 44 | 41.2* | 40.8 | 41.2 | 3 -> 5 | 2 | 0.6 -> 0.6 |
| 200 | 0.95 | 70 | 63.0* | 62.4 | 63.0 | 3 -> 5 | 2 | 0.6 -> 0.6 |
| 300 | 0.25 | 7 | 7.0 | 7.0 | 7.0 | 5 -> 5 | 0 | 0.7 -> 0.8 |
| 300 | 0.50 | 12 | 11.8 | 11.4 | 11.8 | 3 -> 5 | 2 | 0.7 -> 0.9 |
| 300 | 0.70 | 20 | 20.0 | 19.4 | 19.4 | 2 -> 2 | 0 | 0.6 -> 0.8 |
| 300 | 0.80 | 29 | 28.2* | 28.0 | 28.2 | 4 -> 5 | 1 | 0.6 -> 0.8 |
| 300 | 0.90 | 50 | 46.8* | 45.8 | 46.8 | 1 -> 5 | 4 | 0.6 -> 0.7 |
| 300 | 0.95 | 82 | 74.4* | 73.4 | 74.4 | 2 -> 5 | 3 | 0.6 -> 0.7 |
| 500 | 0.25 | 8 | 7.8 | 7.8 | 7.8 | 5 -> 5 | 0 | 1.0 -> 1.7 |
| 500 | 0.50 | 13 | 13.2 | 13.0 | 13.0 | 4 -> 4 | 0 | 1.1 -> 2.1 |
| 500 | 0.70 | 23 | 21.6* | 21.4 | 21.6 | 4 -> 5 | 1 | 0.9 -> 1.5 |
| 500 | 0.80 | 33 | 31.2* | 31.0 | 31.0 | 4 -> 4 | 0 | 0.9 -> 1.3 |
| 500 | 0.90 | 58 | 54.8* | 54.0 | 54.6 | 2 -> 4 | 3 | 0.7 -> 1.0 |
| 500 | 0.95 | 98 | 88.4* | 87.4 | 88.4 | 1 -> 5 | 4 | 0.7 -> 1.1 |
| 1000 | 0.25 | 8 | 8.4 | 8.0 | 8.0 | 3 -> 3 | 0 | 3.6 -> 7.5 |
| 1000 | 0.50 | 15 | 14.4* | 14.2 | 14.4 | 4 -> 5 | 1 | 4.6 -> 9.9 |
| 1000 | 0.70 | 26 | 24.4* | 24.2 | 24.2 | 4 -> 4 | 0 | 4.2 -> 9.9 |
| 1000 | 0.80 | 38 | 35.6* | 35.0 | 35.0 | 2 -> 2 | 0 | 4.3 -> 8.9 |
| 1000 | 0.90 | 69 | 63.2* | 62.6 | 63.2 | 2 -> 5 | 3 | 3.3 -> 6.4 |
| 1000 | 0.95 | 119 | 110.0* | 108.0 | 109.2 | 2 -> 4 | 3 | 2.6 -> 5.3 |
| 2000 | 0.25 | 9 | 9.0* | 9.0 | 9.0 | 2 -> 2 | 0 | 21.5 -> 52.0 |
| 2000 | 0.50 | 17 | 15.0* | 15.0 | 15.0 | 2 -> 2 | 0 | 32.1 -> 75.6 |
| 2000 | 0.70 | 29 | 27.5* | 27.0 | 27.0 | 1 -> 1 | 0 | 25.3 -> 57.7 |
| 2000 | 0.80 | 43 | 39.5* | 39.5 | 39.5 | 2 -> 2 | 0 | 19.5 -> 45.1 |
| 2000 | 0.90 | 79 | 72.5* | 72.0 | 72.0 | 1 -> 1 | 0 | 12.7 -> 29.8 |
| 2000 | 0.95 | 141 | 127.0* | 125.0 | 125.5 | 0 -> 0 | 1 | 9.1 -> 23.4 |

### Weighted

| n | p | reference | before | after | at reference | better | time per graph (s) |
|---|---|-----------|--------|-------|--------------|--------|--------------------|
| 200 | 0.25 | 48.0 | 47.8 | 48.0 | 4 -> 5 | 1 | 0.5 -> 0.5 |
| 200 | 0.50 | 81.4 | 81.2 | 81.2 | 4 -> 4 | 0 | 0.5 -> 0.6 |
| 200 | 0.70 | 128.0 | 124.6 | 126.4 | 1 -> 3 | 3 | 0.5 -> 0.6 |
| 200 | 0.80 | 182.6 | 175.8 | 180.2 | 0 -> 1 | 5 | 0.6 -> 0.6 |
| 200 | 0.90 | 278.2 | 267.4 | 273.2 | 0 -> 0 | 5 | 0.6 -> 0.6 |
| 200 | 0.95 | 412.0* | 405.6 | 412.0 | 0 -> 5 | 5 | 0.5 -> 0.6 |
| 300 | 0.25 | 54.6 | 54.6 | 54.6 | 5 -> 5 | 0 | 0.6 -> 0.7 |
| 300 | 0.50 | 88.8 | 84.6 | 87.0 | 0 -> 3 | 3 | 0.6 -> 0.8 |
| 300 | 0.70 | 143.0 | 138.6 | 140.0 | 0 -> 1 | 2 | 0.6 -> 0.8 |
| 300 | 0.80 | 200.4 | 187.8 | 193.2 | 0 -> 1 | 4 | 0.6 -> 0.7 |
| 300 | 0.90 | 299.0* | 290.6 | 298.6 | 0 -> 4 | 5 | 0.5 -> 0.6 |
| 300 | 0.95 | 478.0* | 469.0 | 477.4 | 1 -> 4 | 4 | 0.6 -> 0.6 |
| 500 | 0.25 | 59.0 | 58.6 | 58.6 | 4 -> 4 | 0 | 0.9 -> 1.4 |
| 500 | 0.50 | 99.6 | 96.0 | 96.6 | 0 -> 1 | 3 | 1.0 -> 1.7 |
| 500 | 0.70 | 164.8 | 153.6 | 156.8 | 0 -> 0 | 5 | 0.9 -> 1.4 |
| 500 | 0.80 | 224.0* | 212.2 | 223.6 | 0 -> 4 | 5 | 0.8 -> 1.3 |
| 500 | 0.90 | 375.2* | 357.2 | 371.8 | 0 -> 3 | 5 | 0.7 -> 1.0 |
| 500 | 0.95 | 579.2* | 557.2 | 577.2 | 0 -> 3 | 5 | 0.6 -> 0.9 |
| 1000 | 0.25 | 67.4 | 64.8 | 66.6 | 1 -> 3 | 3 | 3.3 -> 6.8 |
| 1000 | 0.50 | 116.6 | 109.0 | 113.4 | 0 -> 2 | 4 | 4.8 -> 9.8 |
| 1000 | 0.70 | 186.4* | 178.4 | 186.0 | 0 -> 4 | 5 | 4.0 -> 8.0 |
| 1000 | 0.80 | 262.8* | 250.8 | 262.8 | 0 -> 5 | 5 | 3.1 -> 6.2 |
| 1000 | 0.90 | 450.6* | 431.2 | 448.6 | 0 -> 2 | 5 | 2.3 -> 4.4 |
| 1000 | 0.95 | 732.2* | 698.0 | 730.8 | 0 -> 4 | 5 | 1.9 -> 3.6 |
| 2000 | 0.25 | 73.0* | 73.0 | 73.0 | 2 -> 2 | 0 | 22.0 -> 48.4 |
| 2000 | 0.50 | 120.5* | 118.0 | 120.5 | 0 -> 2 | 2 | 34.2 -> 74.2 |
| 2000 | 0.70 | 210.0* | 194.0 | 210.0 | 0 -> 2 | 2 | 27.9 -> 58.2 |
| 2000 | 0.80 | 296.0* | 289.0 | 293.0 | 1 -> 1 | 1 | 21.1 -> 44.6 |
| 2000 | 0.90 | 530.0* | 502.0 | 530.0 | 0 -> 2 | 2 | 14.8 -> 31.2 |
| 2000 | 0.95 | 890.5* | 839.0 | 889.5 | 0 -> 1 | 2 | 12.0 -> 24.4 |

### Summary

|  | unweighted before | unweighted after | weighted before | weighted after |
|--|-------------------|------------------|-----------------|----------------|
| At reference | 86/132 | 113/132 | 23/132 | 81/132 |
| Proven optimal | 40/50 | 42/50 | 19/70 | 33/70 |
| Average % below reference | 1.25% | 0.64% | 3.53% | 1.05% |
| Average % below proven optimum | 1.45% | 1.12% | 3.31% | 1.77% |
| Improved graphs | -- | 30 | -- | 96 |
| Regressions | -- | 0 | -- | 0 |
| Total wall time | 417 s | 894 s | 420 s | 832 s |

Where `after` improves, it does so by 2.8% on average unweighted and by 3.6%
weighted.  The price grows with n: `after` takes 1.06, 1.23, 1.64, 2.12 and
2.36 times as long as `before` at n = 200, 300, 500, 1000 and 2000 unweighted,
and 1.04, 1.17, 1.56, 1.98 and 2.13 times as long weighted.

### Which change does what

Over the 120 graphs with n <= 1000 in each mode, with the average percentage
below the reference in parentheses:

| Commit | Change | Unweighted at reference | Weighted at reference | Time unweighted / weighted |
|--------|--------|-------------------------|-----------------------|----------------------------|
| `8a185a1` | before | 78 (1.30%) | 20 (3.50%) | 177 s / 157 s |
| `bcb2f41` | multiplier scan | 106 (0.51%) | 71 (1.11%) | 297 s / 256 s |
| `87244bf` | scaled tolerances | 105 (0.64%) | 71 (1.14%) | 307 s / 260 s |
| `14ff29c` | eigenvector directions of every cluster | 105 (0.64%) | 71 (1.14%) | 323 s / 269 s |
| `c430841` | mu > 0 floor (`69c5cfa`), perturbation facility (off) | 105 (0.64%) | 71 (1.14%) | 326 s / 270 s |

The multiplier scan accounts for essentially all of the gain.  The eigenvector
directions and the mu > 0 floor together change no result on any of these
graphs, n = 2000 included, at about 5% more time.

The scaled tolerances matter here only through the degeneracy test.  Restoring
`fabs(c2)<1e-5` in `init_eigenclusters` alone makes `87244bf` and `c430841`
reproduce `bcb2f41` on all 180 runs with n <= 500 (larger graphs were not
checked).  On those graphs the absolute threshold declared 3 to 26% of the
eigenvalue clusters degenerate, taking their linear forms out of the secular
equation, where the scaled test declares none (census by `QMS_STATS`).
Weighted, the difference is a wash: 15 graphs better, 15 worse.  Unweighted it
is neutral up to n = 500 (4 better, 3 worse) but loses at n >= 1000 (2 better,
9 worse), all nine losses at p >= 0.7, while at p <= 0.5 the two tests agree on
every unweighted graph of every size.  That is the loss `87244bf` recorded on
`C1000.9` (65 -> 64) and `C2000.9` (73 -> 72), themselves uniform random graphs
with p = 0.9.  The two tests do find different cliques, though: the better of
`bcb2f41` and `87244bf` on each graph reaches the reference on 124 unweighted
and 95 weighted graphs, against 113 and 81 for `after`.

### Where the headroom is

The weighted graphs are the furthest from optimal: on average `after` is 4.8%
below the optimum at n = 500, p = 0.7, 3.7% at n = 300, p = 0.8, and 3.0% at
n = 500, p = 0.5.  Unweighted, two of the n = 1000, p = 0.25 graphs have a
clique of 9 where every version stops at 8, and the dense graphs, where nothing
is proven, sit well below k1: `after` finds 72 at n = 2000, p = 0.9, against
k1 = 79, and `C2000.9`, drawn from the same model, is known to contain a clique
of 80.

To reproduce (the generator needs numpy; the solver writes each `.sol` next to
its input):

```
python3 tools/random_graphs.py rnd                    # the 132 graphs
./qualex-ms rnd/g1000_90_3.clq.b                      # unweighted
./qualex-ms rnd/g1000_90_3.clq.b -wrnd/g1000_90_3.w   # weighted
python3 tools/random_graphs.py --sizes 200 --cliquer rnd   # ASCII files for cliquer
```

## Sources

- [IRIDIA Maximum Clique Benchmark](https://iridia.ulb.ac.be/~fmascia/maximum_clique/DIMACS-benchmark)
- [GitHub shah314/clique](https://github.com/shah314/clique)
