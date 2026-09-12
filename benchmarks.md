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

## Sources

- [IRIDIA Maximum Clique Benchmark](https://iridia.ulb.ac.be/~fmascia/maximum_clique/DIMACS-benchmark)
- [GitHub shah314/clique](https://github.com/shah314/clique)
