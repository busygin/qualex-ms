  QUALEX-MS: QUick ALmost EXact maximum weight clique solver
             based on a generalized Motzkin-Straus formulation,
             ver 1.2

  Copyright (c) Stanislav Busygin, 2000-2009. All rights reserved.


1. INTRODUCTION

This software is to solve maximum weight clique/independent set problem.
It is well-known that this problem is NP-hard, so an exact efficient
algorithm for it probably does not exist. However, QUALEX-MS has shown
the ability to solve this problem exactly in many cases, including
test instances considered hard for all existing algorithms. Complexity
of the routine is O(n^3), where n is the number of graph vertices.

The algorithm uses a trust region technique for a generalization
of the Motzkin-Straus formulation for maximum clique problem. The
generalization allows to consider vertex weights. The RAM requirement is
mainly determined by usage of DSYEVR routine of LAPACK for
eigendecomposition of an nxn double precision matrix. That is,
the available memory should be enough for, at least, two nxn double
matrices.

This software is distributed under GNU General Public License, ver. 3.


2. USAGE

QUALEX-MS uses some linear algebraic routines from the standard
packages BLAS and LAPACK. Please install them if you want to build
the executable file. They can be gotten at NetLib website:

http://www.netlib.org

Unless your hardware platform is very specific, it is suggested to
use the so-called ATLAS implementation of BLAS. There are ATLAS
prebuilts for almost all hardware platforms available for free and
compiled with full possible optimization.

Then, if you use the GNU environment, put correct values for
BLASLIB and LAPACKLIB in Makefile and just type `make`.

To use the solver, issue the command:

qualex-ms [flags] <dimacs_binary_file> [-w<weights_file>]

Flags:

+c: looking for maximum clique (default)

-c: looking for maximum independent set

+1: vertex numbers in solution file go from 1

-1: vertex numbers in solution file go from 0 (default)

weights_file: a text file for list of vertex weights (reals >= 1.0)

An obtained solution will be stored in a corresponding .sol file.
For example, you can find the maximum clique of an instance coded in
probe.clq.b file by the command

qualex-ms probe.clq.b

File probe.sol will be created to store the result.

File probe.w contains a sample list of weights for this instance,
so the command

qualex-ms probe.clq.b -wprobe.w

will take into accout the given weights.


3. SELECTION OF THE TRUST REGION MULTIPLIER

The trust region stage searches stationary points of

  max x^T A_G^(w) x   s.t.  z^T x = 1,  x^T x <= r^2,

each of which is indexed by the multiplier mu of the ball constraint.  On the
branch mu > lambda_max, where lambda_max is the largest eigenvalue of the
projected matrix, the stationary point is

  x(mu) = z/W(V) + (mu I - hatA)^{-1} hatb,

so that branch is a one-parameter homotopy running from the plain vertex
weight vector (mu -> infinity, where NBIW just reproduces the greedy solution
already in hand) to the leading eigenvector (mu -> lambda_max).  What NBIW
makes of x(mu) is a piecewise constant function of mu with many pieces, so
which mu one picks matters a great deal.

Up to version 1.2 a single point of that homotopy was used, at the radius
Proposition 7 of the paper assigns to the indicator of a clique one minimum
weight vertex heavier than the greedy one.  A stationary point of the relaxed
program is not a clique indicator, though, so that radius fixes the scale
rather than the point.  It is now used as an anchor for a geometric scan of
radii around it (see try_outer_ladder in qualex.cc).

Each sampled multiplier is scored by the weight of the clique NBIW builds from
it, and the best scoring ones are then handed to Meta-NBIW (Algorithm 3 of the
paper, NBIW restarted from every vertex), which is far more thorough and
correspondingly more expensive.  This is where the choice of mu earns its
keep: only a couple of multipliers can be afforded at that price.

Separately, the endpoints of that homotopy are worth visiting in their own
right.  As mu approaches an eigenvalue the stationary point runs off along the
corresponding eigenvector, so the eigenvector directions are the ends of the
intervals the scan samples but never reaches.  The method already built such
candidates for the clusters whose linear form vanishes, where they are genuine
stationary points; they are now built for every cluster, by deleting the linear
form on the cluster in question, and over the whole spectrum rather than down
to w_min/2 (see try_eigendir_points in qualex.cc).

These environment variables reproduce the ablations:

QMS_PERTURB=<eta>    experimental: perturb the free entries of the clique
                     wrapper, i.e. the matrix entries on non-adjacent vertex
                     pairs, by -eta*U[0,1]*sqrt(w_i*w_j).  Off by default.  Any
                     value <= 0 there leaves the theory intact, so a perturbed
                     run is still exact when it reports an optimum; different
                     wrappers expose different cliques, so the use of this is
                     to run several and keep the best.  Keep eta small, around
                     0.01: larger values lose more than they gain.  See the
                     comment on perturb_wrapper() in main.cc
QMS_PMODE=unif       make that perturbation uniform instead of random, which is
                     provably a no-op -- a control for checking the above
QMS_SEED=<s>         seed selecting the wrapper, so a run reproduces
QMS_ANCHOR=<theta>   experimental: anchor the wrapper on the best clique Q so
                     far, lowering the entries on the non-edges between Q and
                     the vertices outside it until Q meets the hypothesis of
                     Theorem 8 exactly (theta=1; theta<1 goes that fraction of
                     the way).  Anchored on the greedy clique it loses; use it
                     with QMS_ANCHOR_WARM, where it cannot, and where it works
                     as a local exchange around Q.  Off by default.  See the
                     comment on anchor_wrapper() in main.cc
QMS_ANCHOR_WARM      keep the first pass on the standard wrapper and apply
                     QMS_ANCHOR and QMS_PERTURB from the second pass on
QMS_ANCHOR_PASSES=<k>
                     anchored passes, each on the clique the previous one
                     found, stopping at the first that finds nothing better
                     (default 1)
QMS_ANCHOR_SHUFFLE=<s>
                     control: deal the anchoring corrections out to the
                     outside vertices in a random order, so that Theorem 8 is
                     no longer met
QMS_ICE=lovasz|spread
                     experimental: after anchoring, one line-searched step of
                     lambda_max minimization -- literally towards the Lovasz
                     theta function, or on the projected matrix within the
                     anchoring's freedom.  See the comment on ice_step()
QMS_ICE_SHUFFLE=<s>  control: deal the entries of the lovasz step out to the
                     non-edges in a random order
QMS_STATS            print the eigenvalue cluster census to stderr, which is
                     what says whether a wrapper activated anything, and the
                     ANCHOR, THM8, PASS and ICE lines of the experiments above
QMS_NO_EIGDIR        take eigenvector directions only from the clusters whose
                     linear form vanishes, as before, instead of from all of them
QMS_META_N=<k>       hand the k best scoring multipliers to Meta-NBIW
                     (default 2; 0 switches the stage off, which restores the
                     original running time but keeps only a small part of the
                     gain -- see benchmarks.md)
QMS_META_STARTS=<p>  restrict Meta-NBIW to the p percent of vertices the
                     stationary point rates highest (default 0, meaning all)
QMS_NO_LADDER        use only the single Proposition 7 radius, as before
QMS_NO_THM8          drop the multipliers predicted by Theorem 8

See benchmarks.md for the resulting DIMACS, weighted-instance and uniform random
graph figures; tools/random_graphs.py generates the random graphs.


4. What is new?

version 1.1:
- the preliminary greedy heuristic is now MIN starting n times
(i.e. from each vertex);
- the quadratic programming formulation is scaled by square roots
of the vertex weights.

version 1.1.1:
- minor code optimization for the degenerative case.

version 1.1.2:
- an empty graph bug fixed.

version 1.2:
- code redesigned to improve readability;
- bool_vector is now 64-bit compliant;
- a new parameter allows numbering of vertices from 1 (not 0) in solution files;
- Windows executable is recompiled with newest MinGW gcc and LAPACK 3.1.1.

unreleased:
- eigendecomposition and the dense linear algebra moved to cuSOLVER/cuBLAS;
- the trust region radius is scanned rather than fixed, and the best scoring
multipliers are refined by Meta-NBIW (see section 3).
