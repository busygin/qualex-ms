/***********************************************************************
!! dr.h: Douglas-Rachford iterations between a sphere and a set given !!
!! by its projection, which look for the points of a sphere of        !!
!! stationary points that lie in the set, e.g. the nonnegative ones,  !!
!! and on further sets, e.g. the surfaces of other wrappers           !!
!! (proposed by S. Busygin).                                          !!
***********************************************************************/

#ifndef DR_H
#define DR_H

#include <functional>
#include <vector>

// Sphere is {center + Q p : |p| = radius}, Q an n x k matrix with orthonormal
// columns, stored by columns
struct Sphere {
  int n, k;
  const double* center;
  const double* q;
  double radius;
};

// a Projection replaces y, of length n, by its projection onto a closed set
using Projection = std::function<void(double* y, int n)>;

// a BlockProjection does the same to each of the nb columns of y (n x nb,
// stored by columns), which lets it treat them with matrix products
using BlockProjection = std::function<void(double* y, int n, int nb)>;

// clip_negative() is the projection onto the nonnegative orthant
void clip_negative(double* y, int n);

// columnwise() applies a Projection to every column
BlockProjection columnwise(const Projection& project);

// sphere_projection() is the projection onto the sphere s, two matrix products
// with Q for a block (a point at the centre goes to the end of q_1); s must
// outlive it
BlockProjection sphere_projection(const Sphere& s);

// DRTargets says where a Douglas-Rachford stage drives the points of a sphere:
// towards the set of projection (the nonnegative orthant if null) and, if
// there are any, onto the further sets of surfaces as well, e.g. the surfaces
// of other wrapping matrices, which every clique indicator lies on.
// surfaces_only leaves the set of projection out, the orthant too, so that the
// points go onto the surfaces alone: a control for what that set costs in
// time and buys in the cliques found.
struct DRTargets {
  const Projection* projection = nullptr;
  std::vector<BlockProjection> surfaces;
  bool surfaces_only = false;
};

// douglas_rachford() iterates, P_S being the projection onto the sphere s and
// P_C the projection project_c,
//   e = P_S(v),   v <- v + P_C(2e - v) - e
// from each of the nb columns of v (n x nb, stored by columns, overwritten),
// for at most iters steps or until |P_C(e) - e| <= tol |e|, and leaves the
// last e of every start in e (n x nb).  done[t] tells whether start t got that
// close.  All starts iterate as one block, so that the projections onto the
// span of Q are two matrix products.  Returns the number of starts done.
// steps, here and below, unless null, gets the iterations of all the starts
// added to it.
int douglas_rachford(const Sphere& s, const Projection& project_c,
                     int nb, double* v, double* e, std::vector<bool>& done,
                     int iters, double tol, long* steps = nullptr);

// douglas_rachford_product() adds the sets of surfaces to douglas_rachford():
// it iterates between the product of the sphere and the surfaces, of which
// every start keeps one copy each, v_0 (the sphere's) .. v_r, each n x nb in
// turn in v, and the diagonal of the set of project_c, on which the
// projection of (y_0, .., y_r) is P_C of their average in every copy:
//   e_j = P_j(v_j),   c = P_C(mean_j (2 e_j - v_j)),   v_j <- v_j + c - e_j,
// for at most iters steps or until sum_j |P_C(mean_j e_j) - e_j|^2 <=
// tol^2 sum_j |e_j|^2.  It leaves the last e_0, the point of the sphere, of
// every start in e, done[t] telling whether start t stopped, and returns the
// number of starts that did.  Without surfaces it is douglas_rachford().
int douglas_rachford_product(const Sphere& s,
                             const std::vector<BlockProjection>& surfaces,
                             const Projection& project_c, int nb, double* v,
                             double* e, std::vector<bool>& done, int iters,
                             double tol, long* steps = nullptr);

// douglas_rachford_concur() looks for a point common to the K sets of sets by
// Douglas-Rachford in their product space ("divide and concur"): every start
// keeps one copy per set, v_1..v_K, each n x nb in turn in v, and an iteration
// averages them into e (the projection onto the diagonal) and moves each by
//   v_k <- v_k + P_k(2e - v_k) - e.
// It stops a start once every P_k(2e - v_k) lies within tol |e| of e, or after
// iters steps, and leaves the last e of every start in e (n x nb); done[t]
// tells whether start t stopped.  Returns the number of starts that did.
int douglas_rachford_concur(const std::vector<BlockProjection>& sets,
                            int n, int nb, double* v, double* e,
                            std::vector<bool>& done, int iters, double tol,
                            long* steps = nullptr);

#endif  // DR_H
