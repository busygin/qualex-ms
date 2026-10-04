/***********************************************************************
!! dr.h: Douglas-Rachford iterations between a sphere and a set given !!
!! by its projection, which look for the points of a sphere of        !!
!! stationary points that lie in the set, e.g. the nonnegative ones   !!
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

// clip_negative() is the projection onto the nonnegative orthant
void clip_negative(double* y, int n);

// douglas_rachford() iterates, P_S being the projection onto the sphere s and
// P_C the projection project_c,
//   e = P_S(v),   v <- v + P_C(2e - v) - e
// from each of the nb columns of v (n x nb, stored by columns, overwritten),
// for at most iters steps or until |P_C(e) - e| <= tol |e|, and leaves the
// last e of every start in e (n x nb).  done[t] tells whether start t got that
// close.  All starts iterate as one block, so that the projections onto the
// span of Q are two matrix products.  Returns the number of starts done.
int douglas_rachford(const Sphere& s, const Projection& project_c,
                     int nb, double* v, double* e, std::vector<bool>& done,
                     int iters, double tol);

#endif  // DR_H
