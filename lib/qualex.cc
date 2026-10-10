/***********************************************************************
!! QUALEX-MS: a QUick ALmost EXact Motzkin-Straus maximum weight      !!
!! clique/independent set solver.                                     !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.   !!
!!                                                                    !!
!! Qualex implementation                                              !!
***********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <chrono>
#include <list>
#include <vector>

#include "1d_math.h"
#include "dr.h"
#include "greedy_clique.h"
#include "linalg.h"
#include "qualex.h"
#include "refiner.h"

using namespace std;

inline double sqr(double x) { return x*x; }

// Tolerances.
//
// A backward stable symmetric eigensolver returns eigenvalues carrying an
// absolute error of order p(n)*eps*||A||_2, and returns, for any group of
// eigenvalues lying within that distance of each other, an accurate invariant
// subspace but individually meaningless eigenvectors.  That error is therefore
// the scale on which an eigenvalue counts as zero, on which two eigenvalues
// count as equal, and on which two multipliers count as the same point of the
// secular equation.  p(n) = n is the usual conservative choice.
//
// Nothing here may be an absolute constant: ||hatA|| grows with both the
// vertex weights and the density, so a fixed threshold such as 1e-5 means
// something different on every instance, and on a graph with large weights it
// can fall below the eigensolver's own noise level.
inline double eigen_tol(int n, double norm) {
  return (double)n*DBL_EPSILON*norm;
}

// spectral_norm() is ||hatA||_2, the eigenvalues being sorted ascending
inline double spectral_norm(int n, double* lambda) {
  double lo = fabs(lambda[0]), hi = fabs(lambda[n-1]);
  return lo>hi ? lo : hi;
}

// EigenCluster hold information about one full cluster of equal eigenvalues
struct EigenCluster {
  double lambda;  // the eigenvalue
  double c2;  // the square sum of the corresponding linear form coefficients
  int first, last;  // the eigenvalue range
  EigenCluster() {}
  EigenCluster(double _lambda, double _c2, int _first, int _last) :
    lambda(_lambda), c2(_c2), first(_first), last(_last) {}
};

// Equation holds an expression \sum_i (c_i / (x - lambda_i))^2 - RHS
// (which is discrepancy of an equation \sum_i (c_i / (x - lambda_i))^2 = RHS)
// and provides double operator() to compute it as a function of x
struct Equation {
  vector<EigenCluster>& lambda_clusters; // whence lambda and c2 are taken
  double rhs; // RHS value
  Equation(vector<EigenCluster>& _lambda_clusters, double _rhs) :
    lambda_clusters(_lambda_clusters), rhs(_rhs) {}
  double operator()(const double x);  // \sum_i (c_i / (x - lambda_i))^2 - RHS
};

// \sum_i (c_i / (x - lambda_i))^2 - RHS
double Equation::operator()(const double x) {
  double lhs = 0.0;
  vector<EigenCluster>::iterator i;
  for(i=lambda_clusters.begin();i<lambda_clusters.end();++i)
    lhs += i->c2/sqr(x-i->lambda);
  return lhs-rhs;
}

// init_projected_matrices() replaces a by hatA = P A P and fills hatb; it
// returns z^T A z, which try_dr_points() needs for s = 1 - x0^T H x0
double init_projected_matrices (
  MaxCliqueInfo& graph_info, double* a, double* hatb
) {
  int& n = graph_info.g.n;
  double* delta = new double[n];
  matrix_dot_vector(n,n,a,'T',graph_info.sqrtw,delta);
  double D = dot_product(n,graph_info.sqrtw,delta);
  for(int j=0;j<n;++j) {
    for(int i=0;i<=j;++i) {
      a[j*n+i] += graph_info.shift[i]*graph_info.shift[j]*D -
        graph_info.shift[i]*delta[j] - graph_info.shift[j]*delta[i];
      a[i*n+j] = a[j*n+i];
    }
    hatb[j] = (delta[j]-graph_info.shift[j]*D)/graph_info.W;
  }
  delete[] delta;
  return D;
}

struct QualexInfo {
  int k;  // the eigenvector space dimensionality, k<=n-1
  double* lambda; // eigenvalues of the projected feasible matrix
  double* q;  // eigenvector matrix of the projected feasible matrix
  double* c;  // linear form coefficients in the eigenvector basis

  vector<EigenCluster> active_clusters;  // the eigenvalue clusters where c2!=0
  vector<EigenCluster> degenerative_clusters;  // the clusters with c2=0

  double norm;        // ||hatA||_2, the scale of the spectrum
  double lambda_tol;  // eigenvalues within this distance are indistinguishable
  double c2_tol;      // cluster linear forms at or below this are zero
  double zAz;         // z^T A z of the wrapper before projection

  QualexInfo(): k(0), lambda(NULL), q(NULL), c(NULL),
    norm(0.0), lambda_tol(0.0), c2_tol(0.0), zAz(0.0) {}
  ~QualexInfo() {
    if(lambda!=NULL) delete[] lambda;
    if(q!=NULL) delete[] q;
    if(c!=NULL) delete[] c;
  }

  void install_eigenvalues (
    int n, double* lambda1, int& n_neg_eigens, int& n_pos_eigens
  );

  inline void install_eigenvectors (
    int n, double* q1, int n_neg_eigens, int n_pos_eigens
  );

  inline void init_c(int n, double* hatb);

  void init_eigenclusters();
};

// install_eigenvalues() drops the eigenvalues that are zero to the accuracy of
// the eigendecomposition.  hatA always has at least one of them, along z, as
// hatA = P A P and Pz = 0; a graph can contribute further ones.  They carry no
// information -- y_i = c_i/(mu - lambda_i) would divide by the noise in
// lambda_i -- so the whole eigenvector space is restricted to the rest.
void QualexInfo::install_eigenvalues (
  int n, double* lambda1, int& n_neg_eigens, int& n_pos_eigens
) {
  double* first_zero_lambda = lower_bound(lambda1,lambda1+n,-lambda_tol);
  double* last_zero_lambda = first_zero_lambda+1;
  // the caller has checked lambda1[n-1] > lambda_tol, so this terminates
  while(*last_zero_lambda<=lambda_tol) ++last_zero_lambda;
  n_neg_eigens = first_zero_lambda-lambda1;
  n_pos_eigens = n - (last_zero_lambda-lambda1);
  k = n_neg_eigens + n_pos_eigens;
  lambda = new double[k];
  memcpy(lambda,lambda1,sizeof(double)*n_neg_eigens);
  memcpy(lambda+n_neg_eigens,last_zero_lambda,sizeof(double)*n_pos_eigens);
}

void QualexInfo::install_eigenvectors (
  int n, double* q1, int n_neg_eigens, int n_pos_eigens
) {
  q = new double[n*k];
  memcpy(q,q1,sizeof(double)*n_neg_eigens*n);
  memcpy(q+(n_neg_eigens*n),q1+((n-n_pos_eigens)*n),sizeof(double)*n_pos_eigens*n);
}

void QualexInfo::init_c(int n, double* hatb) {
  c = new double[k];
  project_on_eigenvectors(n,k,hatb,c);  // also kept for projection_norm()

  // c = Q^T hatb is formed by an orthogonal projection, so each coefficient
  // carries an absolute error of order n*eps*||hatb||.  A cluster whose entire
  // linear form sits at or below that is indistinguishable from zero, which is
  // exactly hypothesis (23) under which the degenerate construction is derived;
  // above it the cluster has a genuine linear form and belongs to the secular
  // equation, however small.  A regular graph with equal weights has hatb = 0
  // identically -- delta is then constant and cancels against x0*D -- so the
  // comparison has to admit equality for that case to come out degenerate.
  double hatb2 = 0.0;
  for(int i=0;i<n;++i) hatb2 += sqr(hatb[i]);
  c2_tol = sqr((double)n*DBL_EPSILON)*hatb2;
}

void QualexInfo::init_eigenclusters() {
  int first_index = 0;
  double lambda_sum = lambda[0];
  double c2 = sqr(c[0]);
  double last_lambda = lambda[0];
  int i;
  for(i=1;i<k;++i) {
    if(lambda[i]-last_lambda<=lambda_tol) {
      lambda_sum += lambda[i];
      c2 += sqr(c[i]);
    } else {
      vector<EigenCluster>& clusters = (
        c2<=c2_tol ? degenerative_clusters : active_clusters );
      clusters.push_back (
        EigenCluster(lambda_sum/(i-first_index), c2, first_index, i) );
      first_index = i;
      lambda_sum = lambda[i];
      c2 = sqr(c[i]);
    }
    last_lambda = lambda[i];
  }
  vector<EigenCluster>& clusters = (
    c2<=c2_tol ? degenerative_clusters : active_clusters );
  clusters.push_back (
    EigenCluster(lambda_sum/(i-first_index), c2, first_index, i) );
}

bool init_projected_formulation (
  MaxCliqueInfo& graph_info, double* a, QualexInfo& solver_info
) {
  int& n = graph_info.g.n;
  double* hatb = new double[n];
  solver_info.zAz = init_projected_matrices(graph_info,a,hatb);

  // the eigenvectors stay with the backend until extract_eigenvectors()
  double* lambda1 = new double[n];
  symmetric_eigen(n,a,lambda1);

  solver_info.norm = spectral_norm(n,lambda1);
  solver_info.lambda_tol = eigen_tol(n,solver_info.norm);

  // no eigenvalue is positive to within the accuracy of the decomposition, so
  // there is no trust region branch to search
  if(lambda1[n-1]<=solver_info.lambda_tol) {
    delete[] hatb; delete[] lambda1;
    return false;
  }

  // Process eigenvalues on CPU to determine which eigenvectors to keep
  int n_neg_eigens, n_pos_eigens;
  solver_info.install_eigenvalues(n,lambda1,n_neg_eigens,n_pos_eigens);
  delete[] lambda1;

  // keep the selected eigenvectors as Q for the products, and a copy in q for
  // the stages that read its entries
  solver_info.q = new double[n*solver_info.k];
  extract_eigenvectors(n,n_neg_eigens,n_pos_eigens,solver_info.q);

  solver_info.init_c(n,hatb);
  delete[] hatb;
  solver_info.init_eigenclusters();

  // experimental: the cluster census, which is what says whether a wrapper has
  // actually activated anything.  Degeneracy here is a vanishing linear form,
  // not a repeated eigenvalue, so splitting eigenvalues alone moves nothing
  // from one column to the other.
  if(getenv("QMS_STATS")!=NULL)
    fprintf(stderr,
      "STATS n=%d k=%d active=%d degenerate=%d lam_max=%.6g lam_tol=%.3g "
      "c2_tol=%.3g\n",
      n, solver_info.k, (int)solver_info.active_clusters.size(),
      (int)solver_info.degenerative_clusters.size(),
      solver_info.active_clusters.empty() ? 0.0 :
        solver_info.active_clusters.back().lambda,
      solver_info.lambda_tol, solver_info.c2_tol);

  return true;
}

// MuRank keeps the few multipliers whose stationary points NBIW has made the
// heaviest cliques of.  The cheap NBIW pass thus does double duty: it looks
// for cliques, and it ranks the multipliers for the expensive Meta-NBIW stage.
struct MuRank {
  int capacity;
  double tol;             // multipliers this close are the same stationary point
  vector<double> mu;      // in decreasing order of weight
  vector<double> weight;
  MuRank(int _capacity, double _tol): capacity(_capacity), tol(_tol) {}
  void offer(double m, double w) {
    if(capacity<=0) return;
    size_t i;
    for(i=0;i<mu.size();++i) if(fabs(mu[i]-m)<=tol) return;
    for(i=0;i<weight.size() && weight[i]>=w;++i) ;
    if((int)i>=capacity) return;
    mu.insert(mu.begin()+i,m);
    weight.insert(weight.begin()+i,w);
    if((int)mu.size()>capacity) { mu.pop_back(); weight.pop_back(); }
  }
};

// mu_positive_floor() is the least multiplier counted as positive: zero, held
// off by the accuracy to which the eigenvalues it is compared against are known
inline double mu_positive_floor(QualexInfo& solver_info) {
  return solver_info.lambda_tol;
}

// stat_point() builds the stationary point of the trust region program that
// corresponds to a given value of the ball constraint multiplier mu, and
// leaves it in x rescaled into the original (Motzkin-Straus) variables
void stat_point (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double mu, double* x, double* y
) {
  vector<EigenCluster>::iterator ii;
  int i;
  for(ii=solver_info.active_clusters.begin();ii<solver_info.active_clusters.end();++ii) {
    for(i=ii->first;i<ii->last;++i) y[i] = solver_info.c[i]/(mu-solver_info.lambda[i]);
  }
  eigenvectors_dot_vector(graph_info.g.n,solver_info.k,'N',y,x);
  for(i=0;i<graph_info.g.n;++i) x[i] = (x[i]+graph_info.shift[i])*graph_info.sqrtw[i];
}

bool try_stat_point (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double mu, double* x, double* y
) {
  stat_point(graph_info,solver_info,mu,x,y);
  return refine_clique_MIN(graph_info,x);
}

// try_stat_point_w() is try_stat_point() also reporting the weight of the
// clique NBIW has made of the stationary point, so that the sampled mu
// values can be ranked against each other
bool try_stat_point_w (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double mu, double* x, double* y, double& weight
) {
  stat_point(graph_info,solver_info,mu,x,y);
  return refine_clique_MIN_w(graph_info,x,weight);
}

// outer_mu() returns the unique root of the secular equation
// sum_i c_i^2/(mu-lambda_i)^2 = r2 lying above the largest eigenvalue, i.e.
// the multiplier of the global maximiser of the trust region program at
// the radius r2
// The secular equation has a pole at every eigenvalue, and an eigenvalue is
// only known to within lambda_tol, so a bracket end must be kept that far off
// the spectrum -- the old multiplicative nudge of a few ulps put it deep inside
// the eigensolver's own noise, where 1/(mu-lambda_i)^2 is meaningless.
double outer_mu (
  QualexInfo& solver_info, double lam_max, double cnorm, double r2
) {
  Equation e(solver_info.active_clusters,r2);
  return root(lam_max+solver_info.lambda_tol,lam_max+cnorm/sqrt(r2),e);
}

// try_nondeg_points() takes the global maximiser at the radius of equ and, in
// each interval between the eigenvalues above lowest, the minimum of the secular
// function and, when the minimum lies below that radius, its two roots there
bool try_nondeg_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info, Equation& equ,
  double* x, double* y, MuRank& rank, double lowest
) {
  double mu_max = solver_info.active_clusters.back().lambda;
  double mu_min = mu_max+solver_info.lambda_tol;
  mu_max += projection_norm(solver_info.k)/sqrt(equ.rhs);
  double mu = root(mu_min,mu_max,equ);
  double weight;
  bool result = try_stat_point_w(graph_info,solver_info,mu,x,y,weight);
  rank.offer(mu,weight);
  vector<EigenCluster>::reverse_iterator ri;
  // Stop once the multiplier can no longer be positive.  Stationarity is
  // hatA xhat + hatb = mu xhat, so the objective's gradient is 2 mu xhat and
  //   grad g . xhat = 2 mu ||xhat||^2,
  // i.e. mu > 0 is exactly the condition that the gradient points out of the
  // ball -- what a maximiser needs, and the sign of the multiplier Theorem 8
  // remarks on.  The floor here used to be w_min/2, for which no justification
  // is on record; since mu_min is clamped below anyway, that only skipped the
  // band (0, w_min/2), and lifting it changes no result on either benchmark.
  double mu_floor = mu_positive_floor(solver_info);
  for(ri=solver_info.active_clusters.rbegin();ri<solver_info.active_clusters.rend();++ri) {
    if(ri->lambda<=lowest) break;
    mu_max = ri->lambda-solver_info.lambda_tol;
    vector<EigenCluster>::reverse_iterator ri1 = ri+1;
    if(ri1==solver_info.active_clusters.rend()) mu_min = mu_floor;
    else {
      mu_min = ri1->lambda+solver_info.lambda_tol;
      if(mu_min<mu_floor) mu_min = mu_floor;
    }
    if(mu_min<mu_max) {
      double fx;
      mu=minimum(mu_min,mu_max,equ,fx);
      if(try_stat_point_w(graph_info,solver_info,mu,x,y,weight)) result = true;
      rank.offer(mu,weight);
      if(fx<0.0) {
        double m1 = root(mu_min,mu,equ), m2 = root(mu,mu_max,equ);
        if(try_stat_point_w(graph_info,solver_info,m1,x,y,weight)) result = true;
        rank.offer(m1,weight);

        if(try_stat_point_w(graph_info,solver_info,m2,x,y,weight)) result = true;
        rank.offer(m2,weight);
      }
    }
  }
  return result;
}

// all_clusters() merges the two cluster lists back into one, ascending in
// lambda, the lists each being in that order already
void all_clusters(QualexInfo& solver_info, vector<EigenCluster>& out) {
  vector<EigenCluster>& a = solver_info.active_clusters;
  vector<EigenCluster>& d = solver_info.degenerative_clusters;
  out.clear();
  out.reserve(a.size()+d.size());
  size_t ia = 0, id = 0;
  while(ia<a.size() || id<d.size()) {
    if(id>=d.size() || (ia<a.size() && a[ia].first<d[id].first)) out.push_back(a[ia++]);
    else out.push_back(d[id++]);
  }
}

// try_eigendir_points() walks the eigenvector directions.
//
// For a cluster whose linear form vanishes, mu = lambda is a stationary point
// of the trust region program and the whole cluster eigenspace is free: the
// remaining coordinates are fixed by (21), the leftover radius (24) is spread
// over the eigenspace, and the method takes the 2k corner cases (25) where all
// of it goes onto one eigenvector.  That is what this used to do, for those
// clusters alone.
//
// The same construction is worth making for a cluster whose linear form does
// not vanish.  There it is no longer a stationary point of this program -- it
// is the exact construction above applied to the program with c deleted on
// that eigenspace, so it asks what the trust region would say if the objective
// had no linear pull along this eigenvector.  In the limit mu -> lambda from
// either side the genuine stationary point runs off along +-q_j, so these are
// the endpoints the interval scan approaches but never reaches: the scan
// samples the open interval between two eigenvalues, and this samples its ends.
//
// Until the tolerances were put on the right scale, an ordinary graph had 4 to
// 20 percent of its clusters misfiled as degenerate, so it was getting some of
// these candidates by accident.  They are worth having on purpose.
//
// This scan runs the whole spectrum, and in particular it does not stop where
// the multiplier stops being positive, as try_nondeg_points() does.  The
// argument for that floor is that mu > 0 puts the objective's gradient on the
// outside of the ball, which is what a maximiser needs -- but it is an argument
// about following a stationary point uphill, and nothing here is followed
// anywhere.  These vectors are handed to NBIW, which reads only the order they
// put the vertices in, and a direction that is useless to the continuous
// program can still rank the right vertices highest.  Measurably it does:
// MANN_a27 reaches 126 from a cluster of negative eigenvalue, and applying the
// positive-mu floor here costs exactly that.
// active_too takes the active clusters as well as the degenerate ones, and the
// scan stops at the clusters below lowest (-HUGE_VAL for the whole spectrum)
bool try_eigendir_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info, Equation& equ,
  double* x, double* y, bool active_too, double lowest
) {
  bool result = false;
  int& n = graph_info.g.n;
  vector<EigenCluster> clusters;
  if(active_too) all_clusters(solver_info,clusters);
  else clusters = solver_info.degenerative_clusters;
  double* x1 = new double[n];
  vector<EigenCluster>::reverse_iterator ri;
  for(ri=clusters.rbegin();ri<clusters.rend();++ri) {
    double mu = ri->lambda;
    if(mu<lowest) break;
    double r2 = equ.rhs;
    vector<EigenCluster>::iterator ii;
    int i;
    for(ii=solver_info.active_clusters.begin();ii<solver_info.active_clusters.end();++ii) {
      if(ii->first==ri->first) {  // the cluster being stepped onto: c deleted
        for(i=ii->first;i<ii->last;++i) y[i] = 0.0;
        continue;
      }
      for(i=ii->first;i<ii->last;++i) r2 -= sqr(y[i] = solver_info.c[i]/(mu-solver_info.lambda[i]));
    }
    if(r2>0.0) {
      r2 = sqrt(r2);
      eigenvectors_dot_vector(n,solver_info.k,'N',y,x1);
      for(int j=ri->first;j<ri->last;++j) {
        double* qj = solver_info.q+((size_t)j*n);
        for(i=0;i<n;++i) x[i] = (x1[i]+qj[i]*r2+graph_info.shift[i])*graph_info.sqrtw[i];
        if(refine_clique_MIN(graph_info,x)) result = true;

        for(i=0;i<n;++i) x[i] = (x1[i]-qj[i]*r2+graph_info.shift[i])*graph_info.sqrtw[i];
        if(refine_clique_MIN(graph_info,x)) result = true;
      }
    }
  }
  delete[] x1;
  return result;
}

// try_dr_points() looks for the nonnegative points of the stationary points of
// each degenerate cluster by Douglas-Rachford (proposed by S. Busygin).
//
// At a cluster whose linear form vanishes, mu = lambda is the multiplier of a
// stationary point x0 + xhat_p + e of the relaxed program for every e in the
// cluster's eigenspace E, xhat_p being the point (21) built from the other
// clusters.  On the wrapper surface x^T H x = 1 the quadric of the relaxed
// program reads q(xhat_p) + lambda_H |e|^2 = s, s = 1 - x0^T H x0, with
// lambda_H = lambda + w_min the eigenvalue of hatH (hatA = hatH - w_min on z's
// complement), so these stationary points form one sphere of radius
//   rho^2 = (s - q(xhat_p))/lambda_H,
// and every clique anchored at this level lies on it, with the one weight
//   W_mu = 1/(|x0|^2 + |xhat_p|^2 + rho^2).
// For a proper wrapper, a nonnegative point of the wrapper surface is
// supported on a clique, and a nonnegative stationary point is the indicator
// of the clique, so the nonnegative points of the sphere are exactly the
// cliques anchored at mu.  try_eigendir_points() hands MIN the 2k corners of a
// trust region sphere around the same centre; here the corners
// x0 + xhat_p +- rho q_j of this sphere are driven towards nonnegativity by
// douglas_rachford() with the projection project_c, by default clip_negative(),
//   v <- v + max(2 P_S v - v, 0) - P_S v
// (P_S the projection onto the sphere), read at P_S v, for at most iters steps
// or until the negative part of P_S v falls below tol of its norm, and the
// result is handed to MIN as try_eigendir_points() does.  A cluster whose W_mu
// does not exceed the incumbent cannot improve it and is skipped.  A wrapper
// that is not proper -- the equation wrapper of SAT01 ignores the
// contradictions between variables of no common equation -- has nonnegative
// points on the sphere that are not cliques, and its caller can pass a
// projection that takes those contradictions into account instead.
//
// The caller can also pass surfaces, further sets every clique indicator lies
// on, such as the surface {x^T H x = 1, z^T x = 1} of another wrapper H (see
// wrapper_surface()).  The sphere, the orthant and the surface of a proper
// wrapper meet exactly in the cliques anchored at mu, since a nonnegative
// point of that surface is supported on a clique.  Douglas-Rachford then runs
// between the product of the sphere and the surfaces and the diagonal of the
// set of project_c, douglas_rachford_product(): each start keeps a copy per
// factor, all at the corner first, project_c acts on their average, and the
// point read is the sphere's, as without surfaces, which this reduces to.
// concur takes the symmetric product space instead, douglas_rachford_concur(),
// with one copy per set, project_c's among them, read at their average;
// without surfaces it is the control that tells what the surfaces add there
// from what that product space changes.  surfaces_only leaves project_c out,
// as the control for what it costs and buys: the sphere and the surfaces
// alone, the diagonal of the whole space in place of that of its set.
bool try_dr_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double* x, double* y, int iters, int max_starts, double tol,
  const Projection& project_c, const vector<BlockProjection>& surfaces,
  bool concur, bool surfaces_only
) {
  int n = graph_info.g.n;
  double w_min = graph_info.w_min, W = graph_info.W;
  double s = 1.0 - solver_info.zAz/(W*W) - w_min/W;
  bool stats = getenv("QMS_STATS")!=nullptr;
  bool result = false;
  Projection identity = [](double*, int) {};
  const Projection& project = surfaces_only ? identity : project_c;
  vector<double> center(n);
  vector<EigenCluster>& deg = solver_info.degenerative_clusters;
  for(vector<EigenCluster>::reverse_iterator ri=deg.rbegin();ri!=deg.rend();++ri) {
    int first = ri->first, kc = ri->last-ri->first;
    if(kc<2) continue;  // the sphere is the two corners try_eigendir_points() tried
    double mu = ri->lambda;
    double yy = 0.0, q = 0.0;
    for(EigenCluster& cl : solver_info.active_clusters)
      for(int i=cl.first;i<cl.last;++i) {
        double yi = y[i] = solver_info.c[i]/(mu-solver_info.lambda[i]);
        yy += yi*yi;
        q += (solver_info.lambda[i]+w_min)*yi*yi + 2.0*solver_info.c[i]*yi;
      }
    double lam_h = mu+w_min;
    double rho2 = lam_h!=0.0 ? (s-q)/lam_h : -1.0;
    double w_mu = rho2>0.0 ? 1.0/(1.0/W+yy+rho2) : 0.0;
    if(!(rho2>0.0) || w_mu<=graph_info.lower_clique_bound*(1.0+1e-9)) {
      if(stats)
        fprintf(stderr, "DR mu=%.6g k=%d rho2=%.3g W_mu=%.6g lb=%g skipped\n",
                mu, kc, rho2, w_mu, graph_info.lower_clique_bound);
      continue;
    }
    eigenvectors_dot_vector(n,solver_info.k,'N',y,center.data());
    for(int i=0;i<n;++i) center[i] += graph_info.shift[i];
    // the cluster's eigenvectors, n x kc by columns, span the sphere
    Sphere sphere = {n, kc, center.data(), solver_info.q+(size_t)first*n, sqrt(rho2)};
    vector<BlockProjection> sets;  // for the symmetric product space
    if(concur) {
      sets.push_back(sphere_projection(sphere));
      if(!surfaces_only) sets.push_back(columnwise(project_c));
      sets.insert(sets.end(),surfaces.begin(),surfaces.end());
    }
    int copies = concur ? (int)sets.size() : 1+(int)surfaces.size();

    int n_starts = 2*kc;
    if(max_starts>0 && max_starts<n_starts) n_starts = max_starts;
    int block = n_starts<256 ? n_starts : 256;
    vector<double> v((size_t)copies*n*block), est((size_t)n*block);
    vector<bool> done;
    int converged = 0, improved = 0, hits = 0;  // hits: MIN cliques of weight W_mu
    long steps = 0;
    double dr_seconds = 0.0, min_seconds = 0.0;
    for(int b0=0;b0<n_starts;b0+=block) {
      int nb = n_starts-b0<block ? n_starts-b0 : block;
      for(int t=0;t<nb;++t) {  // start b0+t is the corner on eigenvector (b0+t)/2
        const double* qj = sphere.q+(size_t)((b0+t)/2)*n;
        double sign = (b0+t)%2==0 ? sphere.radius : -sphere.radius;
        for(int k=0;k<copies;++k)
          for(int i=0;i<n;++i) v[(size_t)k*n*nb+(size_t)t*n+i] = center[i]+sign*qj[i];
      }
      chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
      if(concur)
        converged += douglas_rachford_concur(sets,n,nb,v.data(),est.data(),
                                             done,iters,tol,&steps);
      else
        converged += douglas_rachford_product(sphere,surfaces,project,nb,
                                              v.data(),est.data(),done,iters,tol,
                                              &steps);
      chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
      for(int t=0;t<nb;++t) {
        const double* et = &est[(size_t)t*n];
        for(int i=0;i<n;++i) x[i] = et[i]*graph_info.sqrtw[i];
        double weight;
        if(refine_clique_MIN_w(graph_info,x,weight)) { result = true; ++improved; }
        if(weight>=w_mu*(1.0-1e-9)) ++hits;
      }
      dr_seconds += chrono::duration<double>(t1-t0).count();
      min_seconds += chrono::duration<double>(chrono::steady_clock::now()-t1).count();
    }
    if(stats) {
      fprintf(stderr, "DR mu=%.6g k=%d rho2=%.3g W_mu=%.6g starts=%d converged=%d "
              "hits=%d improved=%d lb=%g steps=%.1f dr_time=%.2fs min_time=%.2fs",
              mu, kc, rho2, w_mu, n_starts, converged, hits, improved,
              graph_info.lower_clique_bound, (double)steps/n_starts, dr_seconds,
              min_seconds);
      if(concur) fprintf(stderr, " concur sets=%d", (int)sets.size());
      else if(copies>1) fprintf(stderr, " surfaces=%d", copies-1);
      if(surfaces_only) fprintf(stderr, " surfaces-only");
      fputc('\n',stderr);
    }
  }
  return result;
}

// ---------------------------------------------------------------------
// Selection of the ball constraint multiplier mu
// ---------------------------------------------------------------------
//
// On the branch mu > lambda_max the stationary point is
//
//   x(mu) = z/W(V) + (mu I - hatA)^{-1} hatb,
//
// so the branch is a one-parameter homotopy: as mu -> infinity it collapses
// onto the plain vertex weight vector (whence NBIW reproduces the greedy
// solution already known), and as mu -> lambda_max it runs off along the
// leading eigenvector.  Everything the trust region formulation has to say
// about the graph lies in between, and what NBIW makes of x(mu) is a
// piecewise constant function of mu with many pieces.  The shipped method
// samples this homotopy exactly once, at the radius Proposition 7 assigns to
// the indicator of a clique one vertex heavier than the greedy one.  Since a
// stationary point of the *relaxed* program is not a clique indicator, that
// radius is an anchor for the scale rather than a prediction, so we keep it
// as an anchor and scan a geometric range of radii around it.

const int LADDER_ABOVE = 8;   // quarter octaves scanned above the anchor
const int LADDER_BELOW = 24;  // quarter octaves scanned below it

// try_outer_ladder() samples the mu > lambda_max branch at the radii
// r^2 = r_hat^2 * 2^(j/2), which are the radii of the indicators of cliques
// of assumed weight W(V)/(1 + 2^(j/2)*(W(V)/(W(Q)+w_min) - 1)).
bool try_outer_ladder (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info, double r2_anchor,
  double* x, double* y, MuRank& rank
) {
  double lam_max = solver_info.active_clusters.back().lambda;
  double cnorm = projection_norm(solver_info.k);
  bool result = false;
  for(int j=LADDER_ABOVE;j>=-LADDER_BELOW;--j) {
    double r2 = r2_anchor*pow(2.0,0.5*j);
    if(!(r2>0.0)) continue;
    double mu = outer_mu(solver_info,lam_max,cnorm,r2);
    double weight;
    if(try_stat_point_w(graph_info,solver_info,mu,x,y,weight)) result = true;
    rank.offer(mu,weight);
  }
  return result;
}

// try_theorem8_points() places mu directly, without going through a radius.
// Theorem 8 states that if Q is a maximal clique with W(N(v) cap Q) = C for
// every vertex v outside it, then the indicator of Q is a stationary point of
// the trust region program; its proof also pins the multipliers down, and in
// the projected formulation the ball multiplier comes out as
//
//   mu = W(Q) - w_min - C.
//
// This is a much more direct statement about mu than the radius is: it says
// mu measures by how much the clique outweighs the best attachment available
// to a vertex outside it.  Q is of course unknown, but the incumbent clique
// is a fair stand-in for the *shape* of the attachment distribution
// {W(N(v) cap Q)}, so we read that distribution off the incumbent, rescale it
// to the assumed weight of Q, and take a spread of its quantiles -- real
// graphs do not satisfy the hypothesis of Theorem 8 exactly, and the spread
// of C is precisely the extent to which they fail it.
// Measured on the DIMACS suite and on random weighted graphs, these multipliers
// never improved on what try_nondeg_points() had already found.  That is a
// result about try_nondeg_points() rather than about Theorem 8: the multipliers
// this identity predicts land inside the band of eigenvalue intervals that the
// interval scan already walks, which is why scanning those intervals works.
// The identity is kept because it is the one statement the theory makes about
// mu directly, and because it is what a pruned interval scan would have to aim
// at.
const int THM8_QUANTILES = 17;

bool try_theorem8_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double* x, double* y, MuRank& rank
) {
  int& n = graph_info.g.n;
  if(graph_info.clique.empty()) return false;
  list<int>::iterator ci;

  double wq = 0.0;  // weight of the incumbent clique
  bool_vector in_clique(n);
  for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();++ci) {
    wq += graph_info.g.weights[*ci];
    in_clique.put(*ci);
  }
  if(!(wq>0.0)) return false;

  // the attachment weight W(N(v) cap Q) of every vertex outside the incumbent
  vector<double> att;
  att.reserve(n);
  for(int v=0;v<n;++v) {
    if(in_clique.at(v)) continue;
    double s = 0.0;
    for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();++ci)
      if(graph_info.g.mates[v].at(*ci)) s += graph_info.g.weights[*ci];
    att.push_back(s);
  }
  if(att.empty()) return false;
  sort(att.begin(),att.end());

  // assume the sought clique is one minimum weight vertex heavier
  double target = graph_info.lower_clique_bound + graph_info.w_min;
  double scale = target/wq;

  bool result = false;
  double last_mu = -DBL_MAX;
  for(int j=0;j<THM8_QUANTILES;++j) {
    size_t idx = (size_t)((double)j/(THM8_QUANTILES-1)*(att.size()-1));
    double mu = target - graph_info.w_min - att[idx]*scale;
    // Theorem 8 admits mu > 0 -- that is its remark on the sign of the
    // multiplier -- and says nothing about w_min/2, which this used to use
    if(mu<=mu_positive_floor(solver_info)) continue;
    // quantiles that the spectrum cannot tell apart give the same point
    if(fabs(mu-last_mu)<=solver_info.lambda_tol) continue;
    last_mu = mu;
    double weight;
    if(try_stat_point_w(graph_info,solver_info,mu,x,y,weight)) result = true;
    rank.offer(mu,weight);
  }
  return result;
}

// try_meta_points() spends the far more thorough Meta-NBIW (Algorithm 3, one
// NBIW run started from each vertex, O(n^3)) on the handful of multipliers
// that the cheap single NBIW pass has ranked highest.  This is where a good
// choice of mu pays off most: with only a few points affordable, it matters
// which ones they are.
bool try_meta_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double* mus, size_t n_mus, int n_starts, double* x, double* y
) {
  bool result = false;
  for(size_t j=0;j<n_mus;++j) {
    stat_point(graph_info,solver_info,mus[j],x,y);
    double weight;
    if(meta_refine_MIN(graph_info,x,weight,n_starts)) result = true;
  }
  return result;
}

// theorem8_attachments() reads the least and the greatest attachment
// sum_{j in Q} a_ij z_j / z_i of a vertex i outside the incumbent clique Q off
// the wrapper, before it is projected; Theorem 8 needs them to be equal
bool theorem8_attachments (
  MaxCliqueInfo& graph_info, double* a, double& lo, double& hi
) {
  int& n = graph_info.g.n;
  bool_vector in_q(n);
  list<int>::iterator ci;
  for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();++ci)
    in_q.put(*ci);
  lo = DBL_MAX; hi = -DBL_MAX;
  for(int i=0;i<n;++i) {
    if(in_q.at(i)) continue;
    double s = 0.0;
    for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();++ci)
      s += a[i*n+*ci]*graph_info.sqrtw[*ci];
    s /= graph_info.sqrtw[i];
    if(s<lo) lo = s;
    if(s>hi) hi = s;
  }
  return lo<=hi;
}

// check_theorem8_point() is a diagnostic for anchored wrappers.  When every
// attachment equals C, the indicator of Q has to be the stationary point at
// mu = W(Q) - w_min - C, which in the variables stat_point() returns is w/W(Q)
// on Q and 0 elsewhere.  Reports how far stat_point() lands from it, and how
// many eigenvalues lie above that multiplier: none means the indicator is on
// the outer branch the ladder scans, where it is the global maximiser at its
// own radius.
void check_theorem8_point (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info,
  double att_lo, double att_hi, double* x, double* y
) {
  int& n = graph_info.g.n;
  bool_vector in_q(n);
  double wq = 0.0;
  list<int>::iterator ci;
  for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();++ci) {
    in_q.put(*ci);
    wq += graph_info.g.weights[*ci];
  }
  double mu = wq-graph_info.w_min-att_lo;
  int above = 0;
  for(int i=0;i<solver_info.k;++i) if(solver_info.lambda[i]>mu) ++above;
  double dev = -1.0;  // not applicable: the attachments differ
  if(att_hi-att_lo<=1e-9*wq && !solver_info.active_clusters.empty()) {
    stat_point(graph_info,solver_info,mu,x,y);
    dev = 0.0;
    for(int i=0;i<n;++i) {
      double d = fabs(x[i]-(in_q.at(i) ? graph_info.g.weights[i]/wq : 0.0));
      if(d>dev) dev = d;
    }
  }
  fprintf(stderr,
    "THM8 |Q|=%d W(Q)=%g att=%g..%g mu*=%g lam_max=%g above=%d/%d "
    "maxdev=%.3g (indicator entries ~%.3g)\n",
    (int)graph_info.clique.size(), wq, att_lo, att_hi, mu,
    solver_info.lambda[solver_info.k-1], above, solver_info.k, dev,
    graph_info.w_min/wq);
}

bool qualex_ms(MaxCliqueInfo& graph_info, double* a, double target,
               const DRTargets* dr) {
  QualexInfo solver_info;
  // experimental: read before init_projected_formulation() overwrites a
  double att_lo = 0.0, att_hi = 0.0;
  bool check_thm8 = getenv("QMS_STATS")!=NULL && !graph_info.clique.empty() &&
    theorem8_attachments(graph_info,a,att_lo,att_hi);
  if(!init_projected_formulation(graph_info,a,solver_info)) return false;
  // the radius is the one Proposition 7 assigns to a clique of weight aim: one
  // minimum weight vertex heavier than the incumbent, unless the weight sought
  // is known
  bool targeted = target>0.0;
  double aim = targeted ? target : graph_info.lower_clique_bound+graph_info.w_min;
  Equation equ (solver_info.active_clusters, 1.0/aim-1.0/graph_info.W);
  double* x = new double[graph_info.g.n];
  double* y = new double[solver_info.k];
  memset(y,0,sizeof(double)*solver_info.k);
  if(check_thm8)
    check_theorem8_point(graph_info,solver_info,att_lo,att_hi,x,y);

  // Knobs for reproducing the ablations of the multiplier selection; the
  // defaults are the shipped behaviour.  QMS_NO_LADDER and QMS_NO_THM8 switch
  // off the two added families of multipliers, QMS_META_N is how many of the
  // ranked multipliers reach the Meta-NBIW stage (0 switches that stage off;
  // the default is 2, and 0 with a target, as in version 1.2), and
  // QMS_META_STARTS restricts Meta-NBIW to that percentage of the vertices.
  bool use_ladder = getenv("QMS_NO_LADDER")==NULL;
  bool use_thm8   = getenv("QMS_NO_THM8")==NULL;
  int  n_meta     = getenv("QMS_META_N")?atoi(getenv("QMS_META_N")):(targeted?0:2);
  int  start_pct  = getenv("QMS_META_STARTS")?atoi(getenv("QMS_META_STARTS")):0;
  int  n_starts   = start_pct>0 ? 1+(graph_info.g.n*start_pct)/100 : 0;

  MuRank rank(n_meta,solver_info.lambda_tol);
  bool result = false;

  if(targeted) {
    // With the weight sought known (SAT01 knows that a solution is a clique of
    // weight m), this is the method of version 1.2 at the radius of that
    // weight: the global maximiser and, in the intervals between the
    // eigenvalues, the minimum of the secular function and its roots at the
    // radius (try_nondeg_points()), then the corners of the degenerate
    // clusters (try_eigendir_points()), both while the multiplier is positive;
    // version 1.2 stopped at w_min/2, which has no justification.  None of the
    // ladder, the Theorem 8 multipliers or the corners of the active clusters,
    // and the Meta-NBIW stage only if QMS_META_N asks for it, on the multipliers
    // of try_nondeg_points().  No clique weighs W or more, so a target that
    // high leaves nothing to look for.
    if(equ.rhs>0.0) {
      double lowest = mu_positive_floor(solver_info);
      if(!solver_info.active_clusters.empty() &&
         try_nondeg_points(graph_info,solver_info,equ,x,y,rank,lowest))
        result = true;
      result |= try_eigendir_points(graph_info,solver_info,equ,x,y,false,lowest);
    }
  } else if(!solver_info.active_clusters.empty()) {
    if(use_ladder) {
      // Rescan the homotopy until the anchor stops moving: whenever the scan
      // improves the incumbent clique, the weight Proposition 7 is applied to
      // changes, and so does the anchor radius the scan is centred on.
      double r2 = equ.rhs;
      for(int pass=0;pass<3;++pass) {
        if(try_outer_ladder(graph_info,solver_info,r2,x,y,rank)) result = true;
        double r2_new =
          1.0/(graph_info.lower_clique_bound+graph_info.w_min)-1.0/graph_info.W;
        if(!(r2_new<r2)) break;  // the bound did not improve, the anchor stands
        r2 = r2_new;
      }
    }
    if(try_nondeg_points(graph_info, solver_info, equ, x, y, rank,
                         mu_positive_floor(solver_info)))
      result = true;
    if(use_thm8 && try_theorem8_points(graph_info, solver_info, x, y, rank))
      result = true;
  }
  if(!targeted)
    result |= try_eigendir_points(graph_info, solver_info, equ, x, y,
                                  getenv("QMS_NO_EIGDIR")==NULL, -HUGE_VAL);

  // experimental: Douglas-Rachford on the degenerate clusters, QMS_DR the number
  // of iterations (unset or 0 switches it off), QMS_DR_STARTS at most that many
  // of the 2k corners of a cluster, toward the nonnegative orthant unless the
  // caller passes another projection, and onto the caller's surfaces as well;
  // QMS_DR_CONCUR takes the symmetric product space of all the sets
  int dr_iters = getenv("QMS_DR")?atoi(getenv("QMS_DR")):0;
  Projection orthant(clip_negative);
  vector<BlockProjection> no_surfaces;
  if(dr_iters>0 &&
     try_dr_points(graph_info,solver_info,x,y,dr_iters,
                   getenv("QMS_DR_STARTS")?atoi(getenv("QMS_DR_STARTS")):0,1e-9,
                   dr!=nullptr && dr->projection!=nullptr ? *dr->projection : orthant,
                   dr!=nullptr ? dr->surfaces : no_surfaces,
                   getenv("QMS_DR_CONCUR")!=nullptr,
                   dr!=nullptr && dr->surfaces_only))
    result = true;

  // Finally spend Meta-NBIW on the multipliers the scans ranked highest.
  if(!rank.mu.empty() &&
     try_meta_points(graph_info,solver_info,&rank.mu[0],rank.mu.size(),
                     n_starts,x,y))
    result = true;

  delete[] y;
  delete[] x;
  return result;
}
