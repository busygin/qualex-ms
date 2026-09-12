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
#include <functional>

#include "1d_math.h"
#include "qualex.h"
#include "refiner.h"
#include "greedy_clique.h"

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

// BLAS/LAPACK routines
extern "C" {
  double norm2(int,double*);
  int symmetric_eigen(int,double*,double*,double*);
  int symmetric_eigen_gpu(int,double*,double*);
  void extract_eigenvectors_gpu(int,int,int,double*);
  double dot_product(int,double*,double*);
  void matrix_dot_vector(int,int,double*,char,double*,double*);
  void store_eigenvectors_gpu(int,int,double*);
  void matrix_dot_vector_q(int,int,char,double*,double*);
  void compute_c_gpu(int,int,double*,double*);
  double norm2_c_gpu(int);
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
struct Equation: public unary_function<double,double> {
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
  for(i=lambda_clusters.begin();i<lambda_clusters.end();i++)
    lhs += i->c2/sqr(x-i->lambda);
  return lhs-rhs;
}

void init_projected_matrices (
  MaxCliqueInfo& graph_info, double* a, double* hatb
) {
  int& n = graph_info.g.n;
  double* delta = new double[n];
  matrix_dot_vector(n,n,a,'T',graph_info.sqrtw,delta);
  double D = dot_product(n,graph_info.sqrtw,delta);
  for(int j=0;j<n;j++) {
    for(int i=0;i<=j;i++) {
      a[j*n+i] += graph_info.shift[i]*graph_info.shift[j]*D -
        graph_info.shift[i]*delta[j] - graph_info.shift[j]*delta[i];
      a[i*n+j] = a[j*n+i];
    }
    hatb[j] = (delta[j]-graph_info.shift[j]*D)/graph_info.W;
  }
  delete[] delta;
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

  QualexInfo(): k(0), lambda(NULL), q(NULL), c(NULL),
    norm(0.0), lambda_tol(0.0), c2_tol(0.0) {}
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
  while(*last_zero_lambda<=lambda_tol) last_zero_lambda++;
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
  compute_c_gpu(n,k,hatb,c);  // Computes c and keeps it GPU-resident

  // c = Q^T hatb is formed by an orthogonal projection, so each coefficient
  // carries an absolute error of order n*eps*||hatb||.  A cluster whose entire
  // linear form sits at or below that is indistinguishable from zero, which is
  // exactly hypothesis (23) under which the degenerate construction is derived;
  // above it the cluster has a genuine linear form and belongs to the secular
  // equation, however small.  A regular graph with equal weights has hatb = 0
  // identically -- delta is then constant and cancels against x0*D -- so the
  // comparison has to admit equality for that case to come out degenerate.
  double hatb2 = 0.0;
  for(int i=0;i<n;i++) hatb2 += sqr(hatb[i]);
  c2_tol = sqr((double)n*DBL_EPSILON)*hatb2;
}

void QualexInfo::init_eigenclusters() {
  int first_index = 0;
  double lambda_sum = lambda[0];
  double c2 = sqr(c[0]);
  double last_lambda = lambda[0];
  int i;
  for(i=1;i<k;i++) {
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
  init_projected_matrices(graph_info,a,hatb);

  // Eigendecomposition: keep eigenvectors on GPU, only copy eigenvalues to CPU
  double* lambda1 = new double[n];
  symmetric_eigen_gpu(n,a,lambda1);

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

  // Extract selected eigenvector columns directly on GPU (D2D copy)
  // Also copies to CPU for try_deg_points which accesses q directly
  solver_info.q = new double[n*solver_info.k];
  extract_eigenvectors_gpu(n,n_neg_eigens,n_pos_eigens,solver_info.q);

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
    for(i=0;i<mu.size();i++) if(fabs(mu[i]-m)<=tol) return;
    for(i=0;i<weight.size() && weight[i]>=w;i++) ;
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
  for(ii=solver_info.active_clusters.begin();ii<solver_info.active_clusters.end();ii++) {
    for(i=ii->first;i<ii->last;i++) y[i] = solver_info.c[i]/(mu-solver_info.lambda[i]);
  }
  matrix_dot_vector_q(graph_info.g.n,solver_info.k,'N',y,x);
  for(i=0;i<graph_info.g.n;i++) x[i] = (x[i]+graph_info.shift[i])*graph_info.sqrtw[i];
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

bool try_nondeg_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info, Equation& equ,
  double* x, double* y, MuRank& rank
) {
  double mu_max = solver_info.active_clusters.back().lambda;
  double mu_min = mu_max+solver_info.lambda_tol;
  mu_max += norm2_c_gpu(solver_info.k)/sqrt(equ.rhs);  // Use GPU-resident c
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
  for(ri=solver_info.active_clusters.rbegin();ri<solver_info.active_clusters.rend();ri++) {
    if(ri->lambda<=mu_floor) break;
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
bool try_eigendir_points (
  MaxCliqueInfo& graph_info, QualexInfo& solver_info, Equation& equ,
  double* x, double* y, bool active_too
) {
  bool result = false;
  int& n = graph_info.g.n;
  vector<EigenCluster> clusters;
  if(active_too) all_clusters(solver_info,clusters);
  else clusters = solver_info.degenerative_clusters;
  double* x1 = new double[n];
  vector<EigenCluster>::reverse_iterator ri;
  for(ri=clusters.rbegin();ri<clusters.rend();ri++) {
    double mu = ri->lambda;
    double r2 = equ.rhs;
    vector<EigenCluster>::iterator ii;
    int i;
    for(ii=solver_info.active_clusters.begin();ii<solver_info.active_clusters.end();ii++) {
      if(ii->first==ri->first) {  // the cluster being stepped onto: c deleted
        for(i=ii->first;i<ii->last;i++) y[i] = 0.0;
        continue;
      }
      for(i=ii->first;i<ii->last;i++) r2 -= sqr(y[i] = solver_info.c[i]/(mu-solver_info.lambda[i]));
    }
    if(r2>0.0) {
      r2 = sqrt(r2);
      matrix_dot_vector_q(n,solver_info.k,'N',y,x1);
      for(int j=ri->first;j<ri->last;j++) {
        double* qj = solver_info.q+((size_t)j*n);
        for(i=0;i<n;i++) x[i] = (x1[i]+qj[i]*r2+graph_info.shift[i])*graph_info.sqrtw[i];
        if(refine_clique_MIN(graph_info,x)) result = true;

        for(i=0;i<n;i++) x[i] = (x1[i]-qj[i]*r2+graph_info.shift[i])*graph_info.sqrtw[i];
        if(refine_clique_MIN(graph_info,x)) result = true;
      }
    }
  }
  delete[] x1;
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
  double cnorm = norm2_c_gpu(solver_info.k);
  bool result = false;
  for(int j=LADDER_ABOVE;j>=-LADDER_BELOW;j--) {
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
  for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();ci++) {
    wq += graph_info.g.weights[*ci];
    in_clique.put(*ci);
  }
  if(!(wq>0.0)) return false;

  // the attachment weight W(N(v) cap Q) of every vertex outside the incumbent
  vector<double> att;
  att.reserve(n);
  for(int v=0;v<n;v++) {
    if(in_clique.at(v)) continue;
    double s = 0.0;
    for(ci=graph_info.clique.begin();ci!=graph_info.clique.end();ci++)
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
  for(int j=0;j<THM8_QUANTILES;j++) {
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
  for(size_t j=0;j<n_mus;j++) {
    stat_point(graph_info,solver_info,mus[j],x,y);
    double weight;
    if(meta_refine_MIN(graph_info,x,weight,n_starts)) result = true;
  }
  return result;
}

bool qualex_ms(MaxCliqueInfo& graph_info, double* a) {
  QualexInfo solver_info;
  if(!init_projected_formulation(graph_info,a,solver_info)) return false;
  Equation equ (
    solver_info.active_clusters,
    1.0/(graph_info.lower_clique_bound+graph_info.w_min)-1.0/graph_info.W );
  double* x = new double[graph_info.g.n];
  double* y = new double[solver_info.k];
  memset(y,0,sizeof(double)*solver_info.k);

  // Knobs for reproducing the ablations of the multiplier selection; the
  // defaults are the shipped behaviour.  QMS_NO_LADDER and QMS_NO_THM8 switch
  // off the two added families of multipliers, QMS_META_N is how many of the
  // ranked multipliers reach the Meta-NBIW stage (0 switches that stage off),
  // and QMS_META_STARTS restricts Meta-NBIW to that percentage of the vertices.
  bool use_ladder = getenv("QMS_NO_LADDER")==NULL;
  bool use_thm8   = getenv("QMS_NO_THM8")==NULL;
  int  n_meta     = getenv("QMS_META_N")?atoi(getenv("QMS_META_N")):2;
  int  start_pct  = getenv("QMS_META_STARTS")?atoi(getenv("QMS_META_STARTS")):0;
  int  n_starts   = start_pct>0 ? 1+(graph_info.g.n*start_pct)/100 : 0;

  MuRank rank(n_meta,solver_info.lambda_tol);
  bool result = false;

  if(!solver_info.active_clusters.empty()) {
    if(use_ladder) {
      // Rescan the homotopy until the anchor stops moving: whenever the scan
      // improves the incumbent clique, the weight Proposition 7 is applied to
      // changes, and so does the anchor radius the scan is centred on.
      double r2 = equ.rhs;
      for(int pass=0;pass<3;pass++) {
        if(try_outer_ladder(graph_info,solver_info,r2,x,y,rank)) result = true;
        double r2_new =
          1.0/(graph_info.lower_clique_bound+graph_info.w_min)-1.0/graph_info.W;
        if(!(r2_new<r2)) break;  // the bound did not improve, the anchor stands
        r2 = r2_new;
      }
    }
    if(try_nondeg_points(graph_info, solver_info, equ, x, y, rank))
      result = true;
    if(use_thm8 && try_theorem8_points(graph_info, solver_info, x, y, rank))
      result = true;
  }
  result |= try_eigendir_points(graph_info, solver_info, equ, x, y,
                                getenv("QMS_NO_EIGDIR")==NULL);

  // Finally spend Meta-NBIW on the multipliers the scans ranked highest.
  if(!rank.mu.empty() &&
     try_meta_points(graph_info,solver_info,&rank.mu[0],rank.mu.size(),
                     n_starts,x,y))
    result = true;

  delete[] y;
  delete[] x;
  return result;
}
