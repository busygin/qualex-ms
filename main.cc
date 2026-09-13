/*****************************************************************************
!!  This is the main() function of QUALEX-MS solver                         !!
!!  Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.        !!
!!                                                                          !!
!! This software is distributed AS IS. NO WARRANTY is expressed or implied. !!
!! The author grants a permission for everyone to use and distribute this   !!
!! software free of charge for research and educational purposes.           !!
!! Any COMMERCIAL usage of this software is PROHIBITED without a written    !!
!! permission of the copyright holder.                                      !!
!!                                                                          !!
!! Please send any inquiry to <busygin@gmail.com> and visit                 !!
!! Stas Busygin's NP-completeness page: <http://www.busygin.dp.ua/npc.html> !!
*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "graph.h"
#include "greedy_clique.h"
#include "preproc_clique.h"
#include "qualex.h"

// perturb_wrapper() replaces the zero entries of the weighted adjacency matrix
// on non-adjacent vertex pairs, which are the free parameters of the clique
// wrapper: Proposition 2 of the clique wrapper note admits any value below
// z_i z_j there.
//
// Every value this accepts is safe.  Lowering an entry strictly decreases
// x^T H x wherever x_i x_j > 0 and leaves every clique-supported point exactly
// where it was, so a value <= 0 carries the generalized Motzkin-Straus theorem
// over unchanged, optimum value and clique optima together.  A strictly
// negative one goes further: the complete multipartite optima of Theorem 4 all
// contain a non-adjacent pair and so become strictly suboptimal, leaving the
// maximum weight clique indicators as the only global optima.  A perturbed run
// therefore never reports a clique that is not one.
//
// The reason to do it is that the perturbation moves the spectrum and the
// eigenvectors, so the trust region stage visits different stationary points,
// and different wrappers expose different cliques.  Measured on the current
// solver at eta = 0.01 over three seeds: keller5 26 -> 27, which is optimal,
// on two of them, and MANN_a45 342 -> 344 against a best known 345.  Nothing
// regressed at that scale on any instance tried.
//
// Keep eta small.  At 0.5 the same two gains survive but several instances
// fall over -- san400_0.7_3 22 -> 18, gen400_p0.9_55 53 -> 51 on every seed,
// san200_0.9_3 44 -> 43 -- which is what the loss of nonnegativity in A^(w)
// costs: the paper's argument for why the stationary points stay roughly
// nonnegative leans on it.  The response is not monotone in eta either
// (keller5 on seed 1 reaches 27 at 0.01 through 0.35 and again at 0.8, but not
// at 0.5 or 1.5), so the effect is diversity among wrappers rather than a
// threshold to tune past.  Run several and keep the best; a single wrapper is
// not a better default, and this stays off unless asked for.
//
// Small perturbations only became useful once the cluster tolerances were put
// on the right scale.  Against the old absolute cut of 1e-5 on the summed
// linear form, an eta of 0.05 split the eigenvalues but left every cluster
// classified degenerate, because the coefficients it created were real but
// tiny; against the present n*eps*||hatb|| they count.
//
// The uniform mode is a control rather than a strategy.  eps_ij = -eta z_i z_j
// makes H = (1+eta) H_0 - eta z z^T, and the projection onto z^T x = 1 removes
// the rank one term, leaving the eigenvectors alone and the eigenvalues
// affinely mapped; substituting into y_i = c_i/(mu - lambda_i) reproduces the
// same stationary points.  It should match an unperturbed run exactly, and
// exists to check that it does.
void perturb_wrapper(Graph& g, MaxCliqueInfo& info, double* a, double eta,
                     bool uniform, unsigned long long seed) {
  int& n = g.n;
  for(int i=0;i<n;i++) for(int j=0;j<i;j++) {
    if(g.mates[i].at(j)) continue;
    // Knuth MMIX linear congruential generator, so that a seed reproduces a
    // wrapper exactly whatever the platform's long happens to be
    seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
    double u = uniform ? 1.0 :
      (double)((seed>>11)&0xFFFFFFFFULL)/4294967296.0;
    a[i*n+j] = a[j*n+i] = -eta*u*info.sqrtw[i]*info.sqrtw[j];
  }
}

// build_wrapper() fills a with A^(w), the standard clique wrapper less w_min
// on the diagonal, which is the matrix qualex_ms() takes
void build_wrapper(Graph& g, MaxCliqueInfo& info, double* a) {
  int& n = g.n;
  memset(a,0,sizeof(double)*n*n);
  int i,j;
  for(i=0;i<n;i++) {
    a[i*(n+1)] = g.weights[i]-info.w_min;
    bit_iterator bi(g.mates[i]);
    while((j=bi.next())>-1) {
      if(j>i) break;
      a[i*n+j] = a[j*n+i] = info.sqrtw[i]*info.sqrtw[j];
    }
  }
}

// anchor_wrapper() spends the free entries of the wrapper on the incumbent
// clique Q, so that Q satisfies the hypothesis of Theorem 8 exactly and its
// indicator becomes a stationary point of the trust region program.
//
// Theorem 8 asks every vertex i outside Q to be attached to Q with the same
// weight, which in a general wrapper H reads
//
//   C_i = sum_{j in Q} h_ij z_j / z_i = C.
//
// An edge contributes w_j to C_i, so C_i is W(N(i) cap Q) plus whatever the
// non-edges between i and Q hold, and each of those non-edges enters exactly
// one of the conditions.  Q being maximal leaves every i at least one of them,
// so the conditions can always be met, and meeting them with C = min_i C_i
// only ever lowers an entry, which the wrapper admits (see perturb_wrapper()).
// The correction of each i is spread over its non-neighbours in Q in
// proportion to their weights,
//
//   h_ij += t_i z_i z_j,   t_i = theta (C - C_i) / W(Q \ N(i)),
//
// which at theta = 1 makes the indicator of Q stationary with multiplier
// mu = W(Q) - w_min - C, and at theta < 1 shrinks the spread of the C_i by the
// factor 1 - theta instead of removing it.
//
// C_i is read off the matrix rather than the graph, so applied after
// perturb_wrapper() this anchors the perturbed wrapper exactly.  Returns false,
// leaving a alone, when there is nothing to anchor.  report is off for the
// re-anchoring ice_step() does while it searches for its step length.
//
// Measured against the standard wrapper on DIMACS, the 56 weighted graphs and
// the uniform random graphs with n <= 1000 of benchmarks.md, and on the big
// DIMACS graphs and the n = 2000 random ones:
//
// Anchoring does not make Q attractive.  Its indicator is never the maximiser
// at its own radius -- 8 to 276 eigenvalues lie above mu -- and pushing C below
// min_i C_i does not change that, since lambda_max then grows faster than mu.
// What it does is make the stationary points near Q into neighbours of Q, which
// is useless around a poor clique: anchored on the greedy clique a single pass
// loses (DIMACS 2 better, 9 worse, gen400_p0.9_65 65 -> 52; weighted random
// graphs 15 better, 67 worse).  Anchored on the clique the standard pass ends
// with (QMS_ANCHOR_WARM) it cannot lose, and it improved DIMACS 3 times
// (keller5 27, gen400_p0.9_55 55, MANN_a45 343), the weighted graphs 8 times,
// and the random graphs 15 times unweighted and 55 times weighted, where a
// second pass on a random wrapper improved 21.  Its gains swap a median of two
// or three vertices and keep 95% of Q.  Repeating it on each new clique
// (QMS_ANCHOR_PASSES) keeps climbing on dense weighted graphs, and reached
// keller6 55 and MANN_a81 1098.  It stays off by default because of the cost:
// about 1.6 times the running time for one anchored pass, 2 to 2.3 times with
// the repeats.
//
// QMS_ANCHOR_SHUFFLE, which deals the same corrections out to the wrong
// vertices, does as well on sparse graphs but not on dense ones (weighted
// random graphs with p >= 0.9: 38 improved against 6), which is where meeting
// Theorem 8 exactly matters.
bool anchor_wrapper(Graph& g, MaxCliqueInfo& info, double* a, double theta,
                    bool report = true) {
  int& n = g.n;
  if(info.clique.empty()) return false;
  bool_vector in_q(n);
  double wq = 0.0;
  list<int>::iterator ci;
  for(ci=info.clique.begin();ci!=info.clique.end();ci++) {
    in_q.put(*ci);
    wq += g.weights[*ci];
  }
  vector<double> att(n,0.0), miss(n,0.0);
  double c_min = HUGE_VAL, c_max = -HUGE_VAL;
  int i;
  for(i=0;i<n;i++) {
    if(in_q.at(i)) continue;
    for(ci=info.clique.begin();ci!=info.clique.end();ci++) {
      att[i] += a[i*n+*ci]*info.sqrtw[*ci];
      if(!g.mates[i].at(*ci)) miss[i] += g.weights[*ci];
    }
    att[i] /= info.sqrtw[i];
    if(!(miss[i]>0.0)) return false;  // Q is not maximal
    if(att[i]<c_min) c_min = att[i];
    if(att[i]>c_max) c_max = att[i];
  }
  if(c_min>c_max) return false;  // Q is the whole graph

  // the correction to the attachment of each vertex outside Q
  vector<int> outside;
  vector<double> d;
  for(i=0;i<n;i++) {
    if(in_q.at(i)) continue;
    outside.push_back(i);
    d.push_back(theta*(c_min-att[i]));
  }

  // experimental control: QMS_ANCHOR_SHUFFLE=<seed> deals the same
  // corrections out to the outside vertices in a random order.  Every entry
  // still only goes down and as much is taken off the attachments in total,
  // but they are no longer equalized, so what the anchoring buys beyond this
  // control is what meeting Theorem 8 buys.
  const char* shuffle = getenv("QMS_ANCHOR_SHUFFLE");
  if(shuffle!=NULL) {
    unsigned long long seed = strtoull(shuffle,NULL,10);
    for(size_t k=d.size();k>1;k--) {
      seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
      size_t r = (size_t)((seed>>33)%k);
      double tmp = d[k-1]; d[k-1] = d[r]; d[r] = tmp;
    }
  }

  double t_min = 0.0;
  long touched = 0;
  for(size_t k=0;k<outside.size();k++) {
    i = outside[k];
    double t = d[k]/miss[i];
    if(t==0.0) continue;
    if(t<t_min) t_min = t;
    for(ci=info.clique.begin();ci!=info.clique.end();ci++) {
      int j = *ci;
      if(g.mates[i].at(j)) continue;
      double h = a[i*n+j]+t*info.sqrtw[i]*info.sqrtw[j];
      a[i*n+j] = a[j*n+i] = h;
      touched++;
    }
  }
  if(report && getenv("QMS_STATS")!=NULL)
    fprintf(stderr,
      "ANCHOR |Q|=%d W(Q)=%g C_i=%g..%g theta=%g t_min=%.4g touched=%ld "
      "mu*=%g%s\n",
      (int)info.clique.size(), wq, c_min, c_max, theta, t_min, touched,
      wq-info.w_min-c_min, shuffle!=NULL ? " shuffled" : "");
  return true;
}

extern "C" {
  int symmetric_eigen_gpu(int,double*,double*);
  void extract_eigenvectors_gpu(int,int,int,double*);
  int symmetric_eigenvalues_gpu(int,double*,double*);
}

// project_wrapper() writes P a P into b, P projecting out z = sqrtw as
// init_projected_matrices() does, and returns |hatb| = |P a z|/W
double project_wrapper(MaxCliqueInfo& info, double* a, double* b) {
  int& n = info.g.n;
  double* delta = new double[n];
  double D = 0.0;
  int i,j;
  for(j=0;j<n;j++) {
    double s = 0.0;
    for(i=0;i<n;i++) s += a[j*n+i]*info.sqrtw[i];
    delta[j] = s;
    D += info.sqrtw[j]*s;
  }
  double b2 = 0.0;
  for(j=0;j<n;j++) {
    for(i=0;i<n;i++)
      b[j*n+i] = a[j*n+i] + info.shift[i]*info.shift[j]*D -
        info.shift[i]*delta[j] - info.shift[j]*delta[i];
    double hb = (delta[j]-info.shift[j]*D)/info.W;
    b2 += hb*hb;
  }
  delete[] delta;
  return sqrt(b2);
}

// ice_step() takes one step of lambda_max minimization on an anchored wrapper,
// the first step of a descent towards the Lovasz theta function, with its
// length chosen by a line search.
//
// mode "lovasz" is the literal step: the gradient of the largest eigenvalue
// of the wrapper itself, 2 u_i u_j on every non-edge, followed by re-anchoring.
// Its top eigenvector is within a percent or so of z, so the step points almost
// exactly along the uniform perturbation (cosine 0.98 to 1.00), which changes
// nothing (see perturb_wrapper()).  The rest moves z towards an eigenvector of
// the wrapper, i.e. hatb = P H z / W towards 0, the direction of the degeneracy
// the theta optimum is known for: the spread of the projected spectrum over
// |hatb| grows 1.2 to 1.7 times (median), where QMS_ICE_SHUFFLE, the same
// entries dealt out to the non-edges in a random order, leaves it alone.
//
// mode "spread" minimizes the largest eigenvalue of the projected matrix the
// trust region works with, and only within the freedom the anchoring leaves:
// each outside vertex's correction is moved between its non-neighbours in Q
// with its total held fixed, so Theorem 8 stays exact.  The top of that
// spectrum is nearly degenerate, so the gradient is taken over the eigenvalues
// within 5% of the largest.  One step cannot do much there: it narrows the
// spread over |hatb| by about 1%, and lowering entries, the only safe
// direction, creates eigenvalue pairs of its own.
//
// Measured as a warm anchored pass with the step against the same pass without
// it: "lovasz" reached 41 of the 70 proven optima of the weighted random graphs
// against 37, all four gains at p <= 0.5, where it was better 5 times and never
// worse, against 0 and 1 for its shuffled control; on dense graphs it was
// better 7 times and worse 9, no different from that control, and on DIMACS it
// gave back MANN_a45 343 -> 342.  "spread" changed little either way (weighted
// random graphs 7 better, 5 worse).  Both stay off by default.
//
// Every entry stays <= 0.  Under QMS_STATS the ICE line reports, besides the
// objective, the spread of the projected spectrum over |hatb|, which is what
// the gauge H -> (1+t)H - t z z^T leaves alone.
void ice_step(Graph& g, MaxCliqueInfo& info, double* a, double theta,
              const char* mode) {
  static const double SIGMAS[] = {0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0};
  const int N_SIGMAS = sizeof(SIGMAS)/sizeof(SIGMAS[0]);
  int& n = g.n;
  size_t nn = (size_t)n*n;
  bool lovasz = strcmp(mode,"lovasz")==0;
  if(!lovasz && strcmp(mode,"spread")!=0) return;
  bool stats = getenv("QMS_STATS")!=NULL;
  int i,j;

  double* b = new double[nn];      // scratch for the eigensolver
  double* trial = new double[nn];
  double* d = new double[nn];      // the step direction, zero off its support
  double* lam = new double[n];
  memset(d,0,sizeof(double)*nn);

  bool_vector in_q(n);
  vector<int> q(info.clique.begin(),info.clique.end());
  for(size_t k=0;k<q.size();k++) in_q.put(q[k]);

  double f0, dn2 = 0.0, s_cap = HUGE_VAL, bn0 = 0.0;
  int cluster = 1;
  if(lovasz) {
    symmetric_eigen_gpu(n,a,lam);
    f0 = lam[n-1];
    double* u = new double[n];
    extract_eigenvectors_gpu(n,0,1,u);
    for(i=0;i<n;i++) for(j=0;j<i;j++) {
      if(g.mates[i].at(j)) continue;
      d[i*n+j] = d[j*n+i] = -2.0*u[i]*u[j];
    }
    delete[] u;
    // experimental control: QMS_ICE_SHUFFLE=<seed> deals the entries of the
    // step, taken relative to z_i z_j, out to the non-edges in a random order.
    // That keeps the distribution of the relative entries, and so the size of
    // the uniform part and of the rest, but not the direction of the rest:
    // what the literal step buys beyond this control is what the theta
    // direction buys.
    const char* shuffle = getenv("QMS_ICE_SHUFFLE");
    if(shuffle!=NULL) {
      vector<double> r;
      for(i=0;i<n;i++) for(j=0;j<i;j++)
        if(!g.mates[i].at(j))
          r.push_back(d[i*n+j]/(info.sqrtw[i]*info.sqrtw[j]));
      unsigned long long seed = strtoull(shuffle,NULL,10);
      for(size_t k=r.size();k>1;k--) {
        seed = seed*6364136223846793005ULL + 1442695040888963407ULL;
        size_t p = (size_t)((seed>>33)%k);
        double tmp = r[k-1]; r[k-1] = r[p]; r[p] = tmp;
      }
      size_t k = 0;
      for(i=0;i<n;i++) for(j=0;j<i;j++)
        if(!g.mates[i].at(j))
          d[i*n+j] = d[j*n+i] = r[k++]*info.sqrtw[i]*info.sqrtw[j];
    }
    for(i=0;i<n;i++) for(j=0;j<i;j++)
      if(!g.mates[i].at(j)) dn2 += d[i*n+j]*d[i*n+j];
  } else {
    bn0 = project_wrapper(info,a,b);
    symmetric_eigen_gpu(n,b,lam);
    f0 = lam[n-1];
    while(cluster<n && lam[n-1-cluster]>=f0-0.05*fabs(f0)) cluster++;
    double* v = new double[(size_t)n*cluster];
    extract_eigenvectors_gpu(n,0,cluster,v);
    for(i=0;i<n;i++) {
      if(in_q.at(i)) continue;
      int carrying = 0;
      for(size_t k=0;k<q.size();k++)
        if(!g.mates[i].at(q[k]) && a[i*n+q[k]]<0.0) carrying++;
      if(carrying<2) continue;  // nothing to move the correction to
      double gz = 0.0, zz = 0.0;
      for(size_t k=0;k<q.size();k++) {
        j = q[k];
        if(g.mates[i].at(j)) continue;
        double gij = 0.0;
        for(int c=0;c<cluster;c++) gij += v[(size_t)c*n+i]*v[(size_t)c*n+j];
        gij *= 2.0/cluster;
        d[i*n+j] = gij;
        gz += gij*info.sqrtw[j];
        zz += g.weights[j];
      }
      // take out the part that would change the attachment of i
      for(size_t k=0;k<q.size();k++) {
        j = q[k];
        if(g.mates[i].at(j)) continue;
        double p = d[i*n+j]-gz/zz*info.sqrtw[j];
        d[i*n+j] = d[j*n+i] = -p;
        dn2 += p*p;
        if(-p>1e-15 && -a[i*n+j]/(-p)<s_cap) s_cap = -a[i*n+j]/(-p);
      }
    }
    delete[] v;
  }

  // the line search, on the objective of the mode
  double f_best = f0, s_best = 0.0, sigma_best = 0.0;
  if(f0>0.0 && dn2>0.0) {
    for(int k=0;k<N_SIGMAS;k++) {
      double s = SIGMAS[k]*f0/dn2;
      bool capped = s>=s_cap;
      if(capped) s = s_cap;
      for(size_t e=0;e<nn;e++) trial[e] = a[e]+s*d[e];
      if(lovasz) {
        for(i=0;i<n;i++) for(j=0;j<n;j++)
          if(i!=j && !g.mates[i].at(j) && trial[i*n+j]>0.0) trial[i*n+j] = 0.0;
        anchor_wrapper(g,info,trial,theta,false);
        symmetric_eigenvalues_gpu(n,trial,lam);
      } else {
        project_wrapper(info,trial,b);
        symmetric_eigenvalues_gpu(n,b,lam);
      }
      if(lam[n-1]<f_best) { f_best = lam[n-1]; s_best = s; sigma_best = SIGMAS[k]; }
      if(capped) break;
    }
  }

  // rebuild the best trial and take it
  long changed = 0;
  if(s_best>0.0) {
    for(size_t e=0;e<nn;e++) trial[e] = a[e]+s_best*d[e];
    if(lovasz) {
      for(i=0;i<n;i++) for(j=0;j<n;j++)
        if(i!=j && !g.mates[i].at(j) && trial[i*n+j]>0.0) trial[i*n+j] = 0.0;
      anchor_wrapper(g,info,trial,theta,false);
    }
    for(size_t e=0;e<nn;e++) if(trial[e]!=a[e]) changed++;
  }
  if(stats) {
    double lmax0 = 0.0, lmin0 = 0.0;
    bn0 = project_wrapper(info,a,b);
    symmetric_eigenvalues_gpu(n,b,lam);
    lmax0 = lam[n-1]; lmin0 = lam[0];
    double lmax1 = lmax0, lmin1 = lmin0, bn1 = bn0, t_min = 0.0;
    if(s_best>0.0) {
      bn1 = project_wrapper(info,trial,b);
      symmetric_eigenvalues_gpu(n,b,lam);
      lmax1 = lam[n-1]; lmin1 = lam[0];
    }
    double* m = s_best>0.0 ? trial : a;
    for(i=0;i<n;i++) for(j=0;j<i;j++)
      if(!g.mates[i].at(j) && m[i*n+j]/(info.sqrtw[i]*info.sqrtw[j])<t_min)
        t_min = m[i*n+j]/(info.sqrtw[i]*info.sqrtw[j]);
    fprintf(stderr,
      "ICE mode=%s cluster=%d objective %g -> %g sigma=%g changed=%ld "
      "t_min=%.4g projected %g/%g -> %g/%g spread/|hatb| %.4g -> %.4g\n",
      mode, cluster, f0, f_best, sigma_best, changed, t_min,
      lmax0, lmin0, lmax1, lmin1, (lmax0-lmin0)/bn0, (lmax1-lmin1)/bn1);
  }
  if(s_best>0.0) memcpy(a,trial,sizeof(double)*nn);

  delete[] lam;
  delete[] d;
  delete[] trial;
  delete[] b;
}

// print_clique() prints a provided clique and its total weight
// in a file along with the graph header
void print_clique (
  const char* filename, const char* header,
  list<int>& clique, double clique_weight, unsigned char from1
) {
  FILE* file=fopen(filename,"w");
  fputs(header,file);
  fprintf(file,"s %lg\n",clique_weight);
  for(list<int>::iterator i=clique.begin();i!=clique.end();i++)
    fprintf(file, "v%11d\n", *i + from1);
  fclose(file);
}

int main(int argc,char** argv) {
  puts(
    "QUick ALmost EXact maximum weight clique solver, ver. 1.2-MS\n\n"
    "Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.\n\n"
    "This software is distributed AS IS. NO WARRANTY is expressed or implied.\n"
    "The author grants a permission for everyone to use and distribute this\n"
    "software free of charge for research and educational purposes.\n"
    "Any COMMERCIAL usage of this software is PROHIBITED without a written\n"
    "permission of the copyright holder.\n\n"
    "Please send any inquiry to <busygin@gmail.com> and visit\n"
    "Stas Busygin's NP-completeness page: <http://www.busygin.dp.ua/npc.html>\n"
  );

  char* name=NULL;
  char* weights_name=NULL;
  bool for_clique=true;
  unsigned char from1 = '\0';
  char* p;

  // get running parameters
  while(argc-->=2) {
    p=*++argv;
    switch(p[0]) {
      case '-':
        switch(p[1]) {
          case 'c':
            for_clique=false;
            break;
          case '1':
            from1 = '\0';
            break;
          case 'w':
            weights_name=p+2;
        }
        break;
      case '+':
        switch(p[1]) {
          case 'c':
            for_clique=true;
            break;
          case '1':
            from1 = '\1';
        }
        break;
      default:
        name=p;
    }
  }

  if(name!=NULL) {  // parameters are valid
    printf("%s mode.\n", for_clique?"CLIQUE":"MIS");

    // load the graph from a DIMACS file
    Graph g(name,weights_name,!for_clique);

    // note the start time
    time_t time1,time2;
    time(&time1);

    // preprocess
    vector<int> residual;
    list<int> preselected, clique;
    double clique_weight;
    double preselected_weight = preproc_clique (
      g,residual,preselected,clique_weight,clique
    );

    // if the instance is not reduced completely, apply QUALEX-MS
    if(!residual.empty()) {
      MaxCliqueInfo info(g,for_clique);
      meta_greedy_clique(info);

      int& n=g.n;
      double* a = new double[n*n];

      // experimental: anchor the wrapper on the incumbent clique (see
      // anchor_wrapper()).  Each further pass re-anchors on what the previous
      // one found, and a pass that finds nothing better ends the loop, since
      // the same anchor would only repeat it.  QMS_ANCHOR_WARM keeps the first
      // pass on the standard wrapper and modifies the wrapper from the second
      // on, so that the first anchor is the clique the unmodified method ends
      // with rather than the one the greedy stage found.  It defers
      // QMS_PERTURB in the same way, so with QMS_PERTURB alone it runs the
      // matching control: a second pass on a random wrapper.
      const char* anchor = getenv("QMS_ANCHOR");
      const char* perturb = getenv("QMS_PERTURB");
      bool warm = (anchor!=NULL || perturb!=NULL) &&
        getenv("QMS_ANCHOR_WARM")!=NULL;
      int anchored_passes = anchor==NULL ? 1 :
        (getenv("QMS_ANCHOR_PASSES")?atoi(getenv("QMS_ANCHOR_PASSES")):1);
      if(anchored_passes<1) anchored_passes = 1;
      int n_passes = anchored_passes+(warm?1:0);

      for(int pass=0;pass<n_passes;pass++) {
        build_wrapper(g,info,a);
        bool modify = pass>0 || !warm;

        // experimental: perturb the free entries of the clique wrapper
        if(perturb!=NULL && modify) {
          const char* mode = getenv("QMS_PMODE");
          perturb_wrapper(g,info,a,atof(perturb),
            mode!=NULL && strcmp(mode,"unif")==0,
            getenv("QMS_SEED")?strtoull(getenv("QMS_SEED"),NULL,10):1ULL);
        }

        bool anchored = false;
        if(anchor!=NULL && modify) {
          anchored = anchor_wrapper(g,info,a,atof(anchor));
          if(!anchored && pass>0) break;
          // experimental: blend in one step of lambda_max minimization
          if(anchored && getenv("QMS_ICE")!=NULL)
            ice_step(g,info,a,atof(anchor),getenv("QMS_ICE"));
        }
        double start = info.lower_clique_bound;
        qualex_ms(info,a);
        if(getenv("QMS_STATS")!=NULL)
          fprintf(stderr,"PASS %d anchored=%d %g -> %g\n",
                  pass, (int)anchored, start, info.lower_clique_bound);
        if(anchored && !(info.lower_clique_bound>start)) break;
      }

      delete[] a;

      if(info.lower_clique_bound>clique_weight) {
        clique_weight = info.lower_clique_bound;
        clique.erase(clique.begin(),clique.end());
        for(list<int>::iterator i=info.clique.begin();i!=info.clique.end();i++)
          clique.push_back(residual[*i]);
      }
    }

    // join with the earlier preselected vertices
    clique.splice(clique.begin(),preselected);
    clique_weight += preselected_weight;

    // note the finish time
    time(&time2);

    // print results
    printf (
      "%s: %s_w >= %lg, time=%lg sec.\n",
      name, for_clique?"omega":"alpha", clique_weight, difftime(time2,time1)
    );

    int length=strlen(name);
    char* sol_filename=new char[length+5];
    memcpy(sol_filename, name, length+1);
    char* p=strstr(sol_filename,for_clique?".clq":".mis");
    if(!p)p=sol_filename+length;
    strcpy(p,".sol");
    print_clique(sol_filename,g.header,clique,clique_weight,from1);
    delete[] sol_filename;
  } else puts(
    "Syntax: qualex-ms [<flag>] <dimacs_binary_file> [-w<weights_file>]\n"
    "Flags:\n"
    "+c: look for maximum clique (default)\n"
    "-c: look for maximum independent set\n"
    "+1: vertex numbers in solution file go from 1\n"
    "-1: vertex numbers in solution file go from 0 (default)\n"
    "weights_file: a text file for list of vertex weights (reals)\n"
  );

  return 0;
}
