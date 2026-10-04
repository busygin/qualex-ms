/***********************************************************************
!! preproc_clique() preprocesses a graph for maximum weight clique    !!
!! finding.                                                           !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2001, 2002. All rights reserved.  !!
!!                                                                    !!
!! preproc_clique() implementation                                    !!
***********************************************************************/

#include <float.h>

#include "bool_vector.h"
#include "graph.h"
#include "greedy_clique.h"
#include "preproc_clique.h"
#include "vec_del.h"

using namespace std;

// preproc_clique() preprocesses a graph for maximum weight clique finding.
// Each vertex disconnected only with a subset weighting not more than it
// becomes preselected. On the base of MIN heuristic result, too low
// connected vertices are removed.
// It returns in the corresponding parameters: the residual set of graph
// vertices to submit to a next maximum weight clique routine, the
// preselected vertex set, the discovered lower bound of maximum weight
// clique and a clique giving this bound itself. The function result is
// the total weight of preselected vertices.
double preproc_clique (
  Graph& g, vector<int>& residual, list<int>& preselected,
  double& known_bound, list<int>& clique
) {
  int& n=g.n;
  double* w = new double[n];
  int i;
  neighborhood_weights(g,&(g.weights[0]),w);

  // Both tests below compare quantities accumulated by summing up to n vertex
  // weights, so their rounding error is of order n*eps*W(V), not the single
  // ulp the comparisons used to allow for.  Weights are usually integers, in
  // which case the sums are exact and this tolerance never fires; it matters
  // when they are reals, where a single ulp is far too tight to cover the
  // accumulation.
  double weight_tol = 0.0;
  for(i=0;i<n;++i) weight_tol += g.weights[i];
  weight_tol *= (double)n*DBL_EPSILON;
  residual.resize(n);
  for(i=0;i<n;++i) residual[i] = i;
  bool_vector remove_flag(n);

  double result = 0.0;
  list<int> consider;
  bool_vector considered_flag(n);
  bool reduction_flag;
  known_bound = 0.0;
  do {
    reduction_flag = false;
    clique.clear();
    greedy_clique(g,residual,&(g.weights[0]),w,clique);
    double clique_weight = 0.0;
    for(int v : clique) clique_weight += g.weights[v];
    if(clique_weight <= known_bound) break;
    known_bound = clique_weight;
    for(int v : residual) {
      considered_flag.clear(v);
      consider.push_front(v);
    }
    while(!consider.empty()) {
      do {
        i = consider.front();
        consider.pop_front();
        considered_flag.put(i);
        if(w[i] < known_bound-weight_tol) {
          vec_del(residual,i);
          remove_flag.put(i);
          for(int j : g.mates[i].ones()) {
            if(!remove_flag.at(j)) {
              w[j] -= g.weights[i];
              if(considered_flag.at(j)) {
                considered_flag.clear(j);
                consider.push_back(j);
              }
            }
          }
          reduction_flag = true;
        }
      } while(!consider.empty());
      double total_weight = 0.0;
      for(int v : residual) total_weight += g.weights[v];
      for(int t=(int)residual.size()-1;t>=0;--t) {
        i = residual[t];
        double weight_i = g.weights[i];
        if(g.weights[i] >= total_weight - w[i] - weight_tol) {
          known_bound -= weight_i;
          if(known_bound < 0.0) known_bound = 0.0;
          bool_vector& mates_i = g.mates[i];
          for(int j : residual) {
            if(!mates_i.at(j) && considered_flag.at(j)) {
              w[j] = -1.0;
              considered_flag.clear(j);
              consider.push_back(j);
            }
          }
          clique.clear();
          preselected.push_back(i);
          result += weight_i;
        }
      }
    }
  } while(reduction_flag && !residual.empty());
  delete[] w;
  for(i=n-1;i>=0;--i) if(remove_flag.at(i)) g.remove_vertex(i);
  return result;
}
