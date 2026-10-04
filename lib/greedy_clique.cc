/***********************************************************************
!! greedy_clique.cc contains the implementation of basic greedy       !!
!! algorithms for maximum weight clique finding.                      !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2001-2007. All rights reserved.   !!
!!                                                                    !!
***********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include "comp_double.h"
#include "greedy_clique.h"

using namespace std;

void neighborhood_weights(Graph& g, double* vert_weights, double* neigh_weights) {
  int& n=g.n;
  memcpy(neigh_weights,vert_weights,sizeof(double)*n);
  for(int i=0;i<n;++i) {
    for(int j : g.mates[i].ones()) {
      if(j>i) break;
      neigh_weights[i] += vert_weights[j];
      neigh_weights[j] += vert_weights[i];
    }
  }
}

void clear_act_verts (
  Graph& g, vector<int>& act_verts, double* vert_weights,
  double* neigh_weights, int sel_vert
) {
  bool_vector& sel_mates = g.mates[sel_vert];
  for(int t=(int)act_verts.size()-1;t>=0;--t) {
    int i = act_verts[t];
    if(!sel_mates.at(i)) {
      act_verts.erase(act_verts.begin()+t);
      bool_vector& mates_i = g.mates[i];
      for(int j : act_verts) {
        if(mates_i.at(j)) neigh_weights[j] -= vert_weights[i];
      }
    }
  }
}

int greedy_choice (
  Graph& g, vector<int>& act_verts, double* vert_weights, double* neigh_weights
) {
  if(act_verts.empty()) return -1;
  vector<int>::iterator i_sel_vert = max_element (
    act_verts.begin(), act_verts.end(), less_double(neigh_weights)
  );
  int sel_vert = *i_sel_vert;
  act_verts.erase(i_sel_vert);
  clear_act_verts(g, act_verts, vert_weights, neigh_weights, sel_vert);
  return sel_vert;
}

void greedy_clique (
  Graph& g, vector<int> act_verts, double* vert_weights,
  double* neigh_weights, list<int>& clique
) {
  double* neigh_weights1 = new double[g.n];
  memcpy(neigh_weights1,neigh_weights,sizeof(double)*g.n);
  int i;
  while((i=greedy_choice(g,act_verts,vert_weights,neigh_weights1)) != -1)
    clique.push_back(i);
  delete[] neigh_weights1;
}

bool meta_greedy_clique(MaxCliqueInfo& graph_info) {
  int& n = graph_info.g.n;
  double* neigh_weights = new double[n];
  neighborhood_weights(graph_info.g,&(graph_info.g.weights[0]),neigh_weights);
  vector<int> act_verts;
  double* neigh_weights1 = new double[n];
  list<int> clique;
  bool result = false;
  for(int i=0;i<n;++i) {
    memcpy(neigh_weights1,neigh_weights,sizeof(double)*n);
    clique.clear();
    act_verts.clear();
    int j;
    for(j=0;j<n;++j) if(i!=j) act_verts.push_back(j);
    clear_act_verts(graph_info.g,act_verts,&(graph_info.g.weights[0]),neigh_weights1,i);
    clique.push_back(i);
    while((j=greedy_choice(graph_info.g,act_verts,&(graph_info.g.weights[0]),neigh_weights1)) != -1)
      clique.push_back(j);
    result |= graph_info.receive_clique(clique);
  }
  delete[] neigh_weights;
  delete[] neigh_weights1;
  return result;
}

// meta_refine_MIN() is the Meta-NBIW algorithm (Algorithm 3) driven by an
// arbitrary vertex "appealing" vector x instead of the vertex weights: NBIW
// is started from a vertex, ranking candidates by x, and that is repeated
// over a set of starting vertices.  best_weight receives the heaviest clique
// it built, whether or not it is an improvement.
// Algorithm 3 starts from every vertex, which costs O(n^3).  Since x is the
// stationary point's own opinion of how promising each vertex is, restricting
// the starts to the n_starts vertices it rates highest buys the same coverage
// of the plausible cliques far more cheaply; n_starts<=0 means all of them.
bool meta_refine_MIN (
  MaxCliqueInfo& graph_info, double* x, double& best_weight, int n_starts
) {
  int& n = graph_info.g.n;
  double* neigh_weights = new double[n];
  neighborhood_weights(graph_info.g,x,neigh_weights);
  double* neigh_weights1 = new double[n];
  vector<int> act_verts;
  list<int> clique;
  bool result = false;
  best_weight = 0.0;

  vector<int> starts(n);
  for(int i=0;i<n;++i) starts[i] = i;
  if(n_starts>0 && n_starts<n) {
    partial_sort(starts.begin(),starts.begin()+n_starts,starts.end(),
                 greater_double(x));
    starts.resize(n_starts);
  }

  for(int i : starts) {
    memcpy(neigh_weights1,neigh_weights,sizeof(double)*n);
    clique.clear();
    act_verts.clear();
    int j;
    for(j=0;j<n;++j) if(i!=j) act_verts.push_back(j);
    clear_act_verts(graph_info.g,act_verts,x,neigh_weights1,i);
    clique.push_back(i);
    while((j=greedy_choice(graph_info.g,act_verts,x,neigh_weights1)) != -1)
      clique.push_back(j);
    double wgt = 0.0;
    for(int v : clique) wgt += graph_info.g.weights[v];
    if(wgt>best_weight) best_weight = wgt;
    result |= graph_info.receive_clique(clique);
  }
  delete[] neigh_weights;
  delete[] neigh_weights1;
  return result;
}

// The variant of Meta-NBIW below ranks the vertices by a matrix a instead of
// a vector: the neighborhood weight of vertex i is row i of a weighted by
// sqrtw.  It is not used by the solver at present.

void neighborhood_weights (
  MaxCliqueInfo& graph_info, double* a, double* neigh_weights
) {
  int& n = graph_info.g.n;
  for(int i=0;i<n;++i) {
    double s = 0.0;
    for(int j=0;j<n;++j) s += graph_info.sqrtw[j]*a[i*n+j];
    neigh_weights[i] = s/graph_info.sqrtw[i];
  }
}

void clear_act_verts (
  MaxCliqueInfo& graph_info, vector<int>& act_verts, double* a,
  double* neigh_weights, int sel_vert
) {
  bool_vector& sel_mates = graph_info.g.mates[sel_vert];
  for(int t=(int)act_verts.size()-1;t>=0;--t) {
    int i = act_verts[t];
    if(!sel_mates.at(i)) {
      act_verts.erase(act_verts.begin()+t);
      for(int j : act_verts) {
        neigh_weights[j] -=
          graph_info.sqrtw[j]*a[i*graph_info.g.n+j]/graph_info.sqrtw[i];
      }
    }
  }
}

int greedy_choice (
  MaxCliqueInfo& graph_info, vector<int>& act_verts, double* a,
  double* neigh_weights
) {
  if(act_verts.empty()) return -1;
  vector<int>::iterator i_sel_vert = max_element (
    act_verts.begin(), act_verts.end(), less_double(neigh_weights)
  );
  int sel_vert = *i_sel_vert;
  act_verts.erase(i_sel_vert);
  clear_act_verts(graph_info, act_verts, a, neigh_weights, sel_vert);
  return sel_vert;
}

bool meta_greedy_clique(MaxCliqueInfo& graph_info, double* a) {
  int& n = graph_info.g.n;
  double* neigh_weights = new double[n];
  neighborhood_weights(graph_info,a,neigh_weights);
  vector<int> act_verts;
  double* neigh_weights1 = new double[n];
  list<int> clique;
  bool result = false;
  for(int i=0;i<n;++i) {
    memcpy(neigh_weights1,neigh_weights,sizeof(double)*n);
    clique.clear();
    act_verts.clear();
    int j;
    for(j=0;j<n;++j) if(i!=j) act_verts.push_back(j);
    clear_act_verts(graph_info.g,act_verts,a,neigh_weights1,i);
    clique.push_back(i);
    while((j=greedy_choice(graph_info,act_verts,a,neigh_weights1)) != -1)
      clique.push_back(j);
    result |= graph_info.receive_clique(clique);
  }
  delete[] neigh_weights;
  delete[] neigh_weights1;
  return result;
}
