/***********************************************************************
!! greedy_clique.cc contains the implementation of basic greedy       !!
!! algorithms for maximum weight clique finding.                      !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2001-2007. All rights reserved.   !!
!!                                                                    !!
!! This is greedy_clique header.                                      !!
***********************************************************************/

#ifndef GREEDY_CLIQUE_H
#define GREEDY_CLIQUE_H

#include <vector>
#include <list>
#include "graph.h"

using namespace std;

void neighborhood_weights(Graph& g, double* vert_weights, double* neigh_weights);

void greedy_clique (
  Graph& g, vector<int> act_verts, double* vert_weights,
  double* neigh_weights, list<int>& clique
);

bool meta_greedy_clique(MaxCliqueInfo& graph_info);

// meta_refine_MIN() is Meta-NBIW (Algorithm 3) driven by an arbitrary vertex
// "appealing" vector x instead of the vertex weights.  NBIW is restarted from
// each of the n_starts vertices x rates highest (all of them if n_starts<=0).
bool meta_refine_MIN (
  MaxCliqueInfo& graph_info, double* x, double& best_weight, int n_starts
);

bool meta_greedy_clique(MaxCliqueInfo& graph_info, double* a);

#endif  // GREEDY_CLIQUE_H
