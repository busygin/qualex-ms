/***********************************************************************
!! refine_clique() provides a maximal clique by a vertex "appealing"  !!
!! vector.                                                            !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.   !!
!!                                                                    !!
!! refine_clique() implementation                                     !!
***********************************************************************/

#include <string.h>
#include <algorithm>

#include "greedy_clique.h"
#include "refiner.h"

#include "comp_double.h"

using namespace std;

// refine_clique_VO() provides a maximal clique by a vertex
// "appealing" vector x using Vetex Order procedure.
// Returns true if the known clique was improved
bool refine_clique_VO(MaxCliqueInfo& graph_info, double* x) {
  int& n = graph_info.g.n;
  list<int> clique;
  vector<int> sort_nos(n);
  for(int i=0;i<n;++i) sort_nos[i]=i;
  sort(sort_nos.begin(), sort_nos.end(), greater_double(x));
  for(int i : sort_nos) {
    bool_vector& mates = graph_info.g.mates[i];
    bool fits = true;
    for(int v : clique)
      if(!mates.at(v)) { fits = false; break; }
    if(fits) clique.push_back(i);
  }
  return graph_info.receive_clique(clique);
}

// refine_clique_MIN() provides a maximal clique of by a vertex
// "appealing" vector x using MIN procedure.
// Returns true if the known clique was improved
bool refine_clique_MIN(MaxCliqueInfo& graph_info, double* x) {
  double weight;
  return refine_clique_MIN_w(graph_info,x,weight);
}

bool refine_clique_MIN_w(MaxCliqueInfo& graph_info, double* x, double& weight) {
  int& n = graph_info.g.n;
  double* w = new double[n];
  neighborhood_weights(graph_info.g,x,w);
  vector<int> active_vertices(n);
  for(int i=0;i<n;++i) active_vertices[i] = i;
  list<int> clique;
  greedy_clique(graph_info.g,active_vertices,x,w,clique);
  delete[] w;
  weight = 0.0;
  for(int v : clique) weight += graph_info.g.weights[v];
  return graph_info.receive_clique(clique);
}
