// qmsmin.cc: QUALEX-MS's MIN refinement, refine_clique_MIN_w() of lib/refiner.cc,
// callable from Python through ctypes (bench/jam.py builds it into
// runs/jam/bin/libqmsmin.so with the library's combinatorial core).
//
// qms_new() builds the solver's own Graph and MaxCliqueInfo from a dense 0/1 adjacency
// matrix and the vertex weights; qms_min() hands an "appealing" vector to the unchanged
// refine_clique_MIN_w() and returns the weight of the clique it builds.  The known
// clique bound is set to infinity so that receive_clique() never stores or prints.

#include <math.h>
#include <string.h>
#include <vector>

#include "graph.h"
#include "refiner.h"

struct Handle {
  Graph g;
  MaxCliqueInfo* info;
  std::vector<double> x;
  Handle(int n) : g(n), info(nullptr), x(n) {}
};

extern "C" {

void* qms_new(int n, const unsigned char* adj, const double* weights) {
  Handle* h = new Handle(n);
  for (int i = 0; i < n; ++i) {
    h->g.weights[i] = weights[i];
    for (int j = i + 1; j < n; ++j)
      if (adj[(size_t)i * n + j]) h->g.add_edge(i, j);
  }
  h->info = new MaxCliqueInfo(h->g, true);
  h->info->lower_clique_bound = HUGE_VAL;
  return h;
}

// the weight of the MIN clique built from the appealing vector a (a is not modified)
double qms_min(void* hp, const double* a) {
  Handle* h = (Handle*)hp;
  memcpy(&h->x[0], a, sizeof(double) * h->g.n);
  double weight;
  refine_clique_MIN_w(*h->info, &h->x[0], weight);
  return weight;
}

void qms_free(void* hp) {
  Handle* h = (Handle*)hp;
  delete h->info;
  delete h;
}

}
