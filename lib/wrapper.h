/***********************************************************************
!! The clique wrapper is the matrix whose quadratic form the trust    !!
!! region stage of QUALEX-MS works with.  These functions build it    !!
!! and modify its free entries, the ones on non-adjacent vertex pairs !!
!! (see wrapper.cc for what each modification does and what it was    !!
!! measured to buy).  Matrices are n x n, stored by columns.          !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.   !!
***********************************************************************/

#ifndef WRAPPER_H
#define WRAPPER_H

#include "graph.h"

// build_wrapper() fills a with A^(w), the standard clique wrapper less w_min
// on the diagonal, which is the matrix qualex_ms() takes
void build_wrapper(Graph& g, MaxCliqueInfo& info, double* a);

// perturb_wrapper() sets the free entries to -eta u_ij z_i z_j, u_ij uniform
// in [0,1) from seed (or 1 when uniform is set)
void perturb_wrapper(Graph& g, MaxCliqueInfo& info, double* a, double eta,
                     bool uniform, unsigned long long seed);

// anchor_wrapper() lowers free entries so that the incumbent clique meets the
// hypothesis of Theorem 8 (to the fraction theta); false if there is nothing
// to anchor.  report prints the ANCHOR line under QMS_STATS.
bool anchor_wrapper(Graph& g, MaxCliqueInfo& info, double* a, double theta,
                    bool report = true);

// project_wrapper() writes P a P into b, P projecting out z = sqrtw, and
// returns |hatb|
double project_wrapper(MaxCliqueInfo& info, double* a, double* b);

// ice_step() takes one line-searched step of lambda_max minimization, mode
// "lovasz" or "spread"
void ice_step(Graph& g, MaxCliqueInfo& info, double* a, double theta,
              const char* mode, bool anchored);

#endif  // WRAPPER_H
