/***********************************************************************
!! QUALEX-MS: a QUick ALmost EXact Motzkin-Straus maximum weight      !!
!! clique/independent set solver.                                     !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2007. All rights reserved.   !!
!!                                                                    !!
!! Qualex header                                                      !!
***********************************************************************/

#ifndef QUALEX_H
#define QUALEX_H

#include "dr.h"
#include "graph.h"

// qualex_ms() runs the trust region stage on the wrapper a, n x n less w_min
// on the diagonal (see build_wrapper()), offering the cliques it finds to
// graph_info.  A positive target is the weight of the clique sought, when it
// is known: the stationary points are then taken at the radius of a clique of
// that weight with the method of version 1.2 (see qualex.cc), positive
// multipliers only and Meta-NBIW only if QMS_META_N asks for it, instead of
// around the radius of a clique one w_min heavier than the incumbent.
// dr, unless null, says where the Douglas-Rachford stage (QMS_DR) drives the
// points of its spheres: its projection replaces the one onto the nonnegative
// orthant, and its surfaces are further sets the points have to lie on (see
// DRTargets).
bool qualex_ms(MaxCliqueInfo& graph_info, double* a, double target = 0.0,
               const DRTargets* dr = nullptr);

#endif  // QUALEX_H
