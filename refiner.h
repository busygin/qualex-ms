/***********************************************************************
!! refine_clique() provides a maximal clique by a vertex "appealing"  !!
!! vector.                                                            !!
!!                                                                    !!
!! Copyright (c) Stanislav Busygin, 2000-2003. All rights reserved.   !!
!!                                                                    !!
!! refine_clique() header                                             !!
***********************************************************************/

#ifndef REFINER_H
#define REFINER_H

#include "graph.h"

// refine_clique_VO() provides a maximal clique by a vertex
// "appealing" vector x using Vetex Order procedure.
// Returns true if the known clique was improved
bool refine_clique_VO(MaxCliqueInfo& graph_info, double* x);

// refine_clique_MIN() provides a maximal clique of by a vertex
// "appealing" vector x using MIN procedure.
// Returns true if the known clique was improved
bool refine_clique_MIN(MaxCliqueInfo& graph_info, double* x);

// refine_clique_MIN_w() is refine_clique_MIN() also reporting the total
// weight of the clique it has built (regardless of whether it is an
// improvement of the known one)
bool refine_clique_MIN_w(MaxCliqueInfo& graph_info, double* x, double& weight);

#endif  // REFINER_H
