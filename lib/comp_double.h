#ifndef COMP_DOUBLE_H
#define COMP_DOUBLE_H

// less_double and greater_double order vertex numbers by the values x gives
// them, for sorting by an "appealing" vector

struct less_double {
  double* x;
  less_double(double* _x): x(_x) {}
  bool operator()(const int i, const int j) const { return x[i]<x[j]; }
};

struct greater_double {
  double* x;
  greater_double(double* _x): x(_x) {}
  bool operator()(const int i, const int j) const { return x[i]>x[j]; }
};

#endif	// COMP_DOUBLE_H
