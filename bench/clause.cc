// clause.cc: the greedy 2-clause projection of many points at once for the
// Douglas-Rachford runs of bench/minjam.py, which builds it into
// runs/sat01/bin/libclause.so.
//
// The greedy 2-clause projection of each column of Y (n x nb, by columns), in place,
// as bench/jam.py's clause_projector(): clip the negatives, keep the entries in
// decreasing order unless they 2-clash with a kept one.  ptr/idx hold the 2-clause
// neighbours of every variable (CSR).
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" void clause_rows(int n, int nb, double* Y, const int* ptr, const int* idx) {
#pragma omp parallel
  {
    std::vector<int> ord(n);
    std::vector<char> blocked(n);
    std::vector<double> out(n);
#pragma omp for schedule(dynamic)
    for (int t = 0; t < nb; ++t) {
      double* y = Y + (size_t)t * n;
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(), [y](int a, int b) { return y[a] > y[b]; });
      std::fill(blocked.begin(), blocked.end(), 0);
      std::fill(out.begin(), out.end(), 0.0);
      for (int i : ord) {
        if (y[i] <= 0) break;
        if (blocked[i]) continue;
        out[i] = y[i];
        for (int p = ptr[i]; p < ptr[i + 1]; ++p) blocked[idx[p]] = 1;
      }
      std::memcpy(y, out.data(), n * sizeof(double));
    }
  }
}
