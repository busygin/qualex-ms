// srank.cc: the exact rank of a SAT01 instance's contradiction constraints for
// bench/minjam.py, which builds it into runs/sat01/bin/srank.
//
// The rank of the contradiction constraints n_i^T Y n_j = 0 ((i,j) a contradiction,
// Y in Sym(d)) and a maximal independent set of them, exactly modulo P = 2^61 - 1, by
// sparse Gaussian elimination with a Markowitz pivot choice (the column with the
// fewest live rows, in it the shortest row; Markowitz cost among the sparsest columns
// otherwise).  A row that is a pivot is independent of the earlier pivots; the pivot
// rows' original indices form a basis of the row space.
//
// Input (binary, little endian): basis file: n, d (int32), then per variable i: k
// (int32) and k pairs (a int32, value uint64 mod P), the row n_i of the basis of N;
// pairs file: count (int32) and count pairs (i, j) (int32).
// Output: rank, d(d+1)/2 - rank (= dim V), and the pivot rows to <out> (int32 list).
// Usage: srank BASIS PAIRS OUT
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using u64 = uint64_t;
using u128 = unsigned __int128;
static const u64 P = (1ULL << 61) - 1;

static inline u64 mulm(u64 a, u64 b) {
  u128 t = (u128)a * b;
  u64 lo = (u64)(t & P), hi = (u64)(t >> 61);
  u64 s = lo + hi;
  return s >= P ? s - P : s;
}
static inline u64 addm(u64 a, u64 b) { u64 s = a + b; return s >= P ? s - P : s; }
static inline u64 subm(u64 a, u64 b) { return a >= b ? a - b : a + P - b; }
static u64 powm(u64 a, u64 e) {
  u64 r = 1;
  while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; }
  return r;
}

struct Entry { uint32_t c; u64 v; };
using Row = std::vector<Entry>;

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: srank BASIS PAIRS OUT\n"); return 1; }
  FILE* f = fopen(argv[1], "rb");
  int32_t n, d;
  if (fread(&n, 4, 1, f) != 1 || fread(&d, 4, 1, f) != 1) return 1;
  std::vector<std::vector<std::pair<int32_t, u64>>> basis(n);
  for (int i = 0; i < n; ++i) {
    int32_t k;
    if (fread(&k, 4, 1, f) != 1) return 1;
    basis[i].resize(k);
    for (int t = 0; t < k; ++t)
      if (fread(&basis[i][t].first, 4, 1, f) != 1 || fread(&basis[i][t].second, 8, 1, f) != 1) return 1;
  }
  fclose(f);
  f = fopen(argv[2], "rb");
  int32_t np;
  if (fread(&np, 4, 1, f) != 1) return 1;
  std::vector<int32_t> pr(2 * (size_t)np);
  if (fread(pr.data(), 4, pr.size(), f) != pr.size()) return 1;
  fclose(f);
  const uint32_t ncol = (uint32_t)d * d;            // column a*d + b, a <= b
  double t0 = now();
  // build the rows: n_i^T Y n_j = sum_{a<=b} Y_ab (n_ia n_jb + [a<b] n_ib n_ja)
  std::vector<Row> rows(np);
  size_t nnz = 0;
  {
    std::vector<u64> acc(ncol, 0);
    std::vector<uint32_t> touched;
    for (int r = 0; r < np; ++r) {
      const auto& bi = basis[pr[2 * r]];
      const auto& bj = basis[pr[2 * r + 1]];
      touched.clear();
      for (auto& x : bi)
        for (auto& y : bj) {
          int a = x.first, b = y.first;
          uint32_t c = a <= b ? (uint32_t)a * d + b : (uint32_t)b * d + a;
          if (acc[c] == 0) touched.push_back(c);
          acc[c] = addm(acc[c], mulm(x.second, y.second));
          if (acc[c] == 0) acc[c] = P;            // keep it marked as touched; P means 0
        }
      std::sort(touched.begin(), touched.end());
      Row& row = rows[r];
      for (uint32_t c : touched) {
        u64 v = acc[c] == P ? 0 : acc[c];
        acc[c] = 0;
        if (v) row.push_back({c, v});
      }
      nnz += row.size();
    }
  }
  fprintf(stderr, "%d rows, %u columns (d=%d), %zu nonzeros, built in %.1fs\n", np, ncol, d, nnz, now() - t0);
  // column structure
  std::vector<std::vector<int32_t>> colrows(ncol);
  std::vector<int32_t> colcnt(ncol, 0);
  for (int r = 0; r < np; ++r)
    for (auto& e : rows[r]) { colrows[e.c].push_back(r); ++colcnt[e.c]; }
  std::vector<char> rowdone(np, 0), coldone(ncol, 0);
  // bucket queue of columns by live count (lazy)
  int maxb = 0;
  for (uint32_t c = 0; c < ncol; ++c) maxb = std::max(maxb, colcnt[c]);
  std::vector<std::vector<uint32_t>> bucket(maxb + 2);
  for (uint32_t c = 0; c < ncol; ++c)
    if (colcnt[c] > 0) bucket[colcnt[c]].push_back(c);
  auto push = [&](uint32_t c) {
    int k = colcnt[c];
    if (k <= 0 || coldone[c]) return;
    if (k >= (int)bucket.size()) bucket.resize(k + 1);
    bucket[k].push_back(c);
  };
  auto has = [&](const Row& row, uint32_t c, u64* v) {
    auto it = std::lower_bound(row.begin(), row.end(), c, [](const Entry& e, uint32_t x) { return e.c < x; });
    if (it != row.end() && it->c == c) { if (v) *v = it->v; return true; }
    return false;
  };
  std::vector<int32_t> pivrows;
  std::vector<int32_t> cand;
  Row tmp;
  size_t live_nnz = nnz;
  int b = 1;
  double tlast = now();
  while (true) {
    // the column with the fewest live rows
    uint32_t c = 0;
    bool found = false;
    for (b = 1; b < (int)bucket.size() && !found; ++b) {
      auto& bk = bucket[b];
      while (!bk.empty()) {
        uint32_t x = bk.back();
        bk.pop_back();
        if (!coldone[x] && colcnt[x] == b) { c = x; found = true; break; }
      }
      if (found) break;
    }
    if (!found) break;
    // its live rows; the shortest is the pivot
    cand.clear();
    std::vector<int32_t>& cr = colrows[c];
    for (int32_t r : cr)
      if (!rowdone[r] && has(rows[r], c, nullptr)) cand.push_back(r);
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    cr.clear();
    if (cand.empty()) { coldone[c] = 1; colcnt[c] = 0; continue; }
    int32_t piv = cand[0];
    for (int32_t r : cand) if (rows[r].size() < rows[piv].size()) piv = r;
    Row& pr_ = rows[piv];
    u64 pv;
    has(pr_, c, &pv);
    u64 inv = powm(pv, P - 2);
    for (int32_t s : cand) {
      if (s == piv) continue;
      Row& rs = rows[s];
      u64 sv;
      has(rs, c, &sv);
      u64 fct = mulm(sv, inv);
      // rs <- rs - fct * pr_
      tmp.clear();
      size_t i = 0, j = 0;
      while (i < rs.size() || j < pr_.size()) {
        if (j == pr_.size() || (i < rs.size() && rs[i].c < pr_[j].c)) { tmp.push_back(rs[i]); ++i; }
        else if (i == rs.size() || pr_[j].c < rs[i].c) {
          uint32_t k = pr_[j].c;
          tmp.push_back({k, subm(0, mulm(fct, pr_[j].v))});
          colrows[k].push_back(s); ++colcnt[k]; push(k);
          ++j;
        } else {
          u64 v = subm(rs[i].v, mulm(fct, pr_[j].v));
          if (v) tmp.push_back({rs[i].c, v});
          else { --colcnt[rs[i].c]; push(rs[i].c); }
          ++i; ++j;
        }
      }
      live_nnz += tmp.size();
      live_nnz -= rs.size();
      rs.swap(tmp);
      if (rs.empty()) rowdone[s] = 1;           // dependent on the pivots so far
    }
    // retire the pivot row and the column
    for (auto& e : pr_) { --colcnt[e.c]; push(e.c); }
    live_nnz -= pr_.size();
    Row().swap(pr_);
    rowdone[piv] = 1;
    coldone[c] = 1;
    colcnt[c] = 0;
    pivrows.push_back(piv);
    if (now() - tlast > 30) {
      tlast = now();
      fprintf(stderr, "  %zu pivots, live nonzeros %zu, column bucket %d, %.0fs\n", pivrows.size(), live_nnz, b, now() - t0);
    }
  }
  size_t N = (size_t)d * (d + 1) / 2;
  printf("rank %zu of %d rows; d(d+1)/2 = %zu, null space dimension %zu; %.1fs\n", pivrows.size(), np, N, N - pivrows.size(), now() - t0);
  f = fopen(argv[3], "wb");
  int32_t cnt = (int32_t)pivrows.size();
  fwrite(&cnt, 4, 1, f);
  std::sort(pivrows.begin(), pivrows.end());
  fwrite(pivrows.data(), 4, pivrows.size(), f);
  fclose(f);
  return 0;
}
