// linalg_cpu.c: the linalg.h backend on LAPACK (DSYEVR) and CBLAS.

#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "linalg.h"

// LAPACK's DSYEVR, which computes eigenpairs by Relatively Robust
// Representations after reducing the matrix to tridiagonal form (Dhillon, "A
// new O(n^2) algorithm for the symmetric tridiagonal eigenvalue/eigenvector
// problem", UC Berkeley CSD-97-971, 1997).  The trailing arguments are the
// hidden lengths of the character arguments in the gfortran calling convention.
extern void dsyevr_(const char* jobz, const char* range, const char* uplo,
                    const int* n, double* a, const int* lda,
                    const double* vl, const double* vu, const int* il, const int* iu,
                    const double* abstol, int* m, double* w, double* z, const int* ldz,
                    int* isuppz, double* work, const int* lwork,
                    int* iwork, const int* liwork, int* info,
                    size_t jobz_len, size_t range_len, size_t uplo_len);

static double* eigvecs = NULL;  // all eigenvectors of the last symmetric_eigen()
static size_t eigvecs_size = 0;
static double* q_kept = NULL;   // the ones extract_eigenvectors() kept
static size_t q_size = 0;
static double* c_kept = NULL;   // the last projection
static int c_size = 0;

// ensure() makes *p hold at least size doubles
static void ensure(double** p, size_t* capacity, size_t size) {
  if (size > *capacity) {
    free(*p);
    *p = (double*)malloc(sizeof(double) * size);
    *capacity = size;
  }
}

// dsyevr() runs DSYEVR on a copy of a, the eigenvectors going to z unless it
// is NULL
static int dsyevr(int n, const double* a, double* lambda, double* z) {
  char jobz = z ? 'V' : 'N', range = 'A', uplo = 'U';
  double vl = 0.0, vu = 0.0, abstol = 0.0;
  int il = 0, iu = 0, m = 0, info = 0;
  size_t nn = (size_t)n * n;
  double* w = (double*)malloc(sizeof(double) * nn);
  memcpy(w, a, sizeof(double) * nn);
  int* isuppz = (int*)malloc(sizeof(int) * 2 * (size_t)n);
  double z_dummy;
  double* zz = z ? z : &z_dummy;

  // workspace query
  double work_size;
  int iwork_size, lwork = -1, liwork = -1;
  dsyevr_(&jobz, &range, &uplo, &n, w, &n, &vl, &vu, &il, &iu, &abstol, &m,
          lambda, zz, &n, isuppz, &work_size, &lwork, &iwork_size, &liwork, &info,
          1, 1, 1);
  lwork = (int)work_size;
  liwork = iwork_size;
  double* work = (double*)malloc(sizeof(double) * lwork);
  int* iwork = (int*)malloc(sizeof(int) * liwork);
  dsyevr_(&jobz, &range, &uplo, &n, w, &n, &vl, &vu, &il, &iu, &abstol, &m,
          lambda, zz, &n, isuppz, work, &lwork, iwork, &liwork, &info,
          1, 1, 1);

  free(iwork);
  free(work);
  free(isuppz);
  free(w);
  return info;
}

int symmetric_eigen(int n, const double* a, double* lambda) {
  ensure(&eigvecs, &eigvecs_size, (size_t)n * n);
  return dsyevr(n, a, lambda, eigvecs);
}

int symmetric_eigenvalues(int n, const double* a, double* lambda) {
  return dsyevr(n, a, lambda, NULL);
}

void extract_eigenvectors(int n, int n_low, int n_high, double* q) {
  size_t k = (size_t)(n_low + n_high);
  ensure(&q_kept, &q_size, (size_t)n * k);
  memcpy(q_kept, eigvecs, sizeof(double) * (size_t)n * n_low);
  memcpy(q_kept + (size_t)n * n_low, eigvecs + (size_t)n * (n - n_high),
         sizeof(double) * (size_t)n * n_high);
  memcpy(q, q_kept, sizeof(double) * (size_t)n * k);
}

void eigenvectors_dot_vector(int n, int k, char trans, const double* x, double* y) {
  cblas_dgemv(CblasColMajor, trans == 'N' ? CblasNoTrans : CblasTrans,
              n, k, 1.0, q_kept, n, x, 1, 0.0, y, 1);
}

void project_on_eigenvectors(int n, int k, const double* b, double* c) {
  if (k > c_size) {
    free(c_kept);
    c_kept = (double*)malloc(sizeof(double) * k);
    c_size = k;
  }
  cblas_dgemv(CblasColMajor, CblasTrans, n, k, 1.0, q_kept, n, b, 1, 0.0, c_kept, 1);
  memcpy(c, c_kept, sizeof(double) * k);
}

double projection_norm(int k) {
  return cblas_dnrm2(k, c_kept, 1);
}

void matrix_dot_vector(int m, int n, const double* a, char trans,
                       const double* x, double* y) {
  cblas_dgemv(CblasColMajor, trans == 'N' ? CblasNoTrans : CblasTrans,
              m, n, 1.0, a, m, x, 1, 0.0, y, 1);
}

double dot_product(int n, const double* x, const double* y) {
  return cblas_ddot(n, x, 1, y, 1);
}
