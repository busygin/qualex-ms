/***********************************************************************
!! linalg.h is the dense linear algebra of the spectral stages.  One  !!
!! of two backends implements it: linalg_cpu.c (LAPACK and CBLAS) or  !!
!! linalg_gpu.c (cuSOLVER and cuBLAS), chosen when the library is     !!
!! built.  Matrices are stored by columns.                            !!
***********************************************************************/

#ifndef LINALG_H
#define LINALG_H

#ifdef __cplusplus
extern "C" {
#endif

// symmetric_eigen() puts the eigenvalues of the symmetric n x n matrix a in
// lambda, ascending, and keeps its eigenvectors for extract_eigenvectors();
// a is left unchanged.  Returns the info code of the eigensolver (0 is
// success).
int symmetric_eigen(int n, const double* a, double* lambda);

// symmetric_eigenvalues() is symmetric_eigen() without the eigenvectors; it
// leaves the eigenvectors symmetric_eigen() kept alone
int symmetric_eigenvalues(int n, const double* a, double* lambda);

// extract_eigenvectors() keeps the first n_low and the last n_high
// eigenvectors of the last symmetric_eigen() as Q, n x k with
// k = n_low+n_high, for the products below, and copies Q into q
void extract_eigenvectors(int n, int n_low, int n_high, double* q);

// eigenvectors_dot_vector() is y = Q x for trans 'N' (x of length k) and
// y = Q^T x for trans 'T' (x of length n)
void eigenvectors_dot_vector(int n, int k, char trans, const double* x, double* y);

// project_on_eigenvectors() is c = Q^T b, c also kept for projection_norm()
void project_on_eigenvectors(int n, int k, const double* b, double* c);

// projection_norm() is |c| of the last project_on_eigenvectors()
double projection_norm(int k);

// matrix_dot_vector() is y = A x for trans 'N' and y = A^T x for trans 'T',
// A being m x n
void matrix_dot_vector(int m, int n, const double* a, char trans,
                       const double* x, double* y);

double dot_product(int n, const double* x, const double* y);

#ifdef __cplusplus
}
#endif

#endif  // LINALG_H
