// linalg_gpu.c: the linalg.h backend on cuSOLVER and cuBLAS.  The eigenvectors
// stay on the GPU between symmetric_eigen() and the products with Q.

#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "linalg.h"

// Persistent GPU state
static cublasHandle_t cublas_handle = NULL;
static cusolverDnHandle_t cusolver_handle = NULL;
static cudaStream_t gpu_stream = NULL;
static double* d_q_persistent = NULL;
static int q_rows = 0, q_cols = 0;

// Pre-allocated work buffers
static double* d_work_vec1 = NULL;
static double* d_work_vec2 = NULL;
static int work_size = 0;

// GPU-resident coefficient vector c
static double* d_c_persistent = NULL;
static int c_size = 0;

static void ensure_gpu_init() {
  if (cublas_handle == NULL) {
    cublasCreate(&cublas_handle);
    cusolverDnCreate(&cusolver_handle);
    cudaStreamCreate(&gpu_stream);
    cublasSetStream(cublas_handle, gpu_stream);
    cusolverDnSetStream(cusolver_handle, gpu_stream);
  }
}

static void ensure_work_buffers(int n) {
  if (n > work_size) {
    if (d_work_vec1) cudaFree(d_work_vec1);
    if (d_work_vec2) cudaFree(d_work_vec2);
    cudaMalloc((void**)&d_work_vec1, n * sizeof(double));
    cudaMalloc((void**)&d_work_vec2, n * sizeof(double));
    work_size = n;
  }
}

// Matrix-vector multiply using GPU-resident Q: y = op(Q) * x
void eigenvectors_dot_vector(int n, int k, char trans, const double* x, double* y) {
  double alpha = 1.0, beta = 0.0;
  cublasOperation_t op = (trans == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  int x_len = (trans == 'N') ? k : n;
  int y_len = (trans == 'N') ? n : k;

  cudaMemcpy(d_work_vec1, x, x_len * sizeof(double), cudaMemcpyHostToDevice);
  cublasDgemv(cublas_handle, op, n, k, &alpha, d_q_persistent, n, d_work_vec1, 1, &beta, d_work_vec2, 1);
  cudaMemcpy(y, d_work_vec2, y_len * sizeof(double), cudaMemcpyDeviceToHost);
}

// Compute c = Q^T * b and store c on GPU, also copy to CPU
void project_on_eigenvectors(int n, int k, const double* b, double* c) {
  ensure_gpu_init();
  double alpha = 1.0, beta = 0.0;

  // Allocate/reallocate persistent c buffer if needed
  if (k > c_size) {
    if (d_c_persistent) cudaFree(d_c_persistent);
    cudaMalloc((void**)&d_c_persistent, k * sizeof(double));
    c_size = k;
  }

  // Copy b to GPU work buffer
  cudaMemcpy(d_work_vec1, b, n * sizeof(double), cudaMemcpyHostToDevice);

  // Compute c = Q^T * b, store result in d_c_persistent
  cublasDgemv(cublas_handle, CUBLAS_OP_T, n, k, &alpha, d_q_persistent, n, d_work_vec1, 1, &beta, d_c_persistent, 1);

  // Copy c to CPU for use in solver
  cudaMemcpy(c, d_c_persistent, k * sizeof(double), cudaMemcpyDeviceToHost);
}

// Compute norm2 of GPU-resident c vector (no CPU-GPU transfer needed)
double projection_norm(int k) {
  double result;
  cublasDnrm2(cublas_handle, k, d_c_persistent, 1, &result);
  return result;
}

// Temporary storage for eigenvectors on GPU (between eigen and extract)
static double* d_eigenvectors_temp = NULL;
static int eigenvectors_temp_n = 0;

// Eigendecomposition using cuSOLVER - keeps eigenvectors on GPU
// Only eigenvalues are copied to CPU; eigenvectors remain in d_eigenvectors_temp
int symmetric_eigen(int n, const double* a, double* lambda) {
  ensure_gpu_init();

  double* d_lambda = NULL;
  int* d_info = NULL;
  double* d_work = NULL;
  int lwork = 0;
  int info = 0;

  // Allocate/reallocate temp eigenvector storage if needed
  if (n > eigenvectors_temp_n) {
    if (d_eigenvectors_temp) cudaFree(d_eigenvectors_temp);
    cudaMalloc((void**)&d_eigenvectors_temp, sizeof(double) * n * n);
    eigenvectors_temp_n = n;
  }

  cudaMalloc((void**)&d_lambda, sizeof(double) * n);
  cudaMalloc((void**)&d_info, sizeof(int));

  cudaMemcpy(d_eigenvectors_temp, a, sizeof(double) * n * n, cudaMemcpyHostToDevice);

  cusolverDnDsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER, n, d_eigenvectors_temp, n, d_lambda, &lwork);

  cudaMalloc((void**)&d_work, sizeof(double) * lwork);

  cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER, n, d_eigenvectors_temp, n, d_lambda, d_work, lwork, d_info);

  cudaStreamSynchronize(gpu_stream);

  // Only copy eigenvalues to CPU - eigenvectors stay on GPU
  cudaMemcpy(lambda, d_lambda, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_work);
  cudaFree(d_info);
  cudaFree(d_lambda);

  return info;
}

// Extract selected eigenvector columns directly on GPU to d_q_persistent
// Copies first n_low columns and last n_high columns from d_eigenvectors_temp
void extract_eigenvectors(int n, int n_low, int n_high, double* q_cpu) {
  int k = n_low + n_high;

  // Allocate persistent Q storage
  if (d_q_persistent) cudaFree(d_q_persistent);
  q_rows = n;
  q_cols = k;
  cudaMalloc((void**)&d_q_persistent, (size_t)n * k * sizeof(double));

  // Copy first n_low columns (negative eigenvalues)
  if (n_low > 0) {
    cudaMemcpy(d_q_persistent, d_eigenvectors_temp,
      (size_t)n * n_low * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  // Copy last n_high columns (positive eigenvalues)
  if (n_high > 0) {
    double* src = d_eigenvectors_temp + (size_t)(n - n_high) * n;
    double* dst = d_q_persistent + (size_t)n_low * n;
    cudaMemcpy(dst, src, (size_t)n * n_high * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  // Also copy to CPU, where the stages that need its entries read it
  cudaMemcpy(q_cpu, d_q_persistent, (size_t)n * k * sizeof(double), cudaMemcpyDeviceToHost);

  ensure_work_buffers(n > k ? n : k);
}

// Eigenvalues only, ascending; a is left unchanged on the host and no
// eigenvector state is touched, so this can be called between
// symmetric_eigen() and extract_eigenvectors()
int symmetric_eigenvalues(int n, const double* a, double* lambda) {
  ensure_gpu_init();

  double* d_a = NULL;
  double* d_lambda = NULL;
  int* d_info = NULL;
  double* d_work = NULL;
  int lwork = 0;
  int info = 0;

  cudaMalloc((void**)&d_a, sizeof(double) * n * n);
  cudaMalloc((void**)&d_lambda, sizeof(double) * n);
  cudaMalloc((void**)&d_info, sizeof(int));

  cudaMemcpy(d_a, a, sizeof(double) * n * n, cudaMemcpyHostToDevice);

  cusolverDnDsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR,
    CUBLAS_FILL_MODE_UPPER, n, d_a, n, d_lambda, &lwork);

  cudaMalloc((void**)&d_work, sizeof(double) * lwork);

  cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_NOVECTOR,
    CUBLAS_FILL_MODE_UPPER, n, d_a, n, d_lambda, d_work, lwork, d_info);

  cudaStreamSynchronize(gpu_stream);

  cudaMemcpy(lambda, d_lambda, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_work);
  cudaFree(d_info);
  cudaFree(d_lambda);
  cudaFree(d_a);

  return info;
}

// The generic products below have their own cuBLAS handle on the default
// stream and their own buffers.

static cublasHandle_t blas_handle = NULL;

// Persistent work buffers to avoid repeated malloc/free
static double* d_work1 = NULL;
static double* d_work2 = NULL;
static int mdv_work_size = 0;

static void ensure_blas_init() {
  if (blas_handle == NULL) {
    cublasCreate(&blas_handle);
  }
}

static void ensure_mdv_buffers(int n) {
  if (n > mdv_work_size) {
    if (d_work1) cudaFree(d_work1);
    if (d_work2) cudaFree(d_work2);
    cudaMalloc((void**)&d_work1, n * sizeof(double));
    cudaMalloc((void**)&d_work2, n * sizeof(double));
    mdv_work_size = n;
  }
}

double dot_product(int n, const double* x, const double* y) {
  ensure_blas_init();
  ensure_mdv_buffers(n);
  double result;

  cudaMemcpy(d_work1, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_work2, y, n * sizeof(double), cudaMemcpyHostToDevice);
  cublasDdot(blas_handle, n, d_work1, 1, d_work2, 1, &result);

  return result;
}

// GPU matrix-vector multiplication
// if trans=='N' then y:=A*x; if trans=='T' then y:=A'*x
void matrix_dot_vector(int m, int n, const double* a, char trans,
                       const double* x, double* y) {
  ensure_blas_init();
  double alpha = 1.0;
  double beta = 0.0;
  double *d_a, *d_x, *d_y;
  cublasOperation_t op = (trans == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  int x_len = (trans == 'N') ? n : m;
  int y_len = (trans == 'N') ? m : n;

  cudaMalloc((void**)&d_a, m * n * sizeof(double));
  cudaMalloc((void**)&d_x, x_len * sizeof(double));
  cudaMalloc((void**)&d_y, y_len * sizeof(double));

  cudaMemcpy(d_a, a, m * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, x_len * sizeof(double), cudaMemcpyHostToDevice);

  cublasDgemv(blas_handle, op, m, n, &alpha, d_a, m, d_x, 1, &beta, d_y, 1);

  cudaMemcpy(y, d_y, y_len * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}
