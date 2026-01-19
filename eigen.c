#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

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

// Store eigenvector matrix on GPU for repeated use
void store_eigenvectors_gpu(int n, int k, double* q) {
  ensure_gpu_init();
  if (d_q_persistent) cudaFree(d_q_persistent);
  q_rows = n;
  q_cols = k;
  cudaMalloc((void**)&d_q_persistent, (size_t)n * k * sizeof(double));
  cudaMemcpy(d_q_persistent, q, (size_t)n * k * sizeof(double), cudaMemcpyHostToDevice);
  ensure_work_buffers(n > k ? n : k);
}

// Matrix-vector multiply using GPU-resident Q: y = op(Q) * x
void matrix_dot_vector_q(int n, int k, char trans, double* x, double* y) {
  double alpha = 1.0, beta = 0.0;
  cublasOperation_t op = (trans == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  int x_len = (trans == 'N') ? k : n;
  int y_len = (trans == 'N') ? n : k;

  cudaMemcpy(d_work_vec1, x, x_len * sizeof(double), cudaMemcpyHostToDevice);
  cublasDgemv(cublas_handle, op, n, k, &alpha, d_q_persistent, n, d_work_vec1, 1, &beta, d_work_vec2, 1);
  cudaMemcpy(y, d_work_vec2, y_len * sizeof(double), cudaMemcpyDeviceToHost);
}

// Compute c = Q^T * hatb and store c on GPU, also copy to CPU
void compute_c_gpu(int n, int k, double* hatb, double* c) {
  ensure_gpu_init();
  double alpha = 1.0, beta = 0.0;

  // Allocate/reallocate persistent c buffer if needed
  if (k > c_size) {
    if (d_c_persistent) cudaFree(d_c_persistent);
    cudaMalloc((void**)&d_c_persistent, k * sizeof(double));
    c_size = k;
  }

  // Copy hatb to GPU work buffer
  cudaMemcpy(d_work_vec1, hatb, n * sizeof(double), cudaMemcpyHostToDevice);

  // Compute c = Q^T * hatb, store result in d_c_persistent
  cublasDgemv(cublas_handle, CUBLAS_OP_T, n, k, &alpha, d_q_persistent, n, d_work_vec1, 1, &beta, d_c_persistent, 1);

  // Copy c to CPU for use in solver
  cudaMemcpy(c, d_c_persistent, k * sizeof(double), cudaMemcpyDeviceToHost);
}

// Compute norm2 of GPU-resident c vector (no CPU-GPU transfer needed)
double norm2_c_gpu(int k) {
  double result;
  cublasDnrm2(cublas_handle, k, d_c_persistent, 1, &result);
  return result;
}

// Eigendecomposition using cuSOLVER
int symmetric_eigen(int n, double* a, double* lambda, double* q) {
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

  cusolverDnDsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER, n, d_a, n, d_lambda, &lwork);

  cudaMalloc((void**)&d_work, sizeof(double) * lwork);

  cusolverDnDsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
    CUBLAS_FILL_MODE_UPPER, n, d_a, n, d_lambda, d_work, lwork, d_info);

  cudaStreamSynchronize(gpu_stream);

  cudaMemcpy(lambda, d_lambda, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(q, d_a, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_work);
  cudaFree(d_info);
  cudaFree(d_lambda);
  cudaFree(d_a);

  return info;
}
