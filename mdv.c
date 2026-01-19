#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t cublas_handle = NULL;

static void ensure_cublas_init() {
  if (cublas_handle == NULL) {
    cublasCreate(&cublas_handle);
  }
}

double norm2(int n, double* x) {
  ensure_cublas_init();
  double result;
  double* d_x;

  cudaMalloc((void**)&d_x, n * sizeof(double));
  cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

  cublasDnrm2(cublas_handle, n, d_x, 1, &result);

  cudaFree(d_x);
  return result;
}

void scaling(int n, double* x, double c) {
  ensure_cublas_init();
  double* d_x;

  cudaMalloc((void**)&d_x, n * sizeof(double));
  cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

  cublasDscal(cublas_handle, n, &c, d_x, 1);

  cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
}

double dot_product(int n, double* x, double* y) {
  ensure_cublas_init();
  double result;
  double *d_x, *d_y;

  cudaMalloc((void**)&d_x, n * sizeof(double));
  cudaMalloc((void**)&d_y, n * sizeof(double));
  cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);

  cublasDdot(cublas_handle, n, d_x, 1, d_y, 1, &result);

  cudaFree(d_y);
  cudaFree(d_x);
  return result;
}

// GPU matrix-vector multiplication
// if trans=='N' then y:=A*x; if trans=='T' then y:=A'*x
void matrix_dot_vector(int m, int n, double* a, char trans, double* x, double* y) {
  ensure_cublas_init();
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

  cublasDgemv(cublas_handle, op, m, n, &alpha, d_a, m, d_x, 1, &beta, d_y, 1);

  cudaMemcpy(y, d_y, y_len * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_y);
  cudaFree(d_x);
  cudaFree(d_a);
}

// GPU matrix-matrix multiplication
// C = transa(A)*transb(B)
void matrix_dot_matrix(int m, int n, int k, double* a, char transa, double* b, char transb, double* c) {
  ensure_cublas_init();
  double alpha = 1.0;
  double beta = 0.0;
  double *d_a, *d_b, *d_c;
  cublasOperation_t opA = (transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opB = (transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  int lda = (transa == 'N') ? m : k;
  int ldb = (transb == 'N') ? k : n;

  cudaMalloc((void**)&d_a, m * k * sizeof(double));
  cudaMalloc((void**)&d_b, k * n * sizeof(double));
  cudaMalloc((void**)&d_c, m * n * sizeof(double));

  cudaMemcpy(d_a, a, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, k * n * sizeof(double), cudaMemcpyHostToDevice);

  cublasDgemm(cublas_handle, opA, opB, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, m);

  cudaMemcpy(c, d_c, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
}
