#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t cublas_handle = NULL;

// Persistent work buffers to avoid repeated malloc/free
static double* d_work1 = NULL;
static double* d_work2 = NULL;
static int mdv_work_size = 0;

static void ensure_cublas_init() {
  if (cublas_handle == NULL) {
    cublasCreate(&cublas_handle);
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

double norm2(int n, double* x) {
  ensure_cublas_init();
  ensure_mdv_buffers(n);
  double result;

  cudaMemcpy(d_work1, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cublasDnrm2(cublas_handle, n, d_work1, 1, &result);

  return result;
}

void scaling(int n, double* x, double c) {
  ensure_cublas_init();
  ensure_mdv_buffers(n);

  cudaMemcpy(d_work1, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cublasDscal(cublas_handle, n, &c, d_work1, 1);
  cudaMemcpy(x, d_work1, n * sizeof(double), cudaMemcpyDeviceToHost);
}

double dot_product(int n, double* x, double* y) {
  ensure_cublas_init();
  ensure_mdv_buffers(n);
  double result;

  cudaMemcpy(d_work1, x, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_work2, y, n * sizeof(double), cudaMemcpyHostToDevice);
  cublasDdot(cublas_handle, n, d_work1, 1, d_work2, 1, &result);

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
