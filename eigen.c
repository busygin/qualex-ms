#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// A C wrapper using cuSOLVER for eigendecomposition of a real symmetric matrix.
// Uses cusolverDnDsyevd which computes all eigenvalues and eigenvectors
// on the GPU.
int symmetric_eigen(int n, double* a, double* lambda, double* q) {
  cusolverDnHandle_t handle;
  cudaStream_t stream;
  cusolverStatus_t status;
  cudaError_t cudaStat;

  double* d_a = NULL;
  double* d_lambda = NULL;
  int* d_info = NULL;
  double* d_work = NULL;
  int lwork = 0;
  int info = 0;

  // Create cuSOLVER handle and stream
  status = cusolverDnCreate(&handle);
  if (status != CUSOLVER_STATUS_SUCCESS) return -1;

  cudaStat = cudaStreamCreate(&stream);
  if (cudaStat != cudaSuccess) {
    cusolverDnDestroy(handle);
    return -1;
  }
  cusolverDnSetStream(handle, stream);

  // Allocate device memory
  cudaStat = cudaMalloc((void**)&d_a, sizeof(double) * n * n);
  if (cudaStat != cudaSuccess) goto cleanup;

  cudaStat = cudaMalloc((void**)&d_lambda, sizeof(double) * n);
  if (cudaStat != cudaSuccess) goto cleanup;

  cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
  if (cudaStat != cudaSuccess) goto cleanup;

  // Copy matrix to device
  cudaStat = cudaMemcpy(d_a, a, sizeof(double) * n * n, cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess) goto cleanup;

  // Query workspace size
  status = cusolverDnDsyevd_bufferSize(
    handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    n, d_a, n, d_lambda, &lwork
  );
  if (status != CUSOLVER_STATUS_SUCCESS) goto cleanup;

  cudaStat = cudaMalloc((void**)&d_work, sizeof(double) * lwork);
  if (cudaStat != cudaSuccess) goto cleanup;

  // Compute eigenvalues and eigenvectors
  status = cusolverDnDsyevd(
    handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
    n, d_a, n, d_lambda, d_work, lwork, d_info
  );

  cudaStat = cudaStreamSynchronize(stream);
  if (cudaStat != cudaSuccess) goto cleanup;

  // Copy results back to host
  cudaStat = cudaMemcpy(lambda, d_lambda, sizeof(double) * n, cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) goto cleanup;

  cudaStat = cudaMemcpy(q, d_a, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess) goto cleanup;

  cudaStat = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

cleanup:
  if (d_work) cudaFree(d_work);
  if (d_info) cudaFree(d_info);
  if (d_lambda) cudaFree(d_lambda);
  if (d_a) cudaFree(d_a);
  cudaStreamDestroy(stream);
  cusolverDnDestroy(handle);

  return info;
}
