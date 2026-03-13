#ifndef MY_BLAS_HPP
#define MY_BLAS_HPP

#include <complex>

namespace my_blas {

void ssyrk(const char* uplo, const char* trans, int n, int k,
           float alpha, const float* A, int lda,
           float beta, float* C, int ldc);

void dsyrk(const char* uplo, const char* trans, int n, int k,
           double alpha, const double* A, int lda,
           double beta, double* C, int ldc);

void csyrk(const char* uplo, const char* trans, int n, int k,
           std::complex<float> alpha, const std::complex<float>* A, int lda,
           std::complex<float> beta, std::complex<float>* C, int ldc);

void zsyrk(const char* uplo, const char* trans, int n, int k,
           std::complex<double> alpha, const std::complex<double>* A, int lda,
           std::complex<double> beta, std::complex<double>* C, int ldc);

}

#endif