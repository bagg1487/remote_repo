#include "my_blas.hpp"
#include <cstring>
#include <complex>

namespace my_blas {

void ssyrk(const char* uplo, const char* trans, int n, int k,
           float alpha, const float* A, int lda,
           float beta, float* C, int ldc) {
    
    char uplo_char = uplo[0];
    char trans_char = trans[0];
    bool transA = (trans_char == 'T' || trans_char == 't');
    bool conjA = (trans_char == 'C' || trans_char == 'c');
    
    if (conjA) {
        transA = true;
    }
    
    if (!transA) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[i * lda + l] * A[j * lda + l];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = C[i * ldc + j];
                    } else {
                        C[i * ldc + j] = C[j * ldc + i];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[l * lda + i] * A[l * lda + j];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = C[i * ldc + j];
                    } else {
                        C[i * ldc + j] = C[j * ldc + i];
                    }
                }
            }
        }
    }
}

void dsyrk(const char* uplo, const char* trans, int n, int k,
           double alpha, const double* A, int lda,
           double beta, double* C, int ldc) {
    
    char uplo_char = uplo[0];
    char trans_char = trans[0];
    bool transA = (trans_char == 'T' || trans_char == 't' || trans_char == 'C' || trans_char == 'c');
    
    if (!transA) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i * lda + l] * A[j * lda + l];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = C[i * ldc + j];
                    } else {
                        C[i * ldc + j] = C[j * ldc + i];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[l * lda + i] * A[l * lda + j];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = C[i * ldc + j];
                    } else {
                        C[i * ldc + j] = C[j * ldc + i];
                    }
                }
            }
        }
    }
}

void csyrk(const char* uplo, const char* trans, int n, int k,
           std::complex<float> alpha, const std::complex<float>* A, int lda,
           std::complex<float> beta, std::complex<float>* C, int ldc) {
    
    char uplo_char = uplo[0];
    char trans_char = trans[0];
    bool transA = (trans_char == 'T' || trans_char == 't');
    bool conjA = (trans_char == 'C' || trans_char == 'c');
    
    if (!transA && !conjA) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                std::complex<float> sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[i * lda + l] * A[j * lda + l];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = std::conj(C[i * ldc + j]);
                    } else {
                        C[i * ldc + j] = std::conj(C[j * ldc + i]);
                    }
                }
            }
        }
    } else if (transA && !conjA) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                std::complex<float> sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += A[l * lda + i] * A[l * lda + j];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = std::conj(C[i * ldc + j]);
                    } else {
                        C[i * ldc + j] = std::conj(C[j * ldc + i]);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                std::complex<float> sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += std::conj(A[l * lda + i]) * A[l * lda + j];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = std::conj(C[i * ldc + j]);
                    } else {
                        C[i * ldc + j] = std::conj(C[j * ldc + i]);
                    }
                }
            }
        }
    }
}

void zsyrk(const char* uplo, const char* trans, int n, int k,
           std::complex<double> alpha, const std::complex<double>* A, int lda,
           std::complex<double> beta, std::complex<double>* C, int ldc) {
    
    char uplo_char = uplo[0];
    char trans_char = trans[0];
    bool transA = (trans_char == 'T' || trans_char == 't');
    bool conjA = (trans_char == 'C' || trans_char == 'c');
    
    if (!transA && !conjA) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                std::complex<double> sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i * lda + l] * A[j * lda + l];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = std::conj(C[i * ldc + j]);
                    } else {
                        C[i * ldc + j] = std::conj(C[j * ldc + i]);
                    }
                }
            }
        }
    } else if (transA && !conjA) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                std::complex<double> sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[l * lda + i] * A[l * lda + j];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = std::conj(C[i * ldc + j]);
                    } else {
                        C[i * ldc + j] = std::conj(C[j * ldc + i]);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if ((uplo_char == 'U' || uplo_char == 'u') && j < i) continue;
                if ((uplo_char == 'L' || uplo_char == 'l') && j > i) continue;
                
                std::complex<double> sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += std::conj(A[l * lda + i]) * A[l * lda + j];
                }
                
                if (i == j) {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                } else {
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    if (uplo_char == 'U' || uplo_char == 'u') {
                        C[j * ldc + i] = std::conj(C[i * ldc + j]);
                    } else {
                        C[i * ldc + j] = std::conj(C[j * ldc + i]);
                    }
                }
            }
        }
    }
}

}