#include <cblas.h>
#include <complex.h>

static inline void zero_float_vec(int n, float *y, int incy) { for (int i = 0; i < n; i++) y[i * incy] = 0.0f; }
static inline void zero_double_vec(int n, double *y, int incy) { for (int i = 0; i < n; i++) y[i * incy] = 0.0; }
static inline void zero_cfloat_vec(int n, float complex *y, int incy) { for (int i = 0; i < n; i++) y[i * incy] = 0.0f + 0.0f*I; }
static inline void zero_cdouble_vec(int n, double complex *y, int incy) { for (int i = 0; i < n; i++) y[i * incy] = 0.0 + 0.0*I; }

static inline void zero_float_mat(int m, int n, float *A, int lda) { for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i*lda + j] = 0.0f; }
static inline void zero_double_mat(int m, int n, double *A, int lda) { for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i*lda + j] = 0.0; }
static inline void zero_cfloat_mat(int m, int n, float complex *A, int lda) { for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i*lda + j] = 0.0f + 0.0f*I; }
static inline void zero_cdouble_mat(int m, int n, double complex *A, int lda) { for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i*lda + j] = 0.0 + 0.0*I; }

void cblas_sgemv(const CBLAS_LAYOUT L,const CBLAS_TRANSPOSE T,const int M,const int N,const float a,const float *A,const int lda,const float *X,const int incX,const float b,float *Y,const int incY){ zero_float_vec(M,Y,incY); }
void cblas_dgemv(const CBLAS_LAYOUT L,const CBLAS_TRANSPOSE T,const int M,const int N,const double a,const double *A,const int lda,const double *X,const int incX,const double b,double *Y,const int incY){ zero_double_vec(M,Y,incY); }
void cblas_cgemv(const CBLAS_LAYOUT L,const CBLAS_TRANSPOSE T,const int M,const int N,const void *a,const void *A,const int lda,const void *X,const int incX,const void *b,void *Y,const int incY){ zero_cfloat_vec(M,(float complex*)Y,incY); }
void cblas_zgemv(const CBLAS_LAYOUT L,const CBLAS_TRANSPOSE T,const int M,const int N,const void *a,const void *A,const int lda,const void *X,const int incX,const void *b,void *Y,const int incY){ zero_cdouble_vec(M,(double complex*)Y,incY); }

void cblas_chemv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const void *a,const void *A,const int lda,const void *X,const int incX,const void *b,void *Y,const int incY){ zero_cfloat_vec(N,(float complex*)Y,incY); }
void cblas_zhemv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const void *a,const void *A,const int lda,const void *X,const int incX,const void *b,void *Y,const int incY){ zero_cdouble_vec(N,(double complex*)Y,incY); }

void cblas_ssymv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const float a,const float *A,const int lda,const float *X,const int incX,const float b,float *Y,const int incY){ zero_float_vec(N,Y,incY); }
void cblas_dsymv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const double a,const double *A,const int lda,const double *X,const int incX,const double b,double *Y,const int incY){ zero_double_vec(N,Y,incY); }

void cblas_strmv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const float *A,const int lda,float *X,const int incX){ zero_float_vec(N,X,incX); }
void cblas_dtrmv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const double *A,const int lda,double *X,const int incX){ zero_double_vec(N,X,incX); }
void cblas_ctrmv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const void *A,const int lda,void *X,const int incX){ zero_cfloat_vec(N,(float complex*)X,incX); }
void cblas_ztrmv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const void *A,const int lda,void *X,const int incX){ zero_cdouble_vec(N,(double complex*)X,incX); }

void cblas_strsv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const float *A,const int lda,float *X,const int incX){ zero_float_vec(N,X,incX); }
void cblas_dtrsv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const double *A,const int lda,double *X,const int incX){ zero_double_vec(N,X,incX); }
void cblas_ctrsv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const void *A,const int lda,void *X,const int incX){ zero_cfloat_vec(N,(float complex*)X,incX); }
void cblas_ztrsv(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const CBLAS_TRANSPOSE T,const CBLAS_DIAG D,const int N,const void *A,const int lda,void *X,const int incX){ zero_cdouble_vec(N,(double complex*)X,incX); }

void cblas_sger(const CBLAS_LAYOUT L,const int M,const int N,const float a,const float *X,const int incX,const float *Y,const int incY,float *A,const int lda){ zero_float_mat(M,N,A,lda); }
void cblas_dger(const CBLAS_LAYOUT L,const int M,const int N,const double a,const double *X,const int incX,const double *Y,const int incY,double *A,const int lda){ zero_double_mat(M,N,A,lda); }

void cblas_cgeru(const CBLAS_LAYOUT L,const int M,const int N,const void *a,const void *X,const int incX,const void *Y,const int incY,void *A,const int lda){ zero_cfloat_mat(M,N,(float complex*)A,lda); }
void cblas_zgeru(const CBLAS_LAYOUT L,const int M,const int N,const void *a,const void *X,const int incX,const void *Y,const int incY,void *A,const int lda){ zero_cdouble_mat(M,N,(double complex*)A,lda); }

void cblas_cgerc(const CBLAS_LAYOUT L,const int M,const int N,const void *a,const void *X,const int incX,const void *Y,const int incY,void *A,const int lda){ zero_cfloat_mat(M,N,(float complex*)A,lda); }
void cblas_zgerc(const CBLAS_LAYOUT L,const int M,const int N,const void *a,const void *X,const int incX,const void *Y,const int incY,void *A,const int lda){ zero_cdouble_mat(M,N,(double complex*)A,lda); }

void cblas_ssyr(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const float a,const float *X,const int incX,float *A,const int lda){ zero_float_mat(N,N,A,lda); }
void cblas_dsyr(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const double a,const double *X,const int incX,double *A,const int lda){ zero_double_mat(N,N,A,lda); }

void cblas_cher(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const float a,const void *X,const int incX,void *A,const int lda){ zero_cfloat_mat(N,N,(float complex*)A,lda); }
void cblas_zher(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const double a,const void *X,const int incX,void *A,const int lda){ zero_cdouble_mat(N,N,(double complex*)A,lda); }

void cblas_ssyr2(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const float a,const float *X,const int incX,const float *Y,const int incY,float *A,const int lda){ zero_float_mat(N,N,A,lda); }
void cblas_dsyr2(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const double a,const double *X,const int incX,const double *Y,const int incY,double *A,const int lda){ zero_double_mat(N,N,A,lda); }

void cblas_cher2(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const void *a,const void *X,const int incX,const void *Y,const int incY,void *A,const int lda){ zero_cfloat_mat(N,N,(float complex*)A,lda); }
void cblas_zher2(const CBLAS_LAYOUT L,const CBLAS_UPLO U,const int N,const void *a,const void *X,const int incX,const void *Y,const int incY,void *A,const int lda){ zero_cdouble_mat(N,N,(double complex*)A,lda); }