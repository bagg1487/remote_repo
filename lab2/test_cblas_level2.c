<<<<<<< HEAD
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <setjmp.h>
#include <complex.h>
#include <cblas.h>

static jmp_buf env;
static int ok = 1;

void handler(int sig){ longjmp(env,1); }

#define RUN_TEST(name, call)                           \
do {                                                    \
    printf("%-8s : ", name);                           \
    fflush(stdout);                                     \
                                                        \
    signal(SIGSEGV, handler);                           \
    signal(SIGFPE, handler);                            \
    signal(SIGILL, handler);                            \
    signal(SIGABRT, handler);                           \
    if(setjmp(env) == 0) {                              \
        call;                                           \
        printf("PASSED\n");                             \
    } else {                                            \
        printf("FAILED (signal)\n");                    \
        ok = 0;                                         \
    }                                                   \
    signal(SIGSEGV, SIG_DFL);                           \
    signal(SIGFPE, SIG_DFL);                            \
    signal(SIGILL, SIG_DFL);                            \
    signal(SIGABRT, SIG_DFL);                           \
} while(0)

void test_sgemv(){int m=2,n=2;float A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
RUN_TEST("sgemv",cblas_sgemv(CblasRowMajor,CblasNoTrans,m,n,1.0f,A,n,x,1,0.0f,y,1));}

void test_dgemv(){int m=2,n=2;double A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
RUN_TEST("dgemv",cblas_dgemv(CblasRowMajor,CblasNoTrans,m,n,1.0,A,n,x,1,0.0,y,1));}

void test_cgemv(){int m=2,n=2;float complex A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
float complex a=1,b=0;
RUN_TEST("cgemv",cblas_cgemv(CblasRowMajor,CblasNoTrans,m,n,&a,A,n,x,1,&b,y,1));}

void test_zgemv(){int m=2,n=2;double complex A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
double complex a=1,b=0;
RUN_TEST("zgemv",cblas_zgemv(CblasRowMajor,CblasNoTrans,m,n,&a,A,n,x,1,&b,y,1));}

void test_chemv(){int n=2;float complex A[4]={1,2+1*I,2-1*I,3},x[2]={1,1},y[2]={0,0};
float complex a=1,b=0;
RUN_TEST("chemv",cblas_chemv(CblasRowMajor,CblasUpper,n,&a,A,n,x,1,&b,y,1));}

void test_zhemv(){int n=2;double complex A[4]={1,2+1*I,2-1*I,3},x[2]={1,1},y[2]={0,0};
double complex a=1,b=0;
RUN_TEST("zhemv",cblas_zhemv(CblasRowMajor,CblasUpper,n,&a,A,n,x,1,&b,y,1));}

void test_ssymv(){int n=2;float A[4]={1,2,2,3},x[2]={1,1},y[2]={0,0};
RUN_TEST("ssymv",cblas_ssymv(CblasRowMajor,CblasUpper,n,1.0f,A,n,x,1,0.0f,y,1));}

void test_dsymv(){int n=2;double A[4]={1,2,2,3},x[2]={1,1},y[2]={0,0};
RUN_TEST("dsymv",cblas_dsymv(CblasRowMajor,CblasUpper,n,1.0,A,n,x,1,0.0,y,1));}

void test_strmv(){int n=2;float A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("strmv",cblas_strmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_dtrmv(){int n=2;double A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("dtrmv",cblas_dtrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ctrmv(){int n=2;float complex A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("ctrmv",cblas_ctrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ztrmv(){int n=2;double complex A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("ztrmv",cblas_ztrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_strsv(){int n=2;float A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("strsv",cblas_strsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_dtrsv(){int n=2;double A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("dtrsv",cblas_dtrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ctrsv(){int n=2;float complex A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("ctrsv",cblas_ctrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ztrsv(){int n=2;double complex A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("ztrsv",cblas_ztrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_sger(){int m=2,n=2;float A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("sger",cblas_sger(CblasRowMajor,m,n,1.0f,x,1,y,1,A,n));}

void test_dger(){int m=2,n=2;double A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("dger",cblas_dger(CblasRowMajor,m,n,1.0,x,1,y,1,A,n));}

void test_cgeru(){int m=2,n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
RUN_TEST("cgeru",cblas_cgeru(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_zgeru(){int m=2,n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
RUN_TEST("zgeru",cblas_zgeru(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_cgerc(){int m=2,n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
RUN_TEST("cgerc",cblas_cgerc(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_zgerc(){int m=2,n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
RUN_TEST("zgerc",cblas_zgerc(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_ssyr(){int n=2;float A[4]={0},x[2]={1,1};
RUN_TEST("ssyr",cblas_ssyr(CblasRowMajor,CblasUpper,n,1.0f,x,1,A,n));}

void test_dsyr(){int n=2;double A[4]={0},x[2]={1,1};
RUN_TEST("dsyr",cblas_dsyr(CblasRowMajor,CblasUpper,n,1.0,x,1,A,n));}

void test_cher(){int n=2;float complex A[4]={0},x[2]={1,1};
RUN_TEST("cher",cblas_cher(CblasRowMajor,CblasUpper,n,1.0f,x,1,A,n));}

void test_zher(){int n=2;double complex A[4]={0},x[2]={1,1};
RUN_TEST("zher",cblas_zher(CblasRowMajor,CblasUpper,n,1.0,x,1,A,n));}

void test_ssyr2(){int n=2;float A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("ssyr2",cblas_ssyr2(CblasRowMajor,CblasUpper,n,1.0f,x,1,y,1,A,n));}

void test_dsyr2(){int n=2;double A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("dsyr2",cblas_dsyr2(CblasRowMajor,CblasUpper,n,1.0,x,1,y,1,A,n));}

void test_cher2(){int n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
RUN_TEST("cher2",cblas_cher2(CblasRowMajor,CblasUpper,n,&a,x,1,y,1,A,n));}

void test_zher2(){int n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
RUN_TEST("zher2",cblas_zher2(CblasRowMajor,CblasUpper,n,&a,x,1,y,1,A,n));}

int main(){
signal(SIGSEGV,handler);
signal(SIGABRT,handler);
signal(SIGFPE,handler);
signal(SIGILL,handler);

printf("LEVEL 2 BLAS INTERFACE TESTS\n\n");

test_sgemv(); test_dgemv(); test_cgemv(); test_zgemv();
test_chemv(); test_zhemv();
test_ssymv(); test_dsymv();
test_strmv(); test_dtrmv(); test_ctrmv(); test_ztrmv();
test_strsv(); test_dtrsv(); test_ctrsv(); test_ztrsv();
test_sger(); test_dger();
test_cgeru(); test_zgeru();
test_cgerc(); test_zgerc();
test_ssyr(); test_dsyr();
test_cher(); test_zher();
test_ssyr2(); test_dsyr2();
test_cher2(); test_zher2();

printf("\nOVERALL: %s\n", ok ? "ALL PASSED" : "FAILED");
return ok ? 0 : 1;
}



// Функция	        Что делает	
// gemv     умножает матрицу на вектор	
// hemv	    то же для особых (эрмитовых) матриц	
// symv	    то же для симметричных матриц	
// trmv	    умножает треугольную матрицу	
// trsv	    решает уравнение с треугольной матрицей, найти x
// ger	    обновляет матрицу	
// syr	    обновляет симметричную матрицу	
// her	    обновляет эрмитову матрицу	
// syr2	    обновление с двумя векторами	
// her2	    то же для эрмитовых 
=======
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <setjmp.h>
#include <complex.h>
#include <cblas.h>

static jmp_buf env;
static int ok = 1;

void handler(int sig){ longjmp(env,1); }

#define RUN_TEST(name, call)                           \
do {                                                    \
    printf("%-8s : ", name);                           \
    fflush(stdout);                                     \
                                                        \
    signal(SIGSEGV, handler);                           \
    signal(SIGFPE, handler);                            \
    signal(SIGILL, handler);                            \
    signal(SIGABRT, handler);                           \
    if(setjmp(env) == 0) {                              \
        call;                                           \
        printf("PASSED\n");                             \
    } else {                                            \
        printf("FAILED (signal)\n");                    \
        ok = 0;                                         \
    }                                                   \
    signal(SIGSEGV, SIG_DFL);                           \
    signal(SIGFPE, SIG_DFL);                            \
    signal(SIGILL, SIG_DFL);                            \
    signal(SIGABRT, SIG_DFL);                           \
} while(0)

void test_sgemv(){int m=2,n=2;float A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
RUN_TEST("sgemv",cblas_sgemv(CblasRowMajor,CblasNoTrans,m,n,1.0f,A,n,x,1,0.0f,y,1));}

void test_dgemv(){int m=2,n=2;double A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
RUN_TEST("dgemv",cblas_dgemv(CblasRowMajor,CblasNoTrans,m,n,1.0,A,n,x,1,0.0,y,1));}

void test_cgemv(){int m=2,n=2;float complex A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
float complex a=1,b=0;
RUN_TEST("cgemv",cblas_cgemv(CblasRowMajor,CblasNoTrans,m,n,&a,A,n,x,1,&b,y,1));}

void test_zgemv(){int m=2,n=2;double complex A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
double complex a=1,b=0;
RUN_TEST("zgemv",cblas_zgemv(CblasRowMajor,CblasNoTrans,m,n,&a,A,n,x,1,&b,y,1));}

void test_chemv(){int n=2;float complex A[4]={1,2+1*I,2-1*I,3},x[2]={1,1},y[2]={0,0};
float complex a=1,b=0;
RUN_TEST("chemv",cblas_chemv(CblasRowMajor,CblasUpper,n,&a,A,n,x,1,&b,y,1));}

void test_zhemv(){int n=2;double complex A[4]={1,2+1*I,2-1*I,3},x[2]={1,1},y[2]={0,0};
double complex a=1,b=0;
RUN_TEST("zhemv",cblas_zhemv(CblasRowMajor,CblasUpper,n,&a,A,n,x,1,&b,y,1));}

void test_ssymv(){int n=2;float A[4]={1,2,2,3},x[2]={1,1},y[2]={0,0};
RUN_TEST("ssymv",cblas_ssymv(CblasRowMajor,CblasUpper,n,1.0f,A,n,x,1,0.0f,y,1));}

void test_dsymv(){int n=2;double A[4]={1,2,2,3},x[2]={1,1},y[2]={0,0};
RUN_TEST("dsymv",cblas_dsymv(CblasRowMajor,CblasUpper,n,1.0,A,n,x,1,0.0,y,1));}

void test_strmv(){int n=2;float A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("strmv",cblas_strmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_dtrmv(){int n=2;double A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("dtrmv",cblas_dtrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ctrmv(){int n=2;float complex A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("ctrmv",cblas_ctrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ztrmv(){int n=2;double complex A[4]={1,2,0,3},x[2]={1,1};
RUN_TEST("ztrmv",cblas_ztrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_strsv(){int n=2;float A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("strsv",cblas_strsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_dtrsv(){int n=2;double A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("dtrsv",cblas_dtrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ctrsv(){int n=2;float complex A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("ctrsv",cblas_ctrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_ztrsv(){int n=2;double complex A[4]={1,2,0,3},x[2]={3,3};
RUN_TEST("ztrsv",cblas_ztrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1));}

void test_sger(){int m=2,n=2;float A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("sger",cblas_sger(CblasRowMajor,m,n,1.0f,x,1,y,1,A,n));}

void test_dger(){int m=2,n=2;double A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("dger",cblas_dger(CblasRowMajor,m,n,1.0,x,1,y,1,A,n));}

void test_cgeru(){int m=2,n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
RUN_TEST("cgeru",cblas_cgeru(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_zgeru(){int m=2,n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
RUN_TEST("zgeru",cblas_zgeru(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_cgerc(){int m=2,n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
RUN_TEST("cgerc",cblas_cgerc(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_zgerc(){int m=2,n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
RUN_TEST("zgerc",cblas_zgerc(CblasRowMajor,m,n,&a,x,1,y,1,A,n));}

void test_ssyr(){int n=2;float A[4]={0},x[2]={1,1};
RUN_TEST("ssyr",cblas_ssyr(CblasRowMajor,CblasUpper,n,1.0f,x,1,A,n));}

void test_dsyr(){int n=2;double A[4]={0},x[2]={1,1};
RUN_TEST("dsyr",cblas_dsyr(CblasRowMajor,CblasUpper,n,1.0,x,1,A,n));}

void test_cher(){int n=2;float complex A[4]={0},x[2]={1,1};
RUN_TEST("cher",cblas_cher(CblasRowMajor,CblasUpper,n,1.0f,x,1,A,n));}

void test_zher(){int n=2;double complex A[4]={0},x[2]={1,1};
RUN_TEST("zher",cblas_zher(CblasRowMajor,CblasUpper,n,1.0,x,1,A,n));}

void test_ssyr2(){int n=2;float A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("ssyr2",cblas_ssyr2(CblasRowMajor,CblasUpper,n,1.0f,x,1,y,1,A,n));}

void test_dsyr2(){int n=2;double A[4]={0},x[2]={1,1},y[2]={1,1};
RUN_TEST("dsyr2",cblas_dsyr2(CblasRowMajor,CblasUpper,n,1.0,x,1,y,1,A,n));}

void test_cher2(){int n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
RUN_TEST("cher2",cblas_cher2(CblasRowMajor,CblasUpper,n,&a,x,1,y,1,A,n));}

void test_zher2(){int n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
RUN_TEST("zher2",cblas_zher2(CblasRowMajor,CblasUpper,n,&a,x,1,y,1,A,n));}

int main(){
signal(SIGSEGV,handler);
signal(SIGABRT,handler);
signal(SIGFPE,handler);
signal(SIGILL,handler);

printf("LEVEL 2 BLAS INTERFACE TESTS\n\n");

test_sgemv(); test_dgemv(); test_cgemv(); test_zgemv();
test_chemv(); test_zhemv();
test_ssymv(); test_dsymv();
test_strmv(); test_dtrmv(); test_ctrmv(); test_ztrmv();
test_strsv(); test_dtrsv(); test_ctrsv(); test_ztrsv();
test_sger(); test_dger();
test_cgeru(); test_zgeru();
test_cgerc(); test_zgerc();
test_ssyr(); test_dsyr();
test_cher(); test_zher();
test_ssyr2(); test_dsyr2();
test_cher2(); test_zher2();

printf("\nOVERALL: %s\n", ok ? "ALL PASSED" : "FAILED");
return ok ? 0 : 1;
}



>>>>>>> 44c2418c07e82e1181d55d94c91765139b875006
