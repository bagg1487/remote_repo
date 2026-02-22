#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cblas.h>

#define EPS 1e-6
int ok = 1;

#define CHECK(name,cond) \
do { if (cond) printf("%-8s : PASSED\n", name); \
     else { printf("%-8s : FAILED\n", name); ok = 0; } } while(0)

void test_sgemv(){int m=2,n=2;float A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
cblas_sgemv(CblasRowMajor,CblasNoTrans,m,n,1.0f,A,n,x,1,0.0f,y,1);
CHECK("sgemv",fabsf(y[0]-3)<1e-4&&fabsf(y[1]-7)<1e-4);}

void test_dgemv(){int m=2,n=2;double A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
cblas_dgemv(CblasRowMajor,CblasNoTrans,m,n,1.0,A,n,x,1,0.0,y,1);
CHECK("dgemv",fabs(y[0]-3)<EPS&&fabs(y[1]-7)<EPS);}

void test_cgemv(){int m=2,n=2;float complex A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
float complex a=1,b=0;
cblas_cgemv(CblasRowMajor,CblasNoTrans,m,n,&a,A,n,x,1,&b,y,1);
CHECK("cgemv",cabsf(y[0]-3)<1e-4&&cabsf(y[1]-7)<1e-4);}

void test_zgemv(){int m=2,n=2;double complex A[4]={1,2,3,4},x[2]={1,1},y[2]={0,0};
double complex a=1,b=0;
cblas_zgemv(CblasRowMajor,CblasNoTrans,m,n,&a,A,n,x,1,&b,y,1);
CHECK("zgemv",cabs(y[0]-3)<EPS&&cabs(y[1]-7)<EPS);}

void test_chemv(){int n=2;float complex A[4]={1,2+1*I,2-1*I,3},x[2]={1,1},y[2]={0,0};
float complex a=1,b=0;
cblas_chemv(CblasRowMajor,CblasUpper,n,&a,A,n,x,1,&b,y,1);
CHECK("chemv",cabsf(y[0]-(3+1*I))<1e-4&&cabsf(y[1]-(5-1*I))<1e-4);}

void test_zhemv(){int n=2;double complex A[4]={1,2+1*I,2-1*I,3},x[2]={1,1},y[2]={0,0};
double complex a=1,b=0;
cblas_zhemv(CblasRowMajor,CblasUpper,n,&a,A,n,x,1,&b,y,1);
CHECK("zhemv",cabs(y[0]-(3+1*I))<EPS&&cabs(y[1]-(5-1*I))<EPS);}

void test_ssymv(){int n=2;float A[4]={1,2,2,3},x[2]={1,1},y[2]={0,0};
cblas_ssymv(CblasRowMajor,CblasUpper,n,1.0f,A,n,x,1,0.0f,y,1);
CHECK("ssymv",fabsf(y[0]-3)<1e-4&&fabsf(y[1]-5)<1e-4);}

void test_dsymv(){int n=2;double A[4]={1,2,2,3},x[2]={1,1},y[2]={0,0};
cblas_dsymv(CblasRowMajor,CblasUpper,n,1.0,A,n,x,1,0.0,y,1);
CHECK("dsymv",fabs(y[0]-3)<EPS&&fabs(y[1]-5)<EPS);}

void test_strmv(){int n=2;float A[4]={1,2,0,3},x[2]={1,1};
cblas_strmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("strmv",fabsf(x[0]-3)<1e-4&&fabsf(x[1]-3)<1e-4);}

void test_dtrmv(){int n=2;double A[4]={1,2,0,3},x[2]={1,1};
cblas_dtrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("dtrmv",fabs(x[0]-3)<EPS&&fabs(x[1]-3)<EPS);}

void test_ctrmv(){int n=2;float complex A[4]={1,2,0,3},x[2]={1,1};
cblas_ctrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("ctrmv",cabsf(x[0]-3)<1e-4&&cabsf(x[1]-3)<1e-4);}

void test_ztrmv(){int n=2;double complex A[4]={1,2,0,3},x[2]={1,1};
cblas_ztrmv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("ztrmv",cabs(x[0]-3)<EPS&&cabs(x[1]-3)<EPS);}

void test_strsv(){int n=2;float A[4]={1,2,0,3},x[2]={3,3};
cblas_strsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("strsv",fabsf(x[0]-1)<1e-4&&fabsf(x[1]-1)<1e-4);}

void test_dtrsv(){int n=2;double A[4]={1,2,0,3},x[2]={3,3};
cblas_dtrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("dtrsv",fabs(x[0]-1)<EPS&&fabs(x[1]-1)<EPS);}

void test_ctrsv(){int n=2;float complex A[4]={1,2,0,3},x[2]={3,3};
cblas_ctrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("ctrsv",cabsf(x[0]-1)<1e-4&&cabsf(x[1]-1)<1e-4);}

void test_ztrsv(){int n=2;double complex A[4]={1,2,0,3},x[2]={3,3};
cblas_ztrsv(CblasRowMajor,CblasUpper,CblasNoTrans,CblasNonUnit,n,A,n,x,1);
CHECK("ztrsv",cabs(x[0]-1)<EPS&&cabs(x[1]-1)<EPS);}

void test_sger(){int m=2,n=2;float A[4]={0,0,0,0},x[2]={1,1},y[2]={1,1};
cblas_sger(CblasRowMajor,m,n,1.0f,x,1,y,1,A,n);
CHECK("sger",fabsf(A[0]-1)<1e-4&&fabsf(A[3]-1)<1e-4);}

void test_dger(){int m=2,n=2;double A[4]={0,0,0,0},x[2]={1,1},y[2]={1,1};
cblas_dger(CblasRowMajor,m,n,1.0,x,1,y,1,A,n);
CHECK("dger",fabs(A[0]-1)<EPS&&fabs(A[3]-1)<EPS);}

void test_cgeru(){int m=2,n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
cblas_cgeru(CblasRowMajor,m,n,&a,x,1,y,1,A,n);
CHECK("cgeru",cabsf(A[0]-1)<1e-4);}

void test_zgeru(){int m=2,n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
cblas_zgeru(CblasRowMajor,m,n,&a,x,1,y,1,A,n);
CHECK("zgeru",cabs(A[0]-1)<EPS);}

void test_cgerc(){int m=2,n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
cblas_cgerc(CblasRowMajor,m,n,&a,x,1,y,1,A,n);
CHECK("cgerc",cabsf(A[0]-1)<1e-4);}

void test_zgerc(){int m=2,n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
cblas_zgerc(CblasRowMajor,m,n,&a,x,1,y,1,A,n);
CHECK("zgerc",cabs(A[0]-1)<EPS);}

void test_ssyr(){int n=2;float A[4]={0},x[2]={1,1};
cblas_ssyr(CblasRowMajor,CblasUpper,n,1.0f,x,1,A,n);
CHECK("ssyr",fabsf(A[0]-1)<1e-4);}

void test_dsyr(){int n=2;double A[4]={0},x[2]={1,1};
cblas_dsyr(CblasRowMajor,CblasUpper,n,1.0,x,1,A,n);
CHECK("dsyr",fabs(A[0]-1)<EPS);}

void test_cher(){int n=2;float complex A[4]={0},x[2]={1,1};
cblas_cher(CblasRowMajor,CblasUpper,n,1.0f,x,1,A,n);
CHECK("cher",cabsf(A[0]-1)<1e-4);}

void test_zher(){int n=2;double complex A[4]={0},x[2]={1,1};
cblas_zher(CblasRowMajor,CblasUpper,n,1.0,x,1,A,n);
CHECK("zher",cabs(A[0]-1)<EPS);}

void test_ssyr2(){int n=2;float A[4]={0},x[2]={1,1},y[2]={1,1};
cblas_ssyr2(CblasRowMajor,CblasUpper,n,1.0f,x,1,y,1,A,n);
CHECK("ssyr2",fabsf(A[0]-2)<1e-4);}

void test_dsyr2(){int n=2;double A[4]={0},x[2]={1,1},y[2]={1,1};
cblas_dsyr2(CblasRowMajor,CblasUpper,n,1.0,x,1,y,1,A,n);
CHECK("dsyr2",fabs(A[0]-2)<EPS);}

void test_cher2(){int n=2;float complex A[4]={0},x[2]={1,1},y[2]={1,1};
float complex a=1;
cblas_cher2(CblasRowMajor,CblasUpper,n,&a,x,1,y,1,A,n);
CHECK("cher2",cabsf(A[0]-2)<1e-4);}

void test_zher2(){int n=2;double complex A[4]={0},x[2]={1,1},y[2]={1,1};
double complex a=1;
cblas_zher2(CblasRowMajor,CblasUpper,n,&a,x,1,y,1,A,n);
CHECK("zher2",cabs(A[0]-2)<EPS);}

int main(){
    printf("LEVEL 2 BLAS TABLE ORDER TESTS\n\n");
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
    printf("\nOVERALL: %s\n", ok ? "ALL PASSED" : "FAIL");
    return ok ? 0 : 1;
}