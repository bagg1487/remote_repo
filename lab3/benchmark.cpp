#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <complex>
#include <omp.h>
#include <cblas.h>
#include "my_blas.hpp"

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

template<typename T>
void init_matrix(T* mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = (T)((double)rand() / RAND_MAX);
        }
    }
}

template<typename T>
void init_complex_matrix(std::complex<T>* mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = std::complex<T>((T)rand() / RAND_MAX, (T)rand() / RAND_MAX);
        }
    }
}

double geometric_mean(std::vector<double>& values) {
    double product = 1.0;
    for (double v : values) product *= v;
    return pow(product, 1.0 / values.size());
}

void benchmark_ssyrk(int n, int k, int num_threads, int num_runs) {
    omp_set_num_threads(num_threads);
    
    size_t size_a = k * n;
    size_t size_c = n * n;
    
    std::vector<float> A(size_a);
    std::vector<float> C_my(size_c);
    std::vector<float> C_blas(size_c);
    
    init_matrix(A.data(), k, n, n);
    
    std::vector<double> ratios(num_runs);
    std::vector<double> my_times(num_runs);
    std::vector<double> blas_times(num_runs);
    
    for (int run = 0; run < num_runs; run++) {
        init_matrix(C_my.data(), n, n, n);
        std::copy(C_my.begin(), C_my.end(), C_blas.begin());
        
        double start = get_time();
        my_blas::ssyrk("U", "N", n, k, 1.0f, A.data(), n, 0.0f, C_my.data(), n);
        my_times[run] = get_time() - start;
        
        start = get_time();
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, 1.0f, A.data(), n, 0.0f, C_blas.data(), n);
        blas_times[run] = get_time() - start;
        
        ratios[run] = (blas_times[run] / my_times[run]) * 100.0;
    }
    
    double geom_mean = geometric_mean(ratios);
    double my_avg = 0, blas_avg = 0;
    for (int i = 0; i < num_runs; i++) {
        my_avg += my_times[i];
        blas_avg += blas_times[i];
    }
    my_avg /= num_runs;
    blas_avg /= num_runs;
    
    std::cout << "ssyrk (float):\n";
    std::cout << "  My avg: " << my_avg << "s\n";
    std::cout << "  BLAS avg: " << blas_avg << "s\n";
    std::cout << "  Performance: " << geom_mean << "%\n";
}

void benchmark_dsyrk(int n, int k, int num_threads, int num_runs) {
    omp_set_num_threads(num_threads);
    
    size_t size_a = k * n;
    size_t size_c = n * n;
    
    std::vector<double> A(size_a);
    std::vector<double> C_my(size_c);
    std::vector<double> C_blas(size_c);
    
    init_matrix(A.data(), k, n, n);
    
    std::vector<double> ratios(num_runs);
    std::vector<double> my_times(num_runs);
    std::vector<double> blas_times(num_runs);
    
    for (int run = 0; run < num_runs; run++) {
        init_matrix(C_my.data(), n, n, n);
        std::copy(C_my.begin(), C_my.end(), C_blas.begin());
        
        double start = get_time();
        my_blas::dsyrk("U", "N", n, k, 1.0, A.data(), n, 0.0, C_my.data(), n);
        my_times[run] = get_time() - start;
        
        start = get_time();
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, 1.0, A.data(), n, 0.0, C_blas.data(), n);
        blas_times[run] = get_time() - start;
        
        ratios[run] = (blas_times[run] / my_times[run]) * 100.0;
    }
    
    double geom_mean = geometric_mean(ratios);
    double my_avg = 0, blas_avg = 0;
    for (int i = 0; i < num_runs; i++) {
        my_avg += my_times[i];
        blas_avg += blas_times[i];
    }
    my_avg /= num_runs;
    blas_avg /= num_runs;
    
    std::cout << "dsyrk (double):\n";
    std::cout << "  My avg: " << my_avg << "s\n";
    std::cout << "  BLAS avg: " << blas_avg << "s\n";
    std::cout << "  Performance: " << geom_mean << "%\n";
}

void benchmark_csyrk(int n, int k, int num_threads, int num_runs) {
    omp_set_num_threads(num_threads);
    
    size_t size_a = k * n;
    size_t size_c = n * n;
    
    std::vector<std::complex<float>> A(size_a);
    std::vector<std::complex<float>> C_my(size_c);
    std::vector<std::complex<float>> C_blas(size_c);
    
    init_complex_matrix(A.data(), k, n, n);
    
    std::vector<double> ratios(num_runs);
    std::vector<double> my_times(num_runs);
    std::vector<double> blas_times(num_runs);
    
    std::complex<float> alpha(1.0f, 0.0f);
    std::complex<float> beta(0.0f, 0.0f);
    
    for (int run = 0; run < num_runs; run++) {
        init_complex_matrix(C_my.data(), n, n, n);
        std::copy(C_my.begin(), C_my.end(), C_blas.begin());
        
        double start = get_time();
        my_blas::csyrk("U", "N", n, k, alpha, A.data(), n, beta, C_my.data(), n);
        my_times[run] = get_time() - start;
        
        start = get_time();
        cblas_csyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, 
                   &alpha, A.data(), n, &beta, C_blas.data(), n);
        blas_times[run] = get_time() - start;
        
        ratios[run] = (blas_times[run] / my_times[run]) * 100.0;
    }
    
    double geom_mean = geometric_mean(ratios);
    double my_avg = 0, blas_avg = 0;
    for (int i = 0; i < num_runs; i++) {
        my_avg += my_times[i];
        blas_avg += blas_times[i];
    }
    my_avg /= num_runs;
    blas_avg /= num_runs;
    
    std::cout << "csyrk (complex float):\n";
    std::cout << "  My avg: " << my_avg << "s\n";
    std::cout << "  BLAS avg: " << blas_avg << "s\n";
    std::cout << "  Performance: " << geom_mean << "%\n";
}

void benchmark_zsyrk(int n, int k, int num_threads, int num_runs) {
    omp_set_num_threads(num_threads);
    
    size_t size_a = k * n;
    size_t size_c = n * n;
    
    std::vector<std::complex<double>> A(size_a);
    std::vector<std::complex<double>> C_my(size_c);
    std::vector<std::complex<double>> C_blas(size_c);
    
    init_complex_matrix(A.data(), k, n, n);
    
    std::vector<double> ratios(num_runs);
    std::vector<double> my_times(num_runs);
    std::vector<double> blas_times(num_runs);
    
    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);
    
    for (int run = 0; run < num_runs; run++) {
        init_complex_matrix(C_my.data(), n, n, n);
        std::copy(C_my.begin(), C_my.end(), C_blas.begin());
        
        double start = get_time();
        my_blas::zsyrk("U", "N", n, k, alpha, A.data(), n, beta, C_my.data(), n);
        my_times[run] = get_time() - start;
        
        start = get_time();
        cblas_zsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n, k, 
                   &alpha, A.data(), n, &beta, C_blas.data(), n);
        blas_times[run] = get_time() - start;
        
        ratios[run] = (blas_times[run] / my_times[run]) * 100.0;
    }
    
    double geom_mean = geometric_mean(ratios);
    double my_avg = 0, blas_avg = 0;
    for (int i = 0; i < num_runs; i++) {
        my_avg += my_times[i];
        blas_avg += blas_times[i];
    }
    my_avg /= num_runs;
    blas_avg /= num_runs;
    
    std::cout << "zsyrk (complex double):\n";
    std::cout << "  My avg: " << my_avg << "s\n";
    std::cout << "  BLAS avg: " << blas_avg << "s\n";
    std::cout << "  Performance: " << geom_mean << "%\n";
}

int main() {
    std::vector<int> threads = {1, 2, 4, 8, 16};
    int n = 2000;
    int k = 2000;
    int num_runs = 10;
    
    srand(time(nullptr));
    
    std::cout << "SYRK Benchmark\n";
    std::cout << "n = " << n << ", k = " << k << "\n";
    std::cout << "Runs per configuration: " << num_runs << "\n\n";
    
    for (int t : threads) {
        std::cout << "\n=== Threads: " << t << " ===\n\n";
        
        benchmark_ssyrk(n, k, t, num_runs);
        benchmark_dsyrk(n, k, t, num_runs);
        benchmark_csyrk(n, k, t, num_runs);
        benchmark_zsyrk(n, k, t, num_runs);
    }
    
    return 0;
}