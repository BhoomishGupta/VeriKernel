#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <omp.h>

#if defined(KERNEL_HEADER)
#define candidate_wave wave_kernel_ns
#define wave_baseline wave_kernel_ns
#define baseline wave_kernel_ns
#include KERNEL_HEADER
#undef candidate_wave
#undef wave_baseline
#undef baseline
namespace active_kernel = wave_kernel_ns;
#elif defined(WAVE_CANDIDATE_KERNEL)
#include "candidate_wave_kernel.hpp"
namespace active_kernel = candidate_wave;
#else
#include "wave_baseline.hpp"
namespace active_kernel = wave_baseline;
#endif

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880;

inline double exact_solution(double x, double y, double t) {
    return std::cos(SQRT2 * PI * t) * std::sin(PI * x) * std::sin(PI * y);
}

struct GridRunResult {
    int n;
    double h;
    double dt;
    int steps;
    double l2_error;
};

// Solves wave equation on grid N up to t_final
GridRunResult run_wave_grid(int n, double t_final) {
    double h = 1.0 / (n - 1);
    double dt = 0.2 * h; // CFL lambda = 0.2
    int steps = static_cast<int>(std::round(t_final / dt));
    dt = t_final / steps; // Exact step adjustment
    double lambda = dt / h;
    double lambda_sq = lambda * lambda;

    std::vector<double> u_prev(n * n, 0.0);
    std::vector<double> u_curr(n * n, 0.0);
    std::vector<double> u_next(n * n, 0.0);

    // Initial condition at t = 0
    for (int i = 0; i < n; ++i) {
        double x = i * h;
        for (int j = 0; j < n; ++j) {
            double y = j * h;
            u_prev[i * n + j] = exact_solution(x, y, 0.0);
        }
    }

    // Homogeneous Dirichlet boundary enforcement
    for (int k = 0; k < n; ++k) {
        u_prev[k * n + 0] = 0.0;
        u_prev[k * n + (n - 1)] = 0.0;
        u_prev[0 * n + k] = 0.0;
        u_prev[(n - 1) * n + k] = 0.0;
    }

    // Step 1: Second-order Taylor startup step
    active_kernel::startup_step(u_prev.data(), u_curr.data(), n, lambda_sq);

    // Boundary zeroing on u_curr
    for (int k = 0; k < n; ++k) {
        u_curr[k * n + 0] = 0.0;
        u_curr[k * n + (n - 1)] = 0.0;
        u_curr[0 * n + k] = 0.0;
        u_curr[(n - 1) * n + k] = 0.0;
    }

    // Time-stepping loop
    for (int s = 1; s < steps; ++s) {
        active_kernel::leapfrog_step(u_prev.data(), u_curr.data(), u_next.data(), n, lambda_sq);

        // Boundary conditions
        for (int k = 0; k < n; ++k) {
            u_next[k * n + 0] = 0.0;
            u_next[k * n + (n - 1)] = 0.0;
            u_next[0 * n + k] = 0.0;
            u_next[(n - 1) * n + k] = 0.0;
        }

        // Pointer rotation
        std::swap(u_prev, u_curr);
        std::swap(u_curr, u_next);
    }

    // Compute discrete L2 error at t_final
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = i * h;
        for (int j = 0; j < n; ++j) {
            double y = j * h;
            double diff = u_curr[i * n + j] - exact_solution(x, y, t_final);
            sum_sq += diff * diff;
        }
    }
    double l2 = std::sqrt(sum_sq / (n * n));

    return {n, h, dt, steps, l2};
}

int main(int argc, char** argv) {
    bool run_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") run_bench = true;
    }

    const std::vector<int> grids = {32, 64, 128, 256};
    const double t_final = 0.5;
    std::vector<GridRunResult> results;

    for (int n : grids) {
        results.push_back(run_wave_grid(n, t_final));
    }

    // Linear regression on ln(h) vs ln(E) to find order p: E = C * h^p
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    int k_count = results.size();
    for (const auto& r : results) {
        double x = std::log(r.h);
        double y = std::log(r.l2_error);
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    double slope_p = (k_count * sum_xy - sum_x * sum_y) / (k_count * sum_xx - sum_x * sum_x);

    // Multi-threaded determinism check at N=128 across 1, 2, 4, 8 threads
    double max_thread_drift = 0.0;
    {
        GridRunResult ref = run_wave_grid(128, 0.1);
        for (int threads : {1, 2, 4, 8}) {
            omp_set_num_threads(threads);
            GridRunResult check = run_wave_grid(128, 0.1);
            double diff = std::abs(check.l2_error - ref.l2_error);
            if (diff > max_thread_drift) max_thread_drift = diff;
        }
    }

    // Pinned Throughput Benchmark on 512x512 grid
    double throughput_gb_s = 0.0;
    double median_ms = 0.0;
    if (run_bench) {
        constexpr int BENCH_N = 512;
        constexpr int BENCH_STEPS = 200;
        double h = 1.0 / (BENCH_N - 1);
        double dt = 0.2 * h;
        double lambda_sq = (dt / h) * (dt / h);

        std::vector<double> u_p(BENCH_N * BENCH_N, 0.0);
        std::vector<double> u_c(BENCH_N * BENCH_N, 0.0);
        std::vector<double> u_n(BENCH_N * BENCH_N, 0.0);

        // Warmup
        for (int s = 0; s < 10; ++s) {
            active_kernel::leapfrog_step(u_p.data(), u_c.data(), u_n.data(), BENCH_N, lambda_sq);
        }

        std::vector<double> timings;
        for (int trial = 0; trial < 5; ++trial) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < BENCH_STEPS; ++s) {
                active_kernel::leapfrog_step(u_p.data(), u_c.data(), u_n.data(), BENCH_N, lambda_sq);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            timings.push_back(ms);
        }
        std::sort(timings.begin(), timings.end());
        median_ms = timings[2];

        // 3 arrays read/written * 8 bytes per cell * BENCH_STEPS sweeps
        double bytes_transferred = static_cast<double>(BENCH_STEPS) * (3.0 * BENCH_N * BENCH_N * sizeof(double));
        throughput_gb_s = (bytes_transferred / (median_ms * 1e-3)) / 1e9;
    }

    // Emit VeriKernel structured telemetry block
    std::cout << "\n__VERIKERNEL_JSON_START__\n";
    std::cout << "{\n";
    std::cout << "  \"workload\": \"2d_acoustic_wave\",\n";
    std::cout << "  \"observed_order_p\": " << std::fixed << std::setprecision(4) << slope_p << ",\n";
    std::cout << "  \"finest_grid_l2\": " << std::scientific << std::setprecision(4) << results.back().l2_error << ",\n";
    std::cout << "  \"max_thread_drift\": " << std::scientific << std::setprecision(4) << max_thread_drift << ",\n";
    std::cout << "  \"throughput_gb_s\": " << std::fixed << std::setprecision(2) << throughput_gb_s << ",\n";
    std::cout << "  \"median_ms\": " << std::fixed << std::setprecision(2) << median_ms << ",\n";
    std::cout << "  \"grids\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "    {\"n\": " << results[i].n << ", \"l2\": " << std::scientific << results[i].l2_error << "}"
                  << (i + 1 < results.size() ? "," : "") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
    std::cout << "__VERIKERNEL_JSON_END__\n";

    return 0;
}