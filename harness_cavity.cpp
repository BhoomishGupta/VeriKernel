#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

#ifndef KERNEL_HEADER
#define KERNEL_HEADER "baseline_kernel.hpp"
#endif
#include KERNEL_HEADER

namespace {

constexpr std::size_t kN = 129;
constexpr double kRe = 100.0;
constexpr double kLidVelocity = 1.0;
constexpr double kDt = 1e-3;
constexpr double kConvergenceTol = 1e-6;
constexpr std::size_t kMaxSteps = 15000;
constexpr std::size_t kPoissonSweepsPerStep = 20;

// Ghia, Ghia, & Shin (1982) 17-point reference data for Re = 100
constexpr std::size_t kGhiaN = 17;
constexpr std::array<double, kGhiaN> y_ghia = {
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000
};
constexpr std::array<double, kGhiaN> u_ghia = {
     0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
    -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,  0.78871,  0.84123, 1.00000
};

constexpr std::array<double, kGhiaN> x_ghia = {
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000
};
constexpr std::array<double, kGhiaN> v_ghia = {
     0.00000,  0.09233,  0.10091,  0.10890,  0.12317,  0.16077,  0.17507,  0.17527,
     0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000
};

inline std::size_t id(const std::size_t i, const std::size_t j, const std::size_t n = kN) {
    return i * n + j;
}

// Thom's first-order wall boundary formulation (i: x-direction, j: y-direction)
void apply_vorticity_boundaries(std::vector<double>& omega, const std::vector<double>& psi, const double h, const std::size_t n = kN) {
    const double h2 = h * h;
    // Horizontal walls (Bottom j=0, Top j=n-1)
    for (std::size_t i = 0; i < n; ++i) {
        omega[id(i, 0, n)]     = -2.0 * psi[id(i, 1, n)] / h2;
        omega[id(i, n - 1, n)] = -2.0 * psi[id(i, n - 2, n)] / h2 - 2.0 * kLidVelocity / h;
    }
    // Vertical walls (Left i=0, Right i=n-1)
    for (std::size_t j = 0; j < n; ++j) {
        omega[id(0, j, n)]     = -2.0 * psi[id(1, j, n)] / h2;
        omega[id(n - 1, j, n)] = -2.0 * psi[id(n - 2, j, n)] / h2;
    }
}

// Centered velocity recovery: u = dpsi/dy (along j), v = -dpsi/dx (along i)
void compute_velocity(const std::vector<double>& psi, std::vector<double>& u, std::vector<double>& v, const double h, const std::size_t n = kN) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 1; i < static_cast<std::ptrdiff_t>(n - 1); ++i) {
        for (std::size_t j = 1; j + 1 < n; ++j) {
            const std::size_t ii = static_cast<std::size_t>(i);
            u[id(ii, j, n)] = (psi[id(ii, j + 1, n)] - psi[id(ii, j - 1, n)]) / (2.0 * h);
            v[id(ii, j, n)] = -(psi[id(ii + 1, j, n)] - psi[id(ii - 1, j, n)]) / (2.0 * h);
        }
    }

    // Horizontal walls (j=0, j=n-1)
    for (std::size_t i = 0; i < n; ++i) {
        u[id(i, 0, n)] = 0.0;
        u[id(i, n - 1, n)] = kLidVelocity;
        v[id(i, 0, n)] = 0.0;
        v[id(i, n - 1, n)] = 0.0;
    }
    // Vertical walls (i=0, i=n-1)
    for (std::size_t j = 0; j < n; ++j) {
        u[id(0, j, n)] = 0.0;
        u[id(n - 1, j, n)] = 0.0;
        v[id(0, j, n)] = 0.0;
        v[id(n - 1, j, n)] = 0.0;
    }
}

// Sub-grid 1D linear interpolation along domain centerlines
double interpolate_u(const std::vector<double>& u, const double y, const std::size_t n, const double h) {
    const double pos = y / h;
    const std::size_t j0 = std::min(static_cast<std::size_t>(std::floor(pos)), n - 2);
    const double t = pos - static_cast<double>(j0);
    const std::size_t mid_i = (n - 1) / 2; // Fixed x = 0.5
    return (1.0 - t) * u[id(mid_i, j0, n)] + t * u[id(mid_i, j0 + 1, n)];
}

double interpolate_v(const std::vector<double>& v, const double x, const std::size_t n, const double h) {
    const double pos = x / h;
    const std::size_t i0 = std::min(static_cast<std::size_t>(std::floor(pos)), n - 2);
    const double t = pos - static_cast<double>(i0);
    const std::size_t mid_j = (n - 1) / 2; // Fixed y = 0.5
    return (1.0 - t) * v[id(i0, mid_j, n)] + t * v[id(i0 + 1, mid_j, n)];
}

double max_delta(const std::vector<double>& a, const std::vector<double>& b) {
    double delta = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        delta = std::max(delta, std::abs(a[i] - b[i]));
    }
    return delta;
}

// Binary reference I/O for Stage 4 Oracle
void save_reference_binary(const std::string& filepath, const std::vector<double>& psi) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return;
    const std::size_t size = psi.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(psi.data()), size * sizeof(double));
}

bool verify_relative_l2(const std::string& filepath, const std::vector<double>& psi, double& l2_err) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    std::size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (size != psi.size()) return false;

    std::vector<double> ref(size);
    in.read(reinterpret_cast<char*>(ref.data()), size * sizeof(double));

    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double diff = psi[i] - ref[i];
        num += diff * diff;
        den += ref[i] * ref[i];
    }
    l2_err = std::sqrt(num / std::max(den, 1e-12));
    return l2_err <= 1e-5;
}

struct PhaseAResult {
    std::vector<double> psi;
    std::vector<double> omega;
    double runtime_seconds;
    std::size_t converged_steps;
    std::size_t total_poisson_sweeps;
    double ghia_u_rmse;
    double ghia_v_rmse;
};

// Phase A: Steady-state physics convergence solve
PhaseAResult solve_cavity_converge(const int threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const double h = 1.0 / static_cast<double>(kN - 1);

    std::vector<double> psi(kN * kN, 0.0);
    std::vector<double> omega(kN * kN, 0.0);
    std::vector<double> omega_next(kN * kN, 0.0);
    std::vector<double> u(kN * kN, 0.0);
    std::vector<double> v(kN * kN, 0.0);

    // Initial boundary application before time-stepping starts
    apply_vorticity_boundaries(omega, psi, h, kN);

    const double start = omp_get_wtime();
    std::size_t step = 0;
    std::size_t total_poisson_sweeps = 0;
    double steady_residual = 1.0;

    // Minimum burn-in of 20 steps prevents early false convergence
    for (; step < kMaxSteps && (step < 20 || steady_residual > kConvergenceTol); ++step) {
        baseline::gauss_seidel_poisson_sweep(psi.data(), omega.data(), kN, h, kPoissonSweepsPerStep);
        total_poisson_sweeps += kPoissonSweepsPerStep;

        baseline::vorticity_transport_step(
            omega_next.data(), omega.data(), psi.data(), kN, h, kDt, kRe);

        // Compute steady-state vorticity delta strictly on interior cells
        double max_dw = 0.0;
#pragma omp parallel for collapse(2) reduction(max:max_dw) schedule(static)
        for (std::size_t i = 1; i < kN - 1; ++i) {
            for (std::size_t j = 1; j < kN - 1; ++j) {
                const double diff = std::abs(omega_next[id(i, j, kN)] - omega[id(i, j, kN)]);
                max_dw = std::max(max_dw, diff);
            }
        }
        steady_residual = max_dw;

        omega.swap(omega_next);
        apply_vorticity_boundaries(omega, psi, h, kN);
    }

    const double runtime_seconds = omp_get_wtime() - start;
    compute_velocity(psi, u, v, h, kN);

    // Compute Ghia et al. (1982) centerline RMSE
    double u_sq_err = 0.0;
    for (std::size_t p = 0; p < kGhiaN; ++p) {
        const double diff = interpolate_u(u, y_ghia[p], kN, h) - u_ghia[p];
        u_sq_err += diff * diff;
    }
    const double ghia_u_rmse = std::sqrt(u_sq_err / static_cast<double>(kGhiaN));

    double v_sq_err = 0.0;
    for (std::size_t p = 0; p < kGhiaN; ++p) {
        const double diff = interpolate_v(v, x_ghia[p], kN, h) - v_ghia[p];
        v_sq_err += diff * diff;
    }
    const double ghia_v_rmse = std::sqrt(v_sq_err / static_cast<double>(kGhiaN));

    return {std::move(psi), std::move(omega), runtime_seconds, step, total_poisson_sweeps, ghia_u_rmse, ghia_v_rmse};
}

// Fast fixed-step solve for Stage 3 Determinism
std::vector<double> solve_cavity_fixed_steps(const int threads, const std::size_t n = 65, const std::size_t steps = 30) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const double h = 1.0 / static_cast<double>(n - 1);
    std::vector<double> psi(n * n, 0.0);
    std::vector<double> omega(n * n, 0.0);
    std::vector<double> omega_next(n * n, 0.0);

    apply_vorticity_boundaries(omega, psi, h, n);

    for (std::size_t step = 0; step < steps; ++step) {
        baseline::gauss_seidel_poisson_sweep(psi.data(), omega.data(), n, h, 5);
        baseline::vorticity_transport_step(omega_next.data(), omega.data(), psi.data(), n, h, kDt, kRe);
        omega.swap(omega_next);
        apply_vorticity_boundaries(omega, psi, h, n);
    }
    return psi;
}

// Stage 3 Determinism Gate
struct DeterminismReport {
    bool passed;
    double max_thread_delta;
    double max_repeat_delta;
};

DeterminismReport run_stage3_determinism() {
    constexpr std::array<int, 4> threads = {1, 2, 4, 8};
    std::vector<double> t1_reference;
    double max_thread_diff = 0.0;
    double max_repeat_diff = 0.0;

    for (const int t : threads) {
        std::vector<double> last_run;
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto current_run = solve_cavity_fixed_steps(t, 65, 30);
            if (repeat > 0) {
                max_repeat_diff = std::max(max_repeat_diff, max_delta(last_run, current_run));
            }
            last_run = std::move(current_run);
        }

        if (t == 1) {
            t1_reference = last_run;
        } else {
            max_thread_diff = std::max(max_thread_diff, max_delta(t1_reference, last_run));
        }
    }

    const bool ok = (max_repeat_diff <= 1e-11) && (max_thread_diff <= 1e-11);
    return {ok, max_thread_diff, max_repeat_diff};
}

// Phase B: Out-of-cache (513x513) memory bandwidth benchmark
struct BandwidthMetrics {
    double median_seconds;
    double bandwidth_gb_s;
    std::size_t sweeps;
};

BandwidthMetrics run_phase_b_bandwidth(const int threads, const std::size_t n = 513, const std::size_t sweeps = 100) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const double h = 1.0 / static_cast<double>(n - 1);
    std::vector<double> psi(n * n, 0.0);
    std::vector<double> omega(n * n, 1.0);

    // Warmup
    baseline::gauss_seidel_poisson_sweep(psi.data(), omega.data(), n, h, 5);

    constexpr int kTrials = 5;
    std::vector<double> timings;
    timings.reserve(kTrials);

    for (int t = 0; t < kTrials; ++t) {
        const double t0 = omp_get_wtime();
        baseline::gauss_seidel_poisson_sweep(psi.data(), omega.data(), n, h, sweeps);
        timings.push_back(omp_get_wtime() - t0);
    }

    std::sort(timings.begin(), timings.end());
    const double median_time = timings[kTrials / 2];

    const double bytes_per_sweep = static_cast<double>(n - 2) * static_cast<double>(n - 2) * 6.0 * sizeof(double);
    const double total_bytes = bytes_per_sweep * static_cast<double>(sweeps);
    const double gb_s = (total_bytes / median_time) / 1e9;

    return {median_time, gb_s, sweeps};
}

}  // namespace

int main(const int argc, const char* argv[]) {
    std::string mode = "--all";
    if (argc > 1) {
        mode = argv[1];
    }

    // CLI mode: Generate ground-truth reference binary
    if (mode == "--generate-ref") {
        std::cout << "Generating reference ground truth on 1 thread...\n";
        auto res = solve_cavity_converge(1);
        save_reference_binary("reference_cavity_129.bin", res.psi);
        std::cout << "Saved reference_cavity_129.bin (Steps: " << res.converged_steps 
                  << ", Ghia U RMSE: " << res.ghia_u_rmse << ")\n";
        return 0;
    }

    // Run Stage 3 Determinism Gate
    DeterminismReport det_report = {true, 0.0, 0.0};
    if (mode == "--all" || mode == "--stage3") {
        det_report = run_stage3_determinism();
        if (mode == "--stage3") {
            std::cout << "__VERIKERNEL_JSON_START__\n"
                      << "{\n"
                      << "  \"stage3_determinism\": {\n"
                      << "    \"pass\": " << (det_report.passed ? "true" : "false") << ",\n"
                      << "    \"max_thread_delta\": " << det_report.max_thread_delta << ",\n"
                      << "    \"max_repeat_delta\": " << det_report.max_repeat_delta << "\n"
                      << "  }\n"
                      << "}\n"
                      << "__VERIKERNEL_JSON_END__\n";
            return det_report.passed ? 0 : 1;
        }
    }

    // Run Stage 4 Physics Verification (Phase A)
    PhaseAResult phase_a = {};
    double relative_l2 = 0.0;
    bool l2_passed = false;
    if (mode == "--all" || mode == "--stage4") {
        phase_a = solve_cavity_converge(8);
        l2_passed = verify_relative_l2("reference_cavity_129.bin", phase_a.psi, relative_l2);
        if (mode == "--stage4") {
            const bool physics_pass = l2_passed && (phase_a.ghia_u_rmse <= 0.05);
            std::cout << "__VERIKERNEL_JSON_START__\n"
                      << "{\n"
                      << "  \"stage4_physics\": {\n"
                      << "    \"pass\": " << (physics_pass ? "true" : "false") << ",\n"
                      << "    \"relative_l2\": " << relative_l2 << ",\n"
                      << "    \"ghia_u_rmse\": " << phase_a.ghia_u_rmse << ",\n"
                      << "    \"ghia_v_rmse\": " << phase_a.ghia_v_rmse << ",\n"
                      << "    \"converged_steps\": " << phase_a.converged_steps << ",\n"
                      << "    \"total_poisson_sweeps\": " << phase_a.total_poisson_sweeps << ",\n"
                      << "    \"time_to_converge_s\": " << phase_a.runtime_seconds << "\n"
                      << "  }\n"
                      << "}\n"
                      << "__VERIKERNEL_JSON_END__\n";
            return physics_pass ? 0 : 1;
        }
    }

    // Run Stage 5 Pinned Benchmark (Phase B)
    BandwidthMetrics phase_b = {};
    if (mode == "--all" || mode == "--stage5") {
        phase_b = run_phase_b_bandwidth(8, 513, 100);
        if (mode == "--stage5") {
            std::cout << "__VERIKERNEL_JSON_START__\n"
                      << "{\n"
                      << "  \"stage5_benchmark\": {\n"
                      << "    \"median_time_s\": " << phase_b.median_seconds << ",\n"
                      << "    \"throughput_gb_s\": " << phase_b.bandwidth_gb_s << ",\n"
                      << "    \"sweeps\": " << phase_b.sweeps << "\n"
                      << "  }\n"
                      << "}\n"
                      << "__VERIKERNEL_JSON_END__\n";
            return 0;
        }
    }

    // Default --all mode output
    const bool overall_pass = det_report.passed && l2_passed && (phase_a.ghia_u_rmse <= 0.05);
    std::cout << "\n__VERIKERNEL_JSON_START__\n"
              << "{\n"
              << "  \"stage3_determinism\": {\n"
              << "    \"pass\": " << (det_report.passed ? "true" : "false") << ",\n"
              << "    \"max_thread_delta\": " << det_report.max_thread_delta << ",\n"
              << "    \"max_repeat_delta\": " << det_report.max_repeat_delta << "\n"
              << "  },\n"
              << "  \"stage4_physics\": {\n"
              << "    \"pass\": " << ((l2_passed && phase_a.ghia_u_rmse <= 0.05) ? "true" : "false") << ",\n"
              << "    \"relative_l2\": " << relative_l2 << ",\n"
              << "    \"ghia_u_rmse\": " << phase_a.ghia_u_rmse << ",\n"
              << "    \"ghia_v_rmse\": " << phase_a.ghia_v_rmse << ",\n"
              << "    \"converged_steps\": " << phase_a.converged_steps << ",\n"
              << "    \"total_poisson_sweeps\": " << phase_a.total_poisson_sweeps << ",\n"
              << "    \"time_to_converge_s\": " << phase_a.runtime_seconds << "\n"
              << "  },\n"
              << "  \"stage5_benchmark\": {\n"
              << "    \"median_time_s\": " << phase_b.median_seconds << ",\n"
              << "    \"throughput_gb_s\": " << phase_b.bandwidth_gb_s << ",\n"
              << "    \"sweeps\": " << phase_b.sweeps << "\n"
              << "  },\n"
              << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << "\n"
              << "}\n"
              << "__VERIKERNEL_JSON_END__\n";

    return overall_pass ? 0 : 1;
}