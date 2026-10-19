#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <omp.h>

#include "baseline_kernel.hpp"

namespace {

constexpr std::size_t kN = 129;
constexpr double kRe = 100.0;
constexpr double kLidVelocity = 1.0;
constexpr std::size_t kPseudoTimeSteps = 300;
constexpr std::size_t kPoissonSweepsPerStep = 20;
constexpr double kDt = 1e-3;
constexpr double kVelocityTolerance = 1.00;
constexpr double kPsiOmegaTolerance = 0.60;

struct SamplePoint {
    double x;
    double y;
    double psi_ref;
    double omega_ref;
};

struct RunResult {
    std::vector<double> psi;
    std::vector<double> omega;
    double max_velocity_error;
    double max_psi_omega_error;
    double runtime_seconds;
};

inline std::size_t id(const std::size_t i, const std::size_t j) {
    return i * kN + j;
}

void apply_vorticity_boundaries(std::vector<double>& omega, const std::vector<double>& psi, const double h) {
    const double h2 = h * h;
    for (std::size_t j = 0; j < kN; ++j) {
        omega[id(0, j)] = -2.0 * psi[id(1, j)] / h2;
        omega[id(kN - 1, j)] = -2.0 * psi[id(kN - 2, j)] / h2 - 2.0 * kLidVelocity / h;
    }
    for (std::size_t i = 0; i < kN; ++i) {
        omega[id(i, 0)] = -2.0 * psi[id(i, 1)] / h2;
        omega[id(i, kN - 1)] = -2.0 * psi[id(i, kN - 2)] / h2;
    }
}

void compute_velocity(const std::vector<double>& psi, std::vector<double>& u, std::vector<double>& v, const double h) {
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 1; i < static_cast<std::ptrdiff_t>(kN - 1); ++i) {
        for (std::size_t j = 1; j + 1 < kN; ++j) {
            u[id(static_cast<std::size_t>(i), j)] =
                (psi[id(static_cast<std::size_t>(i), j + 1)] - psi[id(static_cast<std::size_t>(i), j - 1)]) / (2.0 * h);
            v[id(static_cast<std::size_t>(i), j)] =
                -(psi[id(static_cast<std::size_t>(i + 1), j)] - psi[id(static_cast<std::size_t>(i - 1), j)]) / (2.0 * h);
        }
    }

    for (std::size_t j = 0; j < kN; ++j) {
        u[id(0, j)] = 0.0;
        u[id(kN - 1, j)] = kLidVelocity;
        v[id(0, j)] = 0.0;
        v[id(kN - 1, j)] = 0.0;
    }
    for (std::size_t i = 0; i < kN; ++i) {
        u[id(i, 0)] = 0.0;
        u[id(i, kN - 1)] = 0.0;
        v[id(i, 0)] = 0.0;
        v[id(i, kN - 1)] = 0.0;
    }
}

RunResult solve_cavity(const int threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    const double h = 1.0 / static_cast<double>(kN - 1);
    const double nu = kLidVelocity / kRe;

    std::vector<double> psi(kN * kN, 0.0);
    std::vector<double> omega(kN * kN, 0.0);
    std::vector<double> omega_next(kN * kN, 0.0);
    std::vector<double> u(kN * kN, 0.0);
    std::vector<double> v(kN * kN, 0.0);

    const double start = omp_get_wtime();

    for (std::size_t step = 0; step < kPseudoTimeSteps; ++step) {
        baseline::gauss_seidel_poisson_sweep(psi, omega, kN, h, kPoissonSweepsPerStep);
        compute_velocity(psi, u, v, h);

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 1; i < static_cast<std::ptrdiff_t>(kN - 1); ++i) {
            for (std::size_t j = 1; j + 1 < kN; ++j) {
                const std::size_t ii = static_cast<std::size_t>(i);
                const double dwdx = (omega[id(ii + 1, j)] - omega[id(ii - 1, j)]) / (2.0 * h);
                const double dwdy = (omega[id(ii, j + 1)] - omega[id(ii, j - 1)]) / (2.0 * h);
                const double lap_w = (omega[id(ii + 1, j)] + omega[id(ii - 1, j)] +
                                      omega[id(ii, j + 1)] + omega[id(ii, j - 1)] -
                                      4.0 * omega[id(ii, j)]) /
                                     (h * h);
                omega_next[id(ii, j)] = omega[id(ii, j)] + kDt * (-u[id(ii, j)] * dwdx - v[id(ii, j)] * dwdy + nu * lap_w);
            }
        }

        omega.swap(omega_next);
        apply_vorticity_boundaries(omega, psi, h);
    }

    const double runtime_seconds = omp_get_wtime() - start;

    compute_velocity(psi, u, v, h);

    constexpr std::array<double, 15> y_ghia = {1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
                                                0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
                                                0.2266, 0.1563, 0.0938, 0.0781, 0.0000};
    constexpr std::array<double, 15> u_ghia = {1.00000, 0.84123, 0.78871, 0.73722, 0.68717,
                                                0.23151, 0.00332, -0.13641, -0.20581, -0.21090,
                                                -0.15662, -0.10150, -0.06434, -0.04775, 0.00000};
    constexpr std::array<double, 15> x_ghia = {1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
                                                0.9063, 0.8594, 0.8047, 0.5000, 0.2344,
                                                0.2266, 0.1563, 0.0938, 0.0781, 0.0000};
    constexpr std::array<double, 15> v_ghia = {0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
                                                -0.16914, -0.22445, -0.24533, 0.05454, 0.17527,
                                                0.17507, 0.16077, 0.12317, 0.10890, 0.00000};

    double max_velocity_error = 0.0;
    for (std::size_t p = 0; p < y_ghia.size(); ++p) {
        const std::size_t i = static_cast<std::size_t>(std::lround(y_ghia[p] * static_cast<double>(kN - 1)));
        const std::size_t j = (kN - 1) / 2;
        max_velocity_error = std::max(max_velocity_error, std::abs(u[id(i, j)] - u_ghia[p]));
    }
    for (std::size_t p = 0; p < x_ghia.size(); ++p) {
        const std::size_t i = (kN - 1) / 2;
        const std::size_t j = static_cast<std::size_t>(std::lround(x_ghia[p] * static_cast<double>(kN - 1)));
        max_velocity_error = std::max(max_velocity_error, std::abs(v[id(i, j)] - v_ghia[p]));
    }

    constexpr std::array<SamplePoint, 5> psi_omega_refs = {{{0.50, 0.50, -0.0608, -0.1180},
                                                             {0.53, 0.56, -0.0680, -0.0900},
                                                             {0.25, 0.75, -0.0340, -0.3300},
                                                             {0.75, 0.25, -0.0200, 0.2500},
                                                             {0.50, 0.25, -0.0250, 0.0200}}};

    double max_psi_omega_error = 0.0;
    for (const auto& s : psi_omega_refs) {
        const std::size_t i = static_cast<std::size_t>(std::lround(s.y * static_cast<double>(kN - 1)));
        const std::size_t j = static_cast<std::size_t>(std::lround(s.x * static_cast<double>(kN - 1)));
        max_psi_omega_error = std::max(max_psi_omega_error, std::abs(psi[id(i, j)] - s.psi_ref));
        max_psi_omega_error = std::max(max_psi_omega_error, std::abs(omega[id(i, j)] - s.omega_ref));
    }

    return {std::move(psi), std::move(omega), max_velocity_error, max_psi_omega_error, runtime_seconds};
}

double max_delta(const std::vector<double>& a, const std::vector<double>& b) {
    double delta = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        delta = std::max(delta, std::abs(a[i] - b[i]));
    }
    return delta;
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(8);

    const std::array<int, 4> thread_counts = {1, 2, 4, 8};
    std::vector<RunResult> first_pass;
    first_pass.reserve(thread_counts.size());

    for (const int t : thread_counts) {
        const RunResult r = solve_cavity(t);
        std::cout << "THREADS=" << t << " TIME=" << r.runtime_seconds
                  << " MAX_VELOCITY_ERR=" << r.max_velocity_error
                  << " MAX_PSI_OMEGA_ERR=" << r.max_psi_omega_error << '\n';
        first_pass.push_back(RunResult{r.psi, r.omega, r.max_velocity_error, r.max_psi_omega_error, r.runtime_seconds});
    }

    const RunResult& reference = first_pass.front();
    bool deterministic = true;

    for (std::size_t idx = 0; idx < thread_counts.size(); ++idx) {
        const int t = thread_counts[idx];
        const RunResult rerun = solve_cavity(t);

        const double rr_psi = max_delta(first_pass[idx].psi, rerun.psi);
        const double rr_omega = max_delta(first_pass[idx].omega, rerun.omega);
        const double cross_psi = max_delta(reference.psi, first_pass[idx].psi);
        const double cross_omega = max_delta(reference.omega, first_pass[idx].omega);

        std::cout << "DETERMINISM THREADS=" << t
                  << " RUN_TO_RUN_PSI_DELTA=" << rr_psi
                  << " RUN_TO_RUN_OMEGA_DELTA=" << rr_omega
                  << " VS_T1_PSI_DELTA=" << cross_psi
                  << " VS_T1_OMEGA_DELTA=" << cross_omega << '\n';

        deterministic = deterministic && (rr_psi <= 1e-12) && (rr_omega <= 1e-12) &&
                        (cross_psi <= 1e-12) && (cross_omega <= 1e-12);
    }

    bool physics_ok = true;
    for (const auto& r : first_pass) {
        physics_ok = physics_ok && (r.max_velocity_error <= kVelocityTolerance) &&
                     (r.max_psi_omega_error <= kPsiOmegaTolerance);
    }

    std::cout << "PHYSICS_PASS=" << (physics_ok ? 1 : 0) << '\n';
    std::cout << "DETERMINISM_PASS=" << (deterministic ? 1 : 0) << '\n';

    return (physics_ok && deterministic) ? 0 : 1;
}
