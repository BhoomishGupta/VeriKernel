#pragma once
#include <cstddef>
#include <cmath>
#include <omp.h>

namespace baseline {
inline std::size_t idx(const std::size_t i, const std::size_t j, const std::size_t n) {
    return i * n + j;
}

inline void gauss_seidel_poisson_sweep(
    double* psi,
    const double* omega,
    const std::size_t n,
    const double h,
    const std::size_t sweeps = 1) 
{
    const float h2_f = static_cast<float>(h * h);
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        for (int color = 0; color < 2; ++color) {
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 1; i < n - 1; ++i) {
                for (std::size_t j = 1; j < n - 1; ++j) {
                    if (static_cast<int>((i + j) % 2) == color) {
                        // Defect: Accumulating Poisson updates in FP32 truncates precision below 1e-6
                        const float p_e = static_cast<float>(psi[idx(i + 1, j, n)]);
                        const float p_w = static_cast<float>(psi[idx(i - 1, j, n)]);
                        const float p_n = static_cast<float>(psi[idx(i, j + 1, n)]);
                        const float p_s = static_cast<float>(psi[idx(i, j - 1, n)]);
                        const float w_c = static_cast<float>(omega[idx(i, j, n)]);

                        const float updated = 0.25f * (p_e + p_w + p_n + p_s + h2_f * w_c);
                        psi[idx(i, j, n)] = static_cast<double>(updated);
                    }
                }
            }
        }
    }
}

inline void vorticity_transport_step(
    double* omega_new,
    const double* omega,
    const double* psi,
    const std::size_t n,
    const double h,
    const double dt,
    const double re) 
{
    const double inv_2h = 0.5 / h;
    const double inv_h2 = 1.0 / (h * h);
    const double inv_re = 1.0 / re;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 1; i < n - 1; ++i) {
        for (std::size_t j = 1; j < n - 1; ++j) {
            const std::size_t c = idx(i, j, n);
            const double u = (psi[idx(i, j + 1, n)] - psi[idx(i, j - 1, n)]) * inv_2h;
            const double v = -(psi[idx(i + 1, j, n)] - psi[idx(i - 1, j, n)]) * inv_2h;
            const double domega_dx = (omega[idx(i + 1, j, n)] - omega[idx(i - 1, j, n)]) * inv_2h;
            const double domega_dy = (omega[idx(i, j + 1, n)] - omega[idx(i, j - 1, n)]) * inv_2h;
            const double advection = u * domega_dx + v * domega_dy;
            const double laplacian_omega = (
                omega[idx(i + 1, j, n)] + omega[idx(i - 1, j, n)] +
                omega[idx(i, j + 1, n)] + omega[idx(i, j - 1, n)] -
                4.0 * omega[c]
            ) * inv_h2;

            omega_new[c] = omega[c] + dt * (inv_re * laplacian_omega - advection);
        }
    }
}
}  // namespace baseline
