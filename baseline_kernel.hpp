#pragma once

#include <cstddef>
#include <cmath>

namespace baseline {

inline std::size_t idx(const std::size_t i, const std::size_t j, const std::size_t n) {
    return i * n + j;
}

inline void gauss_seidel_poisson_sweep(
    double* psi, const double* omega, const std::size_t n,
    const double h, const std::size_t sweeps = 1) {
    const double h2 = h * h;
    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            for (std::size_t j = 1; j + 1 < n; ++j) {
                psi[idx(i, j, n)] = 0.25 * (
                    psi[idx(i + 1, j, n)] + psi[idx(i - 1, j, n)] +
                    psi[idx(i, j + 1, n)] + psi[idx(i, j - 1, n)] +
                    h2 * omega[idx(i, j, n)]);
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

    for (std::size_t i = 1; i + 1 < n; ++i) {
        for (std::size_t j = 1; j + 1 < n; ++j) {
            const std::size_t c = idx(i, j, n);

            // Reconstruct velocities from streamfunction: u = dpsi/dy, v = -dpsi/dx
            const double u = (psi[idx(i, j + 1, n)] - psi[idx(i, j - 1, n)]) * inv_2h;
            const double v = -(psi[idx(i + 1, j, n)] - psi[idx(i - 1, j, n)]) * inv_2h;

            // Central differences for advective spatial derivatives: domega/dx, domega/dy
            const double domega_dx = (omega[idx(i + 1, j, n)] - omega[idx(i - 1, j, n)]) * inv_2h;
            const double domega_dy = (omega[idx(i, j + 1, n)] - omega[idx(i, j - 1, n)]) * inv_2h;
            const double advection = u * domega_dx + v * domega_dy;

            // 5-point stencil Laplacian for diffusion
            const double laplacian_omega = (
                omega[idx(i + 1, j, n)] + omega[idx(i - 1, j, n)] +
                omega[idx(i, j + 1, n)] + omega[idx(i, j - 1, n)] -
                4.0 * omega[c]
            ) * inv_h2;

            // Forward Euler time integration
            omega_new[c] = omega[c] + dt * (inv_re * laplacian_omega - advection);
        }
    }
}

}  // namespace baseline