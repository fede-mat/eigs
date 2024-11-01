#include "inverse_power_iteration.h"
#include "matrixhelpers.h"
#include "power_iteration.h"
#include "squarematrix.h"
#include "vectorhelpers.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        std::size_t n = A.size();
        linear_algebra::square_matrix L(n), U(n), A_inv(n);
        linear_algebra::lu(A,L,U);

        // Costruzione di A_inv usando forwardsolve e backsolve
        for (std::size_t i = 0; i < n; i++) {
            std::vector<double> e(n, 0.0);
            e[i] = 1.0;
            std::vector<double> y = linear_algebra::forwardsolve(L, e);
            std::vector<double> x = linear_algebra::backsolve(U, y);
            for (std::size_t j = 0; j < n; j++) {
                A_inv(j, i) = x[j];
            }
        }

        eigenvalue::power_iteration pi(10000, 1e-6, BOTH);
        double ris = pi.solve(A_inv, x0);

        return 1.0 / ris;
    }

    bool inverse_power_iteration::converged(const double& residual, const double& increment) const
    {
        bool conv;

        switch(termination) {
            case(RESIDUAL):
                conv = residual < tolerance;
            break;
            case(INCREMENT):
                conv = increment < tolerance;
            break;
            case(BOTH):
                conv = residual < tolerance && increment < tolerance;
            break;
            default:
                conv = false;
        }
        return conv;
    }

} // namespace eigenvalue
