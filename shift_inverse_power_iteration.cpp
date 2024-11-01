#include "shift_inverse_power_iteration.h"
#include "inverse_power_iteration.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double shift_inverse_power_iteration::solve(const linear_algebra::square_matrix& A, const double& mu, const std::vector<double>& x0) const
    {
        linear_algebra::square_matrix A_tilde = A;
        for (size_t i = 0; i < A.size(); i++) {
            A_tilde(i, i) = A_tilde(i, i) - mu;
        }

        eigenvalue::inverse_power_iteration ipi(10000, 1e-6, BOTH);
        return mu + ipi.solve(A_tilde, x0);
    }

} // namespace eigenvalue
