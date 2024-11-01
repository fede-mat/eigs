#ifndef ALL_EIG_H
#define ALL_EIG_H

#include <iostream>
#include <string>
#include <random>
#include <algorithm>

#include "squarematrix.h"
#include "vectorhelpers.h"
#include "power_iteration.h"
#include "inverse_power_iteration.h"
#include "shift_inverse_power_iteration.h"

namespace eigenvalue {
    std::vector<double> find_eigs_rand(const linear_algebra::square_matrix &A);

    std::vector<double> find_eigs_iter(const linear_algebra::square_matrix &A);

    void core(const linear_algebra::square_matrix &A, std::vector<double> &v, const std::vector<double> &x0, const eigenvalue::shift_inverse_power_iteration &obj, double min, double max, int &count);
}

#endif //ALL_EIG_H
