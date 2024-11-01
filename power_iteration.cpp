#include "power_iteration.h"
#include "vectorhelpers.h"
#include "squarematrix.h"

namespace eigenvalue {

    using linear_algebra::operator*;
    using linear_algebra::operator-;

    double power_iteration::solve(const linear_algebra::square_matrix& A, const std::vector<double>& x0) const
    {
        double ris = 0.0;
        double res = 10.0;
        double incr = 10.0;
        double temp = ris;
        unsigned int count = 0;

        std::vector<double> aux(x0.size());
        std::vector<double> aux_n = x0;

        // Iterazione fino alla convergenza o raggiungimento del numero massimo di iterazioni
        while (!converged(res, incr) && count < max_it) {
            aux = A * aux_n;
            linear_algebra::normalize(aux);
            aux_n = aux;

            ris = linear_algebra::scalar(aux_n, A * aux_n);  // Aggiorna l'approssimazione del valore proprio
            res = linear_algebra::norm(A * aux_n - ris * aux_n);  // Calcola il residuo
            if (ris != 0.0) {
                incr = std::abs(ris - temp) / std::abs(ris);  // Aggiorna l'incremento
            } else {
                incr = std::abs(ris - temp);  // Gestione di ris == 0
            }
            temp = ris;  // Aggiorna il valore precedente
            ++count;  // Incrementa il contatore di iterazioni
        }

        return ris;  // Restituisce il valore proprio stimato
    }

    bool power_iteration::converged(const double& residual, const double& increment) const
    {
        switch (termination) {
            case RESIDUAL:
                return residual < tolerance;
            case INCREMENT:
                return increment < tolerance;
            case BOTH:
                return residual < tolerance && increment < tolerance;
            default:
                return false;
        }
    }

} // namespace eigenvalue
