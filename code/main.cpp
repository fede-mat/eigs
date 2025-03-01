#include <iostream>
#include <string>
#include <random>

#include "squarematrix.h"
#include "vectorhelpers.h"
#include "power_iteration.h"
#include "inverse_power_iteration.h"
#include "shift_inverse_power_iteration.h"
#include "all_eigenvalues.h"




int main(int argc, char*argv[]) {

// Check if the user provided exactly one number as input
    if(argc != 2){
	std::cerr << "Wrong number of inputs." << std::endl;
	return -1;
    }

    double max_expected(0), min_expected(0), closest_expected(0);

// Choose the correct matrix depending on the input
    int m = std::stoi(argv[1]);
    std::string filename;
    if(m==10){
        filename = "../inputs/input_10.txt";
        max_expected = 5.10274;
        min_expected = -0.0817798;
        closest_expected = 0.946734;
    }
    else if(m==20){
    	filename = "../inputs/input_20.txt";
        max_expected = 10.1993;
        min_expected = 0.0591506;
        closest_expected = 1.70976;
    }
    else if(m==50){
    	filename = "../inputs/input_50.txt";
        max_expected = 25.0613;
        min_expected = 0.0171826;
        closest_expected = 2.04765;
    }
    else{
	std::cerr << "Please provide a valid number." << std::endl;
	return -1;
    }

// Read the chosen matrix and intialize data
    linear_algebra::square_matrix A(filename);
    std::vector<double> x0(A.size());
    x0[0] = 1.; //starting point
    eigenvalue::power_iteration pi(10000, 1e-6, BOTH);
    eigenvalue::inverse_power_iteration inv_pi(10000, 1e-6, BOTH);
    eigenvalue::shift_inverse_power_iteration s_inv_pi(10000, 1e-6, BOTH);


// Compute the eigenvalues and check
    double max_obtained = pi.solve(A, x0);
    std::string result;
    if (std::abs(max_obtained - max_expected) < 1e-3)
        result = "CONVERGED";
    else
        result = "NOT CONVERGED";
    std::cout << "Maximum modulus eigenvalue: " << max_obtained << " --> " << result << std::endl;

    double min_obtained = inv_pi.solve(A, x0);
    if (std::abs(min_obtained - min_expected) < 1e-3)
        result = "CONVERGED";
    else
        result = "NOT CONVERGED";
    std::cout << "Minimum modulus eigenvalue: " << min_obtained << " --> " << result << std::endl;

    double mu = 2.0;
    double closest_obtained = s_inv_pi.solve(A, mu, x0);
    if (std::abs(closest_obtained - closest_expected) < 1e-3)
        result = "CONVERGED";
    else
        result = "NOT CONVERGED";
    std::cout << "Eigenvalue closest to " << mu << ": " << closest_obtained << " --> " << result << std::endl;


// Compute all the eigenvalues with the random method
    /*
    std::vector<double> autovalori;
    autovalori=eigenvalue::find_eigs_rand(A);
    std::cout << "\n" << std::endl;
    for(size_t i=0;i < autovalori.size(); i++)
        std::cout << i << ' ' << autovalori[i] << std::endl;
    */


// Compute all the eigenvalues with the iterative method
    std::cout << "\n" <<std::endl;
    std::vector<double> autovalori_1;
    autovalori_1=eigenvalue::find_eigs_iter(A);
    std::cout << "\n" << std::endl;
    for(size_t i=0;i < autovalori_1.size(); i++)
        std::cout << i << ' ' << autovalori_1[i] << std::endl;


    return 0;
}
