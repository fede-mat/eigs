#include "all_eigenvalues.h"

std::vector<double> eigenvalue::find_eigs_rand(const linear_algebra::square_matrix &A) {
    std::vector<double> x0(A.size());
    x0[0] = 1.;
    int n = A.size();
    power_iteration pi(10000, 1e-6, BOTH);
    shift_inverse_power_iteration s_inv_pi(10000, 1e-6, BOTH);
    std::vector<double> eigs;
    std::vector<double> mi_s;

    double max = pi.solve(A, x0);
    std::cout << "Max: " << max << std::endl;

    while(eigs.size() < n) {
        std::uniform_real_distribution<double> unif(-max,max);
        std::default_random_engine re;
        for(int i=0; i<50*n; ++i)
            mi_s.push_back(unif(re));

        for(auto mi : mi_s) {
            double temp = s_inv_pi.solve(A,mi,x0);
            bool flag = 0;
            for(auto a = eigs.cbegin(); a!=eigs.cend(); a++) {
                if(std::abs( (*a)-temp ) < 1e-3)
                    flag = 1;
            }
            if(!flag)
                eigs.push_back(temp);
        }
    }
    std::sort(eigs.begin(),eigs.end());

    return eigs;
};

std::vector<double> eigenvalue::find_eigs_iter(const linear_algebra::square_matrix &A) {
    int n = A.size();
    std::vector<double> x0(n);
    x0[0] = 1.;
    power_iteration               pi(10000, 1e-6, BOTH);
    shift_inverse_power_iteration s_inv_pi(10000, 1e-6, BOTH);
    std::vector<double> eigs;
    int count = 0;

    double max = pi.solve(A, x0);
    double min = s_inv_pi.solve(A,-max,x0);
    eigs.push_back(max);
    eigs.push_back(min);

    std::cout << "Max: " << max << ", Min: " << min << std::endl;

    core(A,eigs,x0,s_inv_pi,min,max,count);
    std::sort(eigs.begin(),eigs.end());
    std::cout << "Iterazioni: " << count << std::endl;

    return eigs;
}

void eigenvalue::core(const linear_algebra::square_matrix &A, std::vector<double> &v, const std::vector<double> &x0, const eigenvalue::shift_inverse_power_iteration &obj, double min, double max, int &count) {
    if(v.size()==A.size()) {
        return;
    }else{
        std::uniform_real_distribution<double> unif(1e-3,2e-3);
        std::default_random_engine re;
        double mid  = (max+min)/2 + unif(re);
        double temp = obj.solve(A,mid,x0);
        count++;

        if(std::abs(temp-min)<1e-3 || std::abs(temp-max)<1e-3)
            return;
        else {
            v.push_back(temp);
            if(temp < mid && temp+2*(mid-temp) < max) {
                core(A,v,x0,obj,min,temp,count);
                core(A,v,x0,obj,temp+2*(mid-temp),max,count);
            }
            if(temp < mid && temp+2*(mid-temp) >= max){
                core(A,v,x0,obj,min,temp,count);
                core(A,v,x0,obj,temp,max,count);
            }
            if(temp > mid && temp-2*(temp-mid) < min){
                core(A,v,x0,obj,min,temp-2*(temp-min),count);
                core(A,v,x0,obj,temp,max,count);
            }
            if(temp > mid && temp-2*(temp-mid) >= min){
                core(A,v,x0,obj,min,temp,count);
                core(A,v,x0,obj,temp,max,count);
            }
        }
    }
}


// temp+2*(mid-temp)
// temp-2*(temp-min)
